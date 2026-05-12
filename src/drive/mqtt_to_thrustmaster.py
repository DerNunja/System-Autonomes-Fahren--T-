import json
import time
import ctypes
import threading
from dataclasses import dataclass, field

import paho.mqtt.client as mqtt

try:
    import sdl2
    import sdl2.ext
except ImportError:
    raise SystemExit("FEHLER: pip install pysdl2 pysdl2-dll")


# ── Konfiguration ──────────────────────────────────────────────────────────

BROKER           = ""       # IP des MQTT-Brokers (z.B. "192.168.1.42")
PORT             = 1883
TOPIC            = "control/steering_cmd"

JOYSTICK_INDEX   = 0        # Index des Thrustmaster (0 = erstes Gerät)

# Regler
KP               = 0.8      # Proportionalverstärkung  ← wichtigster Tuning-Parameter
KD               = 0.05     # Dämpfung (reduziert Überschwingen)
MAX_TORQUE       = 0.6      # Maximales Drehmoment 0.0–1.0  (Sicherheitslimit!)

# Eingang (MQTT)
STEER_GAIN       = 1.0      # Skalierung des Eingangssignals
STEER_DEADZONE   = 0.03     # Totzone gegen Rauschen
EMA_ALPHA        = 0.25     # Glättung des Zielsignals (0=glatt, 1=direkt)
CMD_TIMEOUT_S    = 0.25     # Sekunden ohne MQTT → Rad zentrieren

# Regelschleife
CONTROL_HZ       = 100      # Regler-Frequenz in Hz
EFFECT_MS        = 25       # FF-Effektdauer (≥ 1000/CONTROL_HZ)

# Debug
DEBUG_HZ         = 10.0     # Max. Ausgaben pro Sekunde


# ── Hilfsfunktionen ────────────────────────────────────────────────────────

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def apply_deadzone(x: float, dz: float) -> float:
    if abs(x) < dz:
        return 0.0
    s = 1.0 if x > 0 else -1.0
    return s * (abs(x) - dz) / (1.0 - dz)

def ema(prev: float, new: float, alpha: float) -> float:
    return (1.0 - alpha) * prev + alpha * new


# ── Shared State (thread-safe über Lock) ───────────────────────────────────

@dataclass
class State:
    target_angle : float = 0.0   # Soll-Winkel vom Assistenten (-1..+1)
    steer_ema    : float = 0.0   # Geglätteter Zielwert
    last_msg_t   : float = 0.0   # Zeitstempel letzte MQTT-Nachricht
    last_error   : float = 0.0   # Für D-Anteil
    last_print_t : float = 0.0
    lock         : threading.Lock = field(default_factory=threading.Lock)

state = State()


# ── SDL2 / Thrustmaster ────────────────────────────────────────────────────

def init_wheel():
    sdl2.SDL_Init(sdl2.SDL_INIT_JOYSTICK | sdl2.SDL_INIT_HAPTIC)

    num = sdl2.SDL_NumJoysticks()
    print(f"\n{'='*55}")
    print(f"  Erkannte Geräte: {num}")
    print(f"{'='*55}")
    for i in range(num):
        n = sdl2.SDL_JoystickNameForIndex(i)
        print(f"  [{i}] {n.decode() if n else 'Unbekannt'}")
    print(f"{'='*55}")

    joystick = sdl2.SDL_JoystickOpen(JOYSTICK_INDEX)
    if not joystick:
        raise SystemExit(f"FEHLER: Gerät [{JOYSTICK_INDEX}] nicht öffenbar")

    name = sdl2.SDL_JoystickName(joystick)
    name = name.decode() if name else "Unbekannt"
    print(f"\n  Verwende: [{JOYSTICK_INDEX}] {name}")

    haptic = sdl2.SDL_HapticOpenFromJoystick(joystick)
    if not haptic:
        raise SystemExit("FEHLER: Kein Force Feedback gefunden! Treiber installiert?")

    caps = sdl2.SDL_HapticQuery(haptic)
    if not (caps & sdl2.SDL_HAPTIC_CONSTANT):
        raise SystemExit("FEHLER: Constant Force nicht unterstützt! Falsches Gerät?")

    sdl2.SDL_HapticSetAutocenter(haptic, 0)
    print(f"  Autocenter deaktiviert ✓")
    print(f"  Constant Force verfügbar ✓\n")

    return joystick, haptic, name


def get_angle(joystick) -> float:
    """Aktueller Lenkwinkel: -1.0 (links) bis +1.0 (rechts)"""
    sdl2.SDL_JoystickUpdate()
    return sdl2.SDL_JoystickGetAxis(joystick, 0) / 32767.0


_effect_id = -1

def set_torque(haptic, torque: float):
    """Sendet Constant-Force-Effekt. Ersetzt vorherigen Effekt."""
    global _effect_id

    torque = clamp(torque, -MAX_TORQUE, MAX_TORQUE)
    level  = int(torque * 32767)

    effect = sdl2.SDL_HapticEffect()
    effect.type = sdl2.SDL_HAPTIC_CONSTANT
    effect.constant.direction.type = sdl2.SDL_HAPTIC_CARTESIAN
    effect.constant.direction.dir[0] = 1
    effect.constant.length           = EFFECT_MS * 2
    effect.constant.level            = level
    effect.constant.attack_length    = 0
    effect.constant.attack_level     = abs(level)
    effect.constant.fade_length      = 0
    effect.constant.fade_level       = abs(level)

    # Alten Effekt löschen
    if _effect_id >= 0:
        sdl2.SDL_HapticStopEffect(haptic, _effect_id)
        sdl2.SDL_HapticDestroyEffect(haptic, _effect_id)

    _effect_id = sdl2.SDL_HapticNewEffect(haptic, ctypes.byref(effect))
    if _effect_id >= 0:
        sdl2.SDL_HapticRunEffect(haptic, _effect_id, 1)


def stop_all(haptic):
    global _effect_id
    sdl2.SDL_HapticStopAll(haptic)
    _effect_id = -1


# ── MQTT Callbacks ─────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"[MQTT] Verbunden (rc={reason_code})")
    client.subscribe(TOPIC)
    print(f"[MQTT] Topic: {TOPIC}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception as e:
        print(f"[MQTT] Ungültiges JSON: {e}")
        return

    if "steer_norm" not in payload:
        return

    raw = float(payload["steer_norm"])
    scaled = clamp(raw * STEER_GAIN, -1.0, 1.0)
    dz     = apply_deadzone(scaled, STEER_DEADZONE)

    with state.lock:
        state.steer_ema  = ema(state.steer_ema, dz, EMA_ALPHA)
        state.target_angle = state.steer_ema
        state.last_msg_t = time.time()


# ── PD-Regelschleife ───────────────────────────────────────────────────────

def control_loop(joystick, haptic):
    """
    Läuft mit CONTROL_HZ.
    Berechnet aus Soll/Ist-Winkel das nötige Drehmoment (PD-Regler)
    und schickt es als Constant-Force-Effekt ans Lenkrad.
    """
    dt = 1.0 / CONTROL_HZ
    print(f"[REGLER] Gestartet @ {CONTROL_HZ} Hz  |  KP={KP}  KD={KD}  MAX={MAX_TORQUE:.0%}")
    print(f"{'─'*55}")
    print(f"  {'Soll':>8}  {'Ist':>8}  {'Fehler':>8}  {'Drehm.':>8}")
    print(f"{'─'*55}")

    while True:
        now = time.time()

        with state.lock:
            target    = state.target_angle
            last_t    = state.last_msg_t
            last_err  = state.last_error

        # Timeout: kein MQTT seit CMD_TIMEOUT_S → sanft zentrieren
        if last_t > 0 and (now - last_t) > CMD_TIMEOUT_S:
            with state.lock:
                state.steer_ema    = ema(state.steer_ema, 0.0, 0.3)
                state.target_angle = state.steer_ema
                target             = state.target_angle

        current = get_angle(joystick)
        error   = target - current

        # PD
        d_error  = (error - last_err) / dt
        torque   = KP * error + KD * d_error

        set_torque(haptic, torque)

        with state.lock:
            state.last_error = error

        # Debug-Ausgabe (gedrosselt)
        if now - state.last_print_t > 1.0 / DEBUG_HZ:
            bar_pos = int((current + 1) / 2 * 30)
            bar = "·" * bar_pos + "●" + "·" * (30 - bar_pos)
            print(
                f"\r  [{bar}]"
                f"  Soll:{target:+.3f}"
                f"  Ist:{current:+.3f}"
                f"  Err:{error:+.3f}"
                f"  Trq:{torque:+.3f}  ",
                end="", flush=True
            )
            state.last_print_t = now

        time.sleep(dt)


# ── Einstiegspunkt ─────────────────────────────────────────────────────────

def main():
    if not BROKER:
        raise SystemExit("FEHLER: BROKER-IP am Anfang des Skripts eintragen!")

    joystick, haptic, name = init_wheel()

    # MQTT in eigenem Thread
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="mqtt-to-ffb")
    client.on_connect = on_connect
    client.on_message = on_message
    print(f"[MQTT] Verbinde mit {BROKER}:{PORT} ...")
    client.connect(BROKER, PORT, keepalive=30)
    client.loop_start()

    try:
        control_loop(joystick, haptic)
    except KeyboardInterrupt:
        print("\n\n[EXIT] Stoppe ...")
    finally:
        stop_all(haptic)
        set_torque(haptic, 0.0)
        client.loop_stop()
        client.disconnect()
        sdl2.SDL_HapticClose(haptic)
        sdl2.SDL_JoystickClose(joystick)
        sdl2.SDL_Quit()
        print("[EXIT] Fertig.")


if __name__ == "__main__":
    main()