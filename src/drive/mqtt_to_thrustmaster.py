"""
============================================================
  MQTT → THRUSTMASTER FORCE FEEDBACK LENKSTEUERUNG
============================================================

  Empfängt Wahrnehmungsdaten via MQTT, berechnet daraus lokal
  Lenkbefehle und dreht das Thrustmaster-Lenkrad physisch per
  Force Feedback. Der Simulator liest dann ganz normal die echte
  Radposition aus.

  Warum dieser Ansatz statt virtuellem Xbox-Controller:
  → Simulatoren akzeptieren oft nur das direkt angeschlossene
    Lenkrad als Eingabe, kein virtuelles Gamepad.
  → Das physische Rad dreht sich → Simulator sieht echte
    Achsenbewegung → funktioniert mit jedem Simulator.

──────────────────────────────────────────────────────────
  VORAUSSETZUNGEN
──────────────────────────────────────────────────────────

  1. Thrustmaster Treiber installieren (Windows)
     → https://www.thrustmaster.com/de/support/
     → Neu starten nach Installation!

  2. Autocenter im Thrustmaster Control Panel auf 0% setzen
     → Sonst kämpft der Motor gegen unsere Befehle

  3. Python Pakete:
     pip install pysdl2 pysdl2-dll paho-mqtt

  4. BROKER unten eintragen (IP des MQTT-Brokers)

──────────────────────────────────────────────────────────
"""

import json
import time
import ctypes
import threading
from dataclasses import dataclass, field

import paho.mqtt.client as mqtt

try:
    from .drive_controller import DriveController, DriveControllerResult
except ImportError:
    try:
        from drive.drive_controller import DriveController, DriveControllerResult
    except ImportError:
        from drive_controller import DriveController, DriveControllerResult

try:
    import sdl2
    import sdl2.ext
except ImportError:
    raise SystemExit("FEHLER: pip install pysdl2 pysdl2-dll")


# ── Konfiguration ──────────────────────────────────────────────────────────

BROKER           = "localhost"       # IP des MQTT-Brokers (z.B. "192.168.1.42")
PORT             = 1883
TOPIC_LANESTATE  = "sensor/lanestate"
TOPIC_CMD        = "control/steering_cmd"

JOYSTICK_INDEX   = 0        # Index des Thrustmaster (0 = erstes Gerät)

# Regler (Position → Drehmoment)
KP               = 0.35     # Proportional – niedrig halten gegen Aufschaukeln!
KD               = 0.08     # Dämpfung – reduziert Überschwingen
MAX_TORQUE       = 0.5      # Maximales Drehmoment 0.0–1.0  (Sicherheitslimit!)
FFB_DIRECTION    = +1       # +1 oder -1. Wird beim Start automatisch geprüft.
                            # Falls Rad in falsche Richtung pusht → auf -1 setzen.

# Eingang (MQTT)
STEER_GAIN       = 1.0      # Skalierung des Eingangssignals
STEER_DEADZONE   = 0.03     # Totzone gegen Rauschen
EMA_ALPHA        = 0.25     # Glättung des Zielsignals (0=glatt, 1=direkt)
CMD_TIMEOUT_S    = 0.25     # Sekunden ohne MQTT → Rad zentrieren
MIN_LANE_QUALITY = 0.3      # Mindestqualität der Ego-Lane

# Lateralregler (Perception-State → Soll-Lenkwinkel)
MAX_STEER_RAD    = 0.5
K_STANLEY        = 1.0
V_REF            = 20.0
K_FF             = 8.0
HISTORY_WINDOW_S = 0.5

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

def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── Shared State (thread-safe über Lock) ───────────────────────────────────

@dataclass
class State:
    target_angle : float = 0.0   # Soll-Position aus steer_norm (-1..+1)
    steer_ema    : float = 0.0   # Geglätteter Zielwert
    last_msg_t   : float = 0.0   # Zeitstempel letzte MQTT-Nachricht
    last_error   : float = 0.0   # Für D-Anteil
    last_print_t : float = 0.0
    current_angle: float = 0.0   # Aktueller Radwinkel -1..+1
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
    # SDL_PollEvent MUSS aufgerufen werden damit SDL_JoystickUpdate
    # tatsächlich neue Werte liefert – ohne das bleibt der Wert eingefroren!
    event = sdl2.SDL_Event()
    while sdl2.SDL_PollEvent(ctypes.byref(event)):
        pass
    sdl2.SDL_JoystickUpdate()
    return sdl2.SDL_JoystickGetAxis(joystick, 0) / 32767.0


def test_ffb_direction(joystick, haptic):
    """
    Sendet kurz ein kleines positives Drehmoment und prüft ob das Rad
    sich nach rechts bewegt. Falls nicht → FFB_DIRECTION ist invertiert.
    """
    global FFB_DIRECTION
    print(f"\n[TEST] Prüfe FFB-Richtung ...")
    print(f"  Sende Drehmoment +0.30 nach rechts für 1.0s ...")

    start_angle = get_angle(joystick)
    print(f"  Startwinkel: {start_angle:+.3f}")

    # Drehmoment kontinuierlich anwenden (nicht nur einmal setzen!)
    test_torque = 0.30
    end_time = time.time() + 1.0
    max_delta = 0.0
    last_angle = start_angle

    while time.time() < end_time:
        set_torque(haptic, test_torque)
        angle = get_angle(joystick)
        delta = angle - start_angle
        if abs(delta) > abs(max_delta):
            max_delta = delta
        last_angle = angle
        time.sleep(0.02)

    end_angle = last_angle
    set_torque(haptic, 0.0)

    delta = end_angle - start_angle
    print(f"  Endwinkel:   {end_angle:+.3f}   (Δ = {delta:+.3f}, max |Δ| = {abs(max_delta):.3f})")

    if abs(max_delta) < 0.02:
        print(f"  ⚠ Rad hat sich kaum bewegt. Mögliche Ursachen:")
        print(f"    → Autocenter im Thrustmaster Control Panel noch aktiv")
        print(f"    → Rad steht am physischen Anschlag")
        print(f"    → FFB-Treiber liefert keine Kraft (Logitech/Thrustmaster Software prüfen)")
        time.sleep(1.5)
        return

    if max_delta > 0:
        print(f"  ✓ Rad bewegt sich korrekt nach rechts (FFB_DIRECTION = +1)")
    else:
        print(f"  ⚠ Rad bewegt sich nach LINKS bei positivem Drehmoment!")
        print(f"  → FFB ist invertiert. Setze FFB_DIRECTION = -1 automatisch.")
        FFB_DIRECTION = -1

    time.sleep(0.5)


def homing(joystick, haptic):
    """
    Zentriert das Lenkrad sanft bevor der Regler startet.
    Nutzt set_torque() damit FFB_DIRECTION berücksichtigt wird.
    """
    print(f"\n[HOMING] Zentriere Lenkrad ...")
    KP_HOME      = 0.4
    TORQUE_LIMIT = 0.3
    THRESHOLD    = 0.05   # Gilt als zentriert wenn |Winkel| < 5%
    TIMEOUT      = 5.0    # Max. Sekunden für Homing

    start = time.time()

    while time.time() - start < TIMEOUT:
        angle = get_angle(joystick)

        if abs(angle) < THRESHOLD:
            set_torque(haptic, 0.0)
            print(f"\n[HOMING] Zentriert ✓  (Winkel: {angle:+.3f})")
            time.sleep(0.3)
            return

        torque = clamp(-KP_HOME * angle, -TORQUE_LIMIT, TORQUE_LIMIT)
        set_torque(haptic, torque)

        print(f"\r[HOMING] Winkel: {angle:+.3f}  Torque: {torque:+.3f}  ", end="", flush=True)
        time.sleep(0.02)

    set_torque(haptic, 0.0)
    print(f"\n[HOMING] Timeout – Rad steht bei {get_angle(joystick):+.3f} (weiter ...)")


_effect_id = -1
_effect    = None   # wiederverwendbares Effect-Struct

def set_torque(haptic, torque: float):
    """
    Setzt das Drehmoment als Constant-Force-Effekt.
    Verwendet HapticUpdateEffect statt destroy/create für stabilere
    Treiber-Kommunikation (besonders wichtig bei Thrustmaster, die das
    rapide Neu-Erstellen oft nicht sauber verarbeiten).
    """
    global _effect_id, _effect

    torque = clamp(torque, -MAX_TORQUE, MAX_TORQUE)
    level  = int(torque * FFB_DIRECTION * 32767)

    # Effekt-Struct beim ersten Mal anlegen
    if _effect is None:
        _effect = sdl2.SDL_HapticEffect()
        _effect.type = sdl2.SDL_HAPTIC_CONSTANT
        _effect.constant.direction.type   = sdl2.SDL_HAPTIC_CARTESIAN
        _effect.constant.direction.dir[0] = 1
        _effect.constant.length           = sdl2.SDL_HAPTIC_INFINITY
        _effect.constant.delay            = 0
        _effect.constant.button           = 0
        _effect.constant.interval         = 0
        _effect.constant.attack_length    = 0
        _effect.constant.attack_level     = 0
        _effect.constant.fade_length      = 0
        _effect.constant.fade_level       = 0

    _effect.constant.level = level

    # Beim ersten Aufruf hochladen, danach nur noch updaten
    if _effect_id < 0:
        _effect_id = sdl2.SDL_HapticNewEffect(haptic, ctypes.byref(_effect))
        if _effect_id >= 0:
            sdl2.SDL_HapticRunEffect(haptic, _effect_id, sdl2.SDL_HAPTIC_INFINITY)
        else:
            err = sdl2.SDL_GetError()
            print(f"\n[FFB] NewEffect Fehler: {err}")
    else:
        result = sdl2.SDL_HapticUpdateEffect(haptic, _effect_id, ctypes.byref(_effect))
        if result < 0:
            err = sdl2.SDL_GetError()
            print(f"\n[FFB] UpdateEffect Fehler: {err}")


def stop_all(haptic):
    global _effect_id, _effect
    if _effect_id >= 0:
        sdl2.SDL_HapticStopEffect(haptic, _effect_id)
        sdl2.SDL_HapticDestroyEffect(haptic, _effect_id)
    _effect_id = -1
    _effect = None


# ── MQTT Callbacks ─────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"[MQTT] Verbunden (rc={reason_code})")
    client.subscribe(TOPIC_LANESTATE)
    print(f"[MQTT] Topic: {TOPIC_LANESTATE}")

def build_control_payload(result: DriveControllerResult, source_payload: dict, wheel_norm: float, target_norm: float) -> dict:
    cmd = result.command
    return {
        "t_ms": result.t_ms,
        "valid": result.valid,
        "reason": result.reason,
        "quality": float(result.quality),
        "steer_rad": cmd.steer_rad,
        "steer_norm": cmd.steer_norm,
        "ff_term": cmd.ff_term,
        "offset_m": cmd.error_offset_m,
        "heading_err_rad": safe_float(source_payload.get("heading_error_rad")),
        "curvature": safe_float(source_payload.get("curvature_preview")),
        "wheel_norm": float(wheel_norm),
        "target_norm": float(target_norm),
        "wheel_error_norm": float(target_norm - wheel_norm),
    }

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception as e:
        print(f"[MQTT] Ungültiges JSON: {e}")
        return

    drive_controller = userdata["drive_controller"]
    result = drive_controller.update_from_lanestate(payload, t=time.time())

    raw = float(result.command.steer_norm)
    scaled = clamp(raw * STEER_GAIN, -1.0, 1.0)
    dz = apply_deadzone(scaled, STEER_DEADZONE)

    now = time.time()
    with state.lock:
        state.steer_ema = ema(state.steer_ema, dz, EMA_ALPHA)
        state.target_angle = state.steer_ema
        state.last_msg_t = now
        wheel_norm = state.current_angle
        target_norm = state.target_angle

    control_payload = build_control_payload(result, payload, wheel_norm, target_norm)
    client.publish(TOPIC_CMD, json.dumps(control_payload))

    if not result.valid:
        return


# ── PD-Regelschleife ───────────────────────────────────────────────────────

def control_loop(joystick, haptic):
    """
    Läuft mit CONTROL_HZ.
    Steuert das Rad zur Soll-Position (aus sensor/lanestate) per PD-Regler.
    """
    dt = 1.0 / CONTROL_HZ
    print(f"[REGLER] @ {CONTROL_HZ} Hz  |  KP={KP}  KD={KD}  MAX={MAX_TORQUE:.0%}")
    print(f"{'─'*55}")

    while True:
        now = time.time()

        with state.lock:
            target   = state.target_angle
            last_t   = state.last_msg_t
            last_err = state.last_error

        # Timeout: kein MQTT seit CMD_TIMEOUT_S → sanft zentrieren
        if last_t > 0 and (now - last_t) > CMD_TIMEOUT_S:
            with state.lock:
                state.steer_ema    = ema(state.steer_ema, 0.0, 0.3)
                state.target_angle = state.steer_ema
                target             = state.target_angle

        current = get_angle(joystick)
        error   = target - current

        d_error = (error - last_err) / dt
        torque  = KP * error + KD * d_error

        set_torque(haptic, torque)

        with state.lock:
            state.last_error = error
            state.current_angle = current

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

    drive_controller = DriveController(
        max_steer_rad=MAX_STEER_RAD,
        k_stanley=K_STANLEY,
        v_ref=V_REF,
        k_ff=K_FF,
        history_window_s=HISTORY_WINDOW_S,
        min_quality=MIN_LANE_QUALITY,
    )

    # MQTT in eigenem Thread
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="mqtt-to-ffb")
    client.user_data_set({"drive_controller": drive_controller})
    client.on_connect = on_connect
    client.on_message = on_message
    print(f"[MQTT] Verbinde mit {BROKER}:{PORT} ...")
    client.connect(BROKER, PORT, keepalive=30)
    client.loop_start()

    # Erst FFB-Richtung testen, dann zentrieren, dann Regler starten
    test_ffb_direction(joystick, haptic)
    homing(joystick, haptic)

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
