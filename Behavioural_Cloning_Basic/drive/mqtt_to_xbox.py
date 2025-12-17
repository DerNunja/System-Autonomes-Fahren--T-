import json
import time
import math
from dataclasses import dataclass
from typing import Optional

import paho.mqtt.client as mqtt

# Windows: virtueller Xbox-Controller
import vgamepad as vg

# ----------------- Config -----------------
BROKER = ""          # <- Simulator-PC: lokal; oder IP vom Broker
PORT = 1883
TOPIC = "control/steering_cmd"

DEBUG_PRINT = True
DEBUG_PRINT_HZ = 10.0   # max. Prints pro Sekunde

# Stick-Mapping
# steer_norm: -1..+1  -> LeftStickX: -1..+1
STEER_GAIN = 1.0              # Skalierung (z.B. 0.7 = sanfter)
STEER_DEADZONE = 0.03         # Totzone gegen Zittern
STEER_MAX = 1.0               # Clamp

# Smoothing / Safety
EMA_ALPHA = 0.25              # 0..1 (kleiner = glatter, größer = direkter)
CMD_TIMEOUT_S = 0.25          # wenn länger kein Cmd: Stick zentrieren

# Optional: Falls du später Gas/Bremse mappen willst (Trigger 0..1)
ENABLE_THROTTLE = False
THROTTLE_TOPIC_FIELD = "throttle_norm"   # falls du das später sendest
BRAKE_TOPIC_FIELD = "brake_norm"

# ------------------------------------------

@dataclass
class State:
    last_msg_t: float = 0.0
    steer_ema: float = 0.0
    last_print_t: float = 0.0

state = State()

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def apply_deadzone(x: float, dz: float) -> float:
    if abs(x) < dz:
        return 0.0
    # optional: rescale außerhalb der Totzone (damit 0..1 voll genutzt wird)
    s = 1.0 if x > 0 else -1.0
    x2 = (abs(x) - dz) / (1.0 - dz)
    return s * x2

def ema(prev: float, new: float, alpha: float) -> float:
    return (1.0 - alpha) * prev + alpha * new

# Virtuellen Xbox-Controller erzeugen
gamepad = vg.VX360Gamepad()

def set_steer(x_norm: float):
    # vgamepad erwartet float -1..+1
    gamepad.left_joystick_float(x_value_float=x_norm, y_value_float=0.0)
    gamepad.update()

def set_triggers(throttle: float, brake: float):
    # 0..1
    throttle = clamp(throttle, 0.0, 1.0)
    brake = clamp(brake, 0.0, 1.0)
    gamepad.right_trigger_float(throttle)
    gamepad.left_trigger_float(brake)
    gamepad.update()

def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"[MQTT] connected rc={reason_code}")
    client.subscribe(TOPIC)
    print(f"[MQTT] subscribed: {TOPIC}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception as e:
        print("[MQTT] bad json:", e)
        return

    if "steer_norm" not in payload:
        return

    steer_raw = float(payload["steer_norm"])

    # Gain + Clamp
    steer_clamped = clamp(steer_raw * STEER_GAIN, -STEER_MAX, STEER_MAX)

    # Deadzone
    steer_dz = apply_deadzone(steer_clamped, STEER_DEADZONE)

    # EMA
    state.steer_ema = ema(state.steer_ema, steer_dz, EMA_ALPHA)

    # Anwenden auf Controller
    set_steer(state.steer_ema)

    # ---------- DEBUG PRINT ----------
    if DEBUG_PRINT:
        now = time.time()
        if now - state.last_print_t > 1.0 / DEBUG_PRINT_HZ:
            print(
                f"[STEER] raw={steer_raw:+.3f}  "
                f"clamp={steer_clamped:+.3f}  "
                f"deadzone={steer_dz:+.3f}  "
                f"EMA={state.steer_ema:+.3f}"
            )
            state.last_print_t = now
    # ---------------------------------

    state.last_msg_t = time.time()

def main():
    # Paho “Callback API v2” vermeiden wir, indem wir die neuen Signaturen zulassen (properties-Param etc.)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="mqtt-to-xbox")
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[MQTT] connecting to {BROKER}:{PORT} ...")
    client.connect(BROKER, PORT, keepalive=30)
    client.loop_start()

    print("[RUN] forwarding MQTT steering -> virtual Xbox left stick")
    try:
        while True:
            now = time.time()
            if state.last_msg_t > 0 and (now - state.last_msg_t) > CMD_TIMEOUT_S:
                # Safety: wenn keine Befehle mehr kommen -> neutral
                if abs(state.steer_ema) > 1e-3:
                    state.steer_ema = ema(state.steer_ema, 0.0, 0.5)
                    set_steer(state.steer_ema)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        # neutral beim Exit
        set_steer(0.0)
        if ENABLE_THROTTLE:
            set_triggers(0.0, 0.0)
        client.loop_stop()
        client.disconnect()
        print("[EXIT]")

if __name__ == "__main__":
    main()
