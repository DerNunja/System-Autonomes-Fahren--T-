# sim_vehicle_state_publisher.py
import csv
import json
import time
import paho.mqtt.client as mqtt

# --- Konfiguration ---
BROKER = "localhost"
CSV_FILE = r"C:\Users\firas\Documents\Semester 5\Programmieren mobiler Systeme\run1\run1\recording_2025_11_06__12_06_16.csv"  # <-- hier deinen Dateinamen anpassen
TOPIC_VEHICLE_STATE = "sensor/vehicle_state"

# Wenn du "Realtime-Replay" willst, kannst du RATE_HZ ignorieren
# und stattdessen die timestamp-Differenzen verwenden.
RATE_HZ = 10   # 10 Hz als Standard-Abspielrate

def main():
    client = mqtt.Client(client_id="sim-vehicle-state-pub")
    client.connect(BROKER, 1883, keepalive=60)
    print(f"Verbunden mit MQTT-Broker {BROKER}, starte CSV-Replay...")

    dt = 1.0 / RATE_HZ

    with open(CSV_FILE, newline="") as f:
        reader = csv.DictReader(f)

        # Optional: falls du auf Basis von timestamp abspielen willst:
        # prev_ts = None

        for row in reader:
            # --- Felder robust aus der CSV holen ---

            def ffloat(name, default=0.0):
                val = row.get(name, "")
                try:
                    return float(val)
                except Exception:
                    return default

            msg = {
                "timestamp": ffloat("timestamp", 0.0),
                "throttle": ffloat("throttle", 0.0),
                "brakes": ffloat("brakes", 0.0),
                "wheel_position": ffloat("wheel_position", 0.0),
                "engine_rpm": ffloat("engine_rpm", 0.0),
                "brakes_vol": ffloat("brakes_vol", 0.0),

                # alles, was du nur "roh" mitgeben willst:
                "raw": {
                    "cruiseControl_mask_objects_adas": row.get("cruiseControl{1}.mask_objects_adas"),
                    "car0_shift_up": row.get("car0{1}.shift_up"),
                    "car0_shift_down": row.get("car0{1}.shift_down"),
                    # hier kannst du beliebig weitere Spalten reinpacken
                    # "rrp_pos": row.get("rrp_pos"),
                    # "rrp_lin_vel": row.get("rrp_lin_vel"),
                    # ...
                }
            }

            # --- Publish über MQTT ---
            payload = json.dumps(msg)
            client.publish(TOPIC_VEHICLE_STATE, payload, qos=1)

            # kleine Ausgabe zu Debug-Zwecken:
            # print("Pub sensor/vehicle_state:", payload)

            # --- Variante A: feste Abspielrate ---
            time.sleep(dt)

            # --- Variante B: echte Zeit aus timestamp (wenn timestamp in Sekunden ist) ---
            # ts = msg["timestamp"]
            # if prev_ts is not None:
            #     wait = ts - prev_ts
            #     if wait > 0:
            #         time.sleep(wait)
            # prev_ts = ts

    client.disconnect()
    print("CSV-Replay beendet, MQTT getrennt.")

if __name__ == "__main__":
    main()
