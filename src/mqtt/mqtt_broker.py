# mqtt_bridge.py
import json
import time
import paho.mqtt.client as mqtt

# --- Konfiguration ---
BROKER = "localhost"
RATE_HZ = 10  # Frequenz, mit der das Weltmodell publiziert wird (10 Hz)
TOPIC_WORLD_STATE = "world/state"
TOPIC_PERCEPTION_BASE = "sensor/#" # Abonniert alle Perception-Ergebnisse

# Weltmodell (Die interne Datenstruktur) ---
# Enthält nur die Daten
world = {
    "objects": [],
    "lanestate": {"lane_center": 0.0, "curvature": 0.0},
    "vehicle_state": {},              # <--- NEU
    "last_update_ts": time.time()
}


# --- Callback: Eingehende MQTT-Nachrichten verarbeiten ---
def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
    except Exception:
        print(f"Fehler beim Parsen von JSON in {msg.topic}")
        return

    if msg.topic == "sensor/objects":
        if isinstance(data, list):
            world["objects"] = data

    elif msg.topic == "sensor/lanestate":
        world["lanestate"].update({k: v for k, v in data.items() if k in ("lane_center", "curvature")})

    elif msg.topic == "sensor/vehicle_state":      # <--- NEU
        if isinstance(data, dict):
            world["vehicle_state"] = data

    world["last_update_ts"] = time.time()



# --- Haupt-Loop und Veröffentlichung ---
def main():
    client = mqtt.Client(client_id="bridge-data-hub")
    client.on_message = on_message
    
    try:
        client.connect(BROKER, 1883, keepalive=60)
        
        # Abonniert alle Topics unter sensor/
        print(f"Verbinde zu Broker {BROKER}. Abonniere {TOPIC_PERCEPTION_BASE}...")
        client.subscribe(TOPIC_PERCEPTION_BASE, qos=1) 
        
        client.loop_start() # Startet den Empfangs-Loop im Hintergrund

        dt = 1.0 / RATE_HZ
        
        while True:
            # ---  Weltmodell publizieren (IHR HAUPT-OUTPUT) ---
            # Sendet das gesamte aggregierte Weltmodell
            world_json = json.dumps(world)
            client.publish(TOPIC_WORLD_STATE, world_json, qos=0, retain=False)
            
            # print(f"Published {TOPIC_WORLD_STATE}: {world_json[:50]}...") # Nur zur Kontrolle

            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\nMQTT Bridge wird beendet...")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()