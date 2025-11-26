# mock_perception_publisher.py
import json
import time
import paho.mqtt.client as mqtt

BROKER = "localhost"
RATE_HZ = 5  # 5 Mal pro Sekunde die Ergebnisse publizieren

def main():
    client = mqtt.Client(client_id="mock-perception-pub")
    try:
        client.connect(BROKER, 1883, keepalive=60)
    except Exception as e:
        print(f"Fehler bei Verbindung zum Broker: {e}")
        return

    print("Mock Perception Client gestartet. Publiziert simulierte Ergebnisse...")
    
    dt = 1.0 / RATE_HZ
    
    try:
        while True:
            # --- Simulierte Wahrnehmungsdaten ---
            
            # 1. Objekte (z.B. ein Auto in 15m und eine Ampel)
            objects_data = [
                {"type": "car", "distance": 15.0 + 2 * (time.time() % 3), "speed": 10.0},
                {"type": "traffic_light", "status": "green", "distance": 50.0}
            ]
            
            # 2. Fahrspur (gerade und leicht mittig versetzt)
            lanestate_data = {
                "lane_center": 0.1,  # Leicht rechts von der Mitte
                "curvature": 0.0001
            }

            # --- MQTT-Publikation ---
            client.publish("sensor/objects", json.dumps(objects_data), qos=1)
            client.publish("sensor/lanestate", json.dumps(lanestate_data), qos=1)
            
            # print("Simulierte Daten gesendet.")

            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\nMock Perception beendet.")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()