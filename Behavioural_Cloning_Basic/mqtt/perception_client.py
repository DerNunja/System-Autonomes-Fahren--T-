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

    print("Mock Perception Client gestartet. Publiziert NUR simulierte OBJECT-Daten...")
    
    dt = 1.0 / RATE_HZ
    
    try:
        while True:
            # --- Simulierte Wahrnehmungsdaten (Objekte) ---

            objects_data = [
                {
                    "type": "traffic_sign",
                    "sign_type": "speed_limit",
                    "value": 50,          # z.B. 50 km/h
                    "distance": 25.0
                },
                {
                    "type": "traffic_light",
                    "status": "green",
                    "distance": 50.0
                }
            ]

            # NUR NOCH OBJECTS publizieren
            client.publish("sensor/objects", json.dumps(objects_data), qos=1)

            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\nMock Perception beendet.")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()
