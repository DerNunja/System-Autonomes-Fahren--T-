# lane_detection_client.py

import json
import time
import cv2
import numpy as np
import paho.mqtt.client as mqtt

# HIER anpassen: Modulname, in dem dein init_lanedetector & process_frame liegen
from lanedetec_runner import init_lanedetector, process_frame
# zb.from lanedetector import init_lanedetector, process_frame
#init_lanedetector initialisiert einmalig das Lane-Detection-System (Modell, Gewichte, Konfiguration, Device und Preprocessing).
#Dadurch kann process_frame effizient für jedes Videobild ausgeführt werden.


BROKER = "localhost"
TOPIC_LANESTATE = "sensor/lanestate"
RATE_HZ = 10  # Publikationsfrequenz


def lanes_to_lanestate(lanes_xy, model_w, lane_width_m=3.7):
    """
    Schätzt lane_center (in Metern) und curvature (Dummy) aus den erkannten Lanes.

    lanes_xy: Liste von Lanes, jede Lane = Liste von (x_model, y_canon)
    model_w:  Breite des Modell-Eingangs (cfg.train_width)
    lane_width_m: angenommene Spurbreite (ca. 3.7m)
    """

    # Mindestens 2 Lanes nötig (linke & rechte Spurbegrenzung)
    if len(lanes_xy) < 2:
        return 0.0, 0.0

    lane_a = lanes_xy[0]
    lane_b = lanes_xy[1]

    def bottom_point(lane):
        if not lane:
            return None
        # Punkt mit größter y-Koordinate = am nächsten zur Kamera
        return max(lane, key=lambda p: p[1])

    pa = bottom_point(lane_a)
    pb = bottom_point(lane_b)
    if pa is None or pb is None:
        return 0.0, 0.0

    x_a, _ = pa
    x_b, _ = pb

    # sortieren: links / rechts
    x_left, x_right = sorted([x_a, x_b])

    # Spurmitte in Modellpix koord.
    lane_center_x = 0.5 * (x_left + x_right)

    # Fahrzeugmitte = Bildmitte
    img_center_x = model_w / 2.0

    offset_px = lane_center_x - img_center_x
    lane_width_px = max(1e-3, abs(x_right - x_left))

    meters_per_px = lane_width_m / lane_width_px
    lane_center_m = offset_px * meters_per_px

    curvature = 0.0  # vorerst Dummy

    return lane_center_m, curvature


def main():
    # --- MQTT vorbereiten ---
    client = mqtt.Client(client_id="lane-detection-client")
    client.connect(BROKER, 1883, keepalive=60)
    print("Lane Detection Client: mit Broker verbunden")

    # --- Lane-Net initialisieren ---
    net, cfg, img_transforms, device = init_lanedetector()
    print("Lane-Net initialisiert.")

    # --- Videoquelle ---
    # TODO: HIER an die NDI-Quelle anpassen.
    # Platzhalter: Webcam
    cap = cv2.VideoCapture(0)

    dt = 1.0 / RATE_HZ

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("Kein Frame erhalten, warte...")
                time.sleep(0.1)
                continue

            # Lane-Detection
            vis, lanes_xy, lanes_info = process_frame(
                frame_bgr, net, cfg, img_transforms, device
            )

            # lane_center / curvature schätzen
            lane_center_m, curvature = lanes_to_lanestate(
                lanes_xy,
                model_w=cfg.train_width,
                lane_width_m=3.7
            )

            # Nachricht bauen
            lanestate_msg = {
                "lane_center": float(lane_center_m),
                "curvature": float(curvature)
            }

            # Per MQTT publishen
            client.publish(TOPIC_LANESTATE, json.dumps(lanestate_msg), qos=1)

            # Debug-Bild mit Lanes anzeigen
            cv2.imshow("Lane Detection", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC zum Beenden
                break

            time.sleep(dt)

    except KeyboardInterrupt:
        print("Lane Detection Client beendet (KeyboardInterrupt).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.disconnect()

if __name__ == "__main__":
    main()
