import cv2 as cv
from ultralytics import YOLO
import os
from datetime import datetime
import time
import uuid
from typing import Dict, Any, List, Optional


# =========================
# 🌍 WELTMODELL
# =========================
class Weltmodell:

    def __init__(self, lane_detector: Optional[Any] = None):
        self.ego = {
            "position": (0.0, 0.0),
            "geschwindigkeit": 0.0,
            "richtung": 0.0,
            "lenkwinkel": 0.0
        }

        self.schilder: Dict[str, Dict[str, Any]] = {}
        self.ampeln: Dict[str, Dict[str, Any]] = {}

        self.lane_info: Dict[str, Any] = {
            'curvature_m': None,
            'lateral_offset_m': None,
            'confidence': 0.0,
            'left_fit': None,
            'right_fit': None,
            'last_update': None
        }

        self.lane_detector = lane_detector

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def add_schild(self, position, label: str, score: float = 1.0):
        obj_id = self._new_id("schild")
        self.schilder[obj_id] = {
            'id': obj_id,
            'label': label,
            'position': position,
            'score': score,
            'timestamp': time.time()
        }

    def add_ampel(self, position, state: str, score: float = 1.0):
        obj_id = self._new_id("ampel")
        self.ampeln[obj_id] = {
            'id': obj_id,
            'state': state,
            'position': position,
            'score': score,
            'timestamp': time.time()
        }

    def update_from_vision(self, detections: List[Dict[str, Any]]):
        for d in detections:
            cname = d.get('class_name', '').lower()
            x1, y1, x2, y2 = d.get('box', [0, 0, 0, 0])

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            position = (cx, cy)
            score = d.get('score', 0.0)

            if 'light' in cname or 'ampel' in cname:
                state = 'unknown'
                if 'red' in cname:
                    state = 'red'
                elif 'green' in cname:
                    state = 'green'
                elif 'yellow' in cname:
                    state = 'yellow'

                self.add_ampel(position, state, score)

            elif 'sign' in cname or 'speed' in cname:
                self.add_schild(position, cname, score)

    def print_state(self):
        print("\n=== 🌍 Weltmodell ===")
        print(f"Schilder: {len(self.schilder)} | Ampeln: {len(self.ampeln)}")


# =========================
# 🔄 YOLO → DETECTIONS
# =========================
def yolo_to_detections(results):
    detections = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = r.names[cls_id]

            detections.append({
                "box": [x1, y1, x2, y2],
                "score": score,
                "class_name": class_name
            })

    return detections


# =========================
# 🎥 LIVE YOLO + WELTMODELL
# =========================
def live_YOLO(model_path, source, gpu=True, skip_frame=1):
    model = YOLO(model_path)

    if gpu:
        model.to("cuda")

    weltmodell = Weltmodell()

    cap = cv.VideoCapture(source)
    frame_count = 0
    paused = False

    # Screenshot folder
    save_dir = "screenshots"
    os.makedirs(save_dir, exist_ok=True)

    while cap.isOpened():

        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended")
                break

            frame_count += 1

            if frame_count % skip_frame != 0:
                display_frame = frame
            else:
                results = model(frame, conf=0.5, verbose=False)

                # 🔥 Update Weltmodell
                detections = yolo_to_detections(results)
                weltmodell.update_from_vision(detections)

                # Debug output every 30 frames
                if frame_count % 30 == 0:
                    weltmodell.print_state()

                display_frame = results[0].plot()

            cv.imshow("YOLO Live", display_frame)

        key = cv.waitKey(30) & 0xFF

        if key == ord("q"):
            print("Exit")
            break

        elif key == ord("p"):
            paused = not paused
            print("Paused:", paused)

        elif key == ord("s"):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{save_dir}/frame_{frame_count}_{timestamp}.png"
            cv.imwrite(filename, display_frame)
            print(f"Saved: {filename}")

    cap.release()
    cv.destroyAllWindows()


# =========================
# 🚀 RUN
# =========================
model_path = r"C:\Users\Zayd Maatouf\Documents\5 Semester\runs\detect\train11\weights\best.pt"
source = r"C:\Users\Zayd Maatouf\Downloads\2026-03-25 10-22-03.mp4"

live_YOLO(model_path, source, gpu=True, skip_frame=1)
