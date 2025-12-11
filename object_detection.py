# detect_objects.py
import cv2
from ultralytics import YOLO
from typing import List, Dict, Any

def run_yolo(image_path: str, model_path: str = "yolov12s.pt", conf_thresh: float = 0.25) -> List[Dict[str, Any]]:
    """
    Run YOLOv12 on an image and return detections in Weltmodell format.
    Each detection: {'box':[x1,y1,x2,y2], 'score':float, 'class_name':str}
    """
    model = YOLO(model_path)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = model.predict(img_rgb, conf=conf_thresh, verbose=False)
    detections = []

    for r in results:
        if not hasattr(r, 'boxes') or r.boxes is None:
            continue
        for b in r.boxes:
            try:
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                score = float(b.conf[0])
                class_name = str(b.cls[0])  # numeric by default
                if hasattr(b, 'names'):
                    class_name = b.names[int(b.cls[0])]
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'score': score,
                    'class_name': class_name
                })
            except Exception:
                continue
    return detections

if __name__ == "__main__":
    # simple CLI test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov12s.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    dets = run_yolo(args.image, args.model, args.conf)
    print(f"Detections ({len(dets)}):")
    for d in dets:
        print(d)
