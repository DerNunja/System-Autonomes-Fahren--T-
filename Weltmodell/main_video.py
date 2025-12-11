import cv2
from Weltmodell import Weltmodell
from object_detection import run_yolo

VIDEO_PATH = "simulation.mp4"
MODEL_PATH = "yolov12s.pt"
CONF_THRESH = 0.25

def main():
    wm = Weltmodell()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = run_yolo(frame_rgb, MODEL_PATH, CONF_THRESH)
        wm.update_from_vision(detections)
        if frame_count % 10 == 0:
            print(f"\n--- Frame {frame_count} ---")
            wm.print_state()

    cap.release()
    print("[INFO] Video processing complete.")

if __name__ == "__main__":
    main()
