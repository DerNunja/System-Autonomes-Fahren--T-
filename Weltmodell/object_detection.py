import cv2 as cv
from ultralytics import YOLO
import os
from datetime import datetime

# bei model bitte pfad zu Gewichte eingeben

model = r"C:\Users\Zayd Maatouf\Documents\5 Semester\runs\detect\train11\weights\best.pt"
source = r"C:\Users\Zayd Maatouf\Downloads\2026-03-25 10-22-03.mp4"

def live_YOLO(model_path, source, gpu: bool, skip_frame: int):
    model = YOLO(model_path)

    if gpu:
        model.to("cuda")

    frame_count = 0
    cap = cv.VideoCapture(source)

    paused = False

    # Ordner für Screenshots
    save_dir = "screenshots"
    os.makedirs(save_dir, exist_ok=True)

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame_count += 1

            if frame_count % skip_frame != 0:
                display_frame = frame
            else:
                results = model(frame, conf=0.5, verbose=False)
                display_frame = results[0].plot()

            cv.imshow("frame", display_frame)

        key = cv.waitKey(30) & 0xFF

        if key == ord("q"):
            print("Programm beendet")
            break

        elif key == ord("p"):
            paused = not paused
            print("Pause:", paused)

        elif key == ord("s"):
            # Screenshot speichern
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{save_dir}/frame_{frame_count}_{timestamp}.png"
            cv.imwrite(filename, display_frame)
            print(f"Screenshot gespeichert: {filename}")

    cap.release()
    cv.destroyAllWindows()
    
    
def yolo_(model, source, frame_num: int):
    model = YOLO(model)
    cap = cv.VideoCapture(source)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if frame_count % frame_num:
            vorhersage = model(frame)          # vorhersagen
            print("Frame Anzahl:" %frame_count)


        if 0xFF == ord("q"):    # mit taste q ausschalten
            break

    
    
    cap.release()




live_YOLO(model, source=source, gpu=True, skip_frame=1)
