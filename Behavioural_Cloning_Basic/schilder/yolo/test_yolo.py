import cv2 as cv
from ultralytics import YOLO

# bei model bitte pfad zu Gewichte eingeben

model = r"C:\Users\Miral Ibrahim\OneDrive\Desktop\Hs\Fahrsimulator\System-Autonomes-Fahren--T-\Behavioural_Cloning_Basic\schilder\gewichte\yolov8_weights.pt"
source = r"C:\Users\Miral Ibrahim\OneDrive\Desktop\Hs\Fahrsimulator\data\2025-11-13 08-18-38.mp4"

def live_YOLO(model, source, gpu: bool, skip_frame:int):
    model = YOLO(model)
    if gpu:
        model.to("cuda")
    
    frame_count = 0
    skip_frames = skip_frame  
    
    cap = cv.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        frame_count += 1
        
        # Nur jedes N-te Frame verarbeiten
        if frame_count % skip_frames != 0:
            cv.imshow("frame", frame)  # Zeige Original
        else:
            # Frame verarbeiten
            results = model(frame, conf=0.5, verbose=False)     #verbose= False macht die Terminal ausgaben weg
            annotated_frame = results[0].plot()
            cv.imshow("frame", annotated_frame)
        
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    
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




#live_YOLO(model, source=source, gpu=False, skip_frame=15)

model = YOLO(model)
model(source="https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Zeichen_274-60_-_Zul%C3%A4ssige_H%C3%B6chstgeschwindigkeit%2C_StVO_2017.svg/960px-Zeichen_274-60_-_Zul%C3%A4ssige_H%C3%B6chstgeschwindigkeit%2C_StVO_2017.svg.png", show=True, save=True)