import cv2
from ultralytics import YOLO

model = YOLO(r"C:\Users\Miral Ibrahim\OneDrive\Desktop\Hs\Fahrsimulator\System-Autonomes-Fahren--T-\Behavioural_Cloning_Basic\schilder\gewichte\yolov8_weights.pt")

results = model.predict(
    source=r"C:\Users\Miral Ibrahim\OneDrive\Desktop\Hs\Fahrsimulator\data\2025-11-13 08-18-38.mp4",
    conf=0.25,
    imgsz=640,
    device="cpu",
    save=True
)




#cap = cv2.VideoCapture("path/to/video.mp4")

#while cap.isOpened():
#    ret, frame = cap.read()
#    if not ret:
#        break
#    if 
#
#    results = model(frame)
#
#    annotated_frame = results[0].plot()
#    cv2.imshow("YOLO Detection", annotated_frame)
#
#    if cv2.waitKey(1) & 0xFF == ord("q"):
#        break

#cap.release()
#cv2.destroyAllWindows()

