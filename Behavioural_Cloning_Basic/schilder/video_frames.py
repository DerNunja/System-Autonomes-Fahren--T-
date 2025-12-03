import cv2
import os

def bilder_extra(path: str, BilderMenge: int, frame_num: int, output_ordner: str):
    if not os.path.exists(output_ordner):
        os.makedirs(output_ordner)
    
    cap = cv2.VideoCapture(path)
    frame_count = 0
    bilder = 0
    
    while bilder <= BilderMenge:
        ret, frame = cap.read()
        if not ret:  # Video zu Ende
            break
        
        if frame_count % frame_num == 0:
            frame_name = os.path.join(output_ordner, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_name, frame)
            bilder += 1
        
        frame_count += 1
    
    cap.release()
    print(f"{bilder} Bilder gespeichert in '{output_ordner}'")


# Ausgabe Position des schildes in x-y-koordinaten + Abstand
# Art des Schildes
# wünsch durchgezogene oder gestrichelte linien
# Ausgabe egal
# 60 FPS
# 19.12 projekt , paper februar
# Züge bis Ende Februar Hausarbeit


# Mask R-CNN, 