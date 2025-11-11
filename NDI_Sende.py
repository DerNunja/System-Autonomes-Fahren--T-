# ndi_sender_sim.py
import NDIlib as ndi
import cv2
import numpy as np
import time

VIDEO_PATH = "dein_video_vom_prof.mp4" # <-- HIER  DATEi EINFÜGEN
SOURCE_NAME = "Simu_Video_Feed"

def run_sender():
    if not ndi.initialize():
        print("NDI kann nicht initialisiert werden.")
        return

    # 1. Video laden
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Fehler: Video {VIDEO_PATH} konnte nicht geöffnet werden.")
        ndi.destroy()
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 # falls nicht verfügbar
    frame_delay = 1.0 / fps

    # 2. NDI Sender initialisieren
    send_settings = ndi.SendCreateSettings(application_name=SOURCE_NAME)
    ndi_sender = ndi.SendCreate(send_settings)

    if ndi_sender is None:
        print("Fehler beim Erstellen des NDI-Senders.")
        ndi.destroy()
        return

    print(f"NDI Sender '{SOURCE_NAME}' gestartet. Sende Frames ({width}x{height} @ {fps:.2f} FPS)...")

    # 3. Frame-Loop: Video lesen und senden
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read() # Frame von OpenCV lesen
        
        if not ret:
            print("Ende des Videos erreicht. Loope das Video.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Video neu starten
            continue

        # WICHTIG: OpenCV nutzt BGR, NDI erwartet oft BGRA oder RGB. Wir konvertieren zu BGRA
        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        # 4. NDI VideoFrameV2 erstellen und befüllen
        ndi_frame = ndi.VideoFrameV2()
        ndi_frame.data = frame_bgra
        ndi_frame.xres = width
        ndi_frame.yres = height
        ndi_frame.FourCC = ndi.FOURCC_video_type_BGRA # Format setzen

        # 5. Frame senden
        ndi.SendSendVideoV2(ndi_sender, ndi_frame)
        
        # 6. Wartezeit für korrekte FPS
        elapsed_time = time.time() - start_time
        sleep_time = frame_delay - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Aufräumen
    cap.release()
    ndi.SendDestroy(ndi_sender)
    ndi.destroy()
    print("Sender beendet.")

if __name__ == "__main__":
    run_sender()

