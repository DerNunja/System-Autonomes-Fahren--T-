# ndi_receiver_mock.py
import NDIlib as ndi
import cv2
import numpy as np
import time

SOURCE_NAME = "Simu_Video_Feed"

def run_receiver():
    if not ndi.initialize():
        print("NDI kann nicht initialisiert werden.")
        return

    # 1. Quelle finden
    finder = ndi.FindCreateV2()
    ndi_source = None
    
    print(f"Suche nach NDI-Quelle '{SOURCE_NAME}'...")
    while ndi_source is None:
        time.sleep(0.5)
        sources = ndi.FindGetCurrentSources(finder)
        for src in sources:
            if src.p_ndi_name.find(SOURCE_NAME) != -1:
                ndi_source = src
                print("Quelle gefunden!")
                break
    
    ndi.FindDestroy(finder)

    # 2. Verbindung zur Quelle herstellen
    recv_settings = ndi.RecvCreateV3()
    ndi_receiver = ndi.RecvCreateV3(recv_settings)
    ndi.RecvConnect(ndi_receiver, ndi_source)

    print("NDI Receiver gestartet. Empfange Frames...")

    # 3. Empfangs-Loop: Frames empfangen und anzeigen
    while True:
        # Empfange den nächsten Frame mit 5000ms Timeout
        t, video_frame, audio_frame, metadata_frame = ndi.RecvCaptureV3(ndi_receiver, 5000) 

        if t == ndi.frame_type_video:
            # Konvertiere den Frame in ein NumPy-Array
            frame_data = np.copy(video_frame.data)
            
            # WICHTIG: NDI gibt BGRA aus, OpenCV erwartet BGR für die Anzeige
            frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_BGRA2BGR)
            
            # 4. Frame anzeigen (Simuliert die Weitergabe an die KI)
            cv2.imshow(f"NDI Feed von {SOURCE_NAME}", frame_bgr)
            
            # Hier würde normal die KI-Logik laufen:
            # objects, lanestate = run_perception(frame_bgr)
            # mqtt_client.publish("sensor/objects", objects)

            # 5. Freigabe des NDI-Frames
            ndi.RecvFreeVideoV2(ndi_receiver, video_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Aufräumen
    cv2.destroyAllWindows()
    ndi.RecvDestroy(ndi_receiver)
    ndi.destroy()
    print("Receiver beendet.")

if __name__ == "__main__":
    run_receiver()