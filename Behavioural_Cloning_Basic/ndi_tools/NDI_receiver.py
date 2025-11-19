# ndi_receiver_mock.py
import NDIlib as ndi
import cv2
import numpy as np
import time
import os
import sys

# Pfad zu diesem File (…/Behavioural_Cloning_Basic/ndi_tools/NDI_receiver.py)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Projektroot: …/Behavioural_Cloning_Basic
PROJECT_ROOT = os.path.dirname(THIS_DIR)

# Projektroot auf sys.path setzen, falls noch nicht drin
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from LaneDetection.lanedetec_runner import init_lanedetector, process_frame

SOURCE_NAME = "Simu_Video_Feed"

def run_receiver():
    if not ndi.initialize():
        print("NDI konnte nicht initialisiert werden.")
        return

    # --- Quellen suchen ---
    finder = ndi.find_create_v2()
    if not finder:
        print("Fehler beim Erstellen des Finders.")
        ndi.destroy()
        return

    print(f"Suche nach NDI-Quelle, die '{SOURCE_NAME}' enthält...")

    ndi_source = None
    try:
        while ndi_source is None:
            # 1000 ms warten, dann aktuelle Liste holen
            ndi.find_wait_for_sources(finder, 1000)
            sources = ndi.find_get_current_sources(finder) or []
            print(f"Gefundene Quellen: {[s.ndi_name for s in sources]}")

            for s in sources:
                if SOURCE_NAME in s.ndi_name:
                    ndi_source = s
                    break
    finally:
        ndi.find_destroy(finder)

    if ndi_source is None:
        print("Keine passende NDI-Quelle gefunden.")
        ndi.destroy()
        return

    print(f"Quelle gefunden: {ndi_source.ndi_name}")

    # --- Receiver konfigurieren & erstellen ---
    recv_cfg = ndi.RecvCreateV3()
    recv_cfg.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
    ndi_receiver = ndi.recv_create_v3(recv_cfg)

    if not ndi_receiver:
        print("Fehler beim Erstellen des NDI-Receivers.")
        ndi.destroy()
        return

    # mit der Quelle verbinden
    ndi.recv_connect(ndi_receiver, ndi_source)
    print("NDI Receiver gestartet. Empfange Frames...  (q zum Beenden)")

    try:
        while True:
            # 5000 ms Timeout
            t, video_frame, audio_frame, metadata_frame = ndi.recv_capture_v2(ndi_receiver, 5000)

            if t == ndi.FRAME_TYPE_VIDEO:
                # Daten nach NumPy kopieren
                frame_bgra = np.copy(video_frame.data)
                frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

                cv2.imshow(f"NDI Feed von {SOURCE_NAME}", frame_bgr)

                # Frame freigeben
                ndi.recv_free_video_v2(ndi_receiver, video_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Receiver abgebrochen (KeyboardInterrupt).")

    finally:
        cv2.destroyAllWindows()
        ndi.recv_destroy(ndi_receiver)
        ndi.destroy()
        print("Receiver beendet.")


if __name__ == "__main__":
    run_receiver()