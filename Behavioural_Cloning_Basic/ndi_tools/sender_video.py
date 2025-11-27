import cv2
import time
from typing import Optional, Tuple

from visiongraph_ndi.NDIVideoOutput import NDIVideoOutput

# ========================== USER CONFIG ========================== #
# Modus wählen
USE_LIVE_SOURCE = False   # True = Live-Kamera / RTSP, False = Videodatei

# Datei-Modus
VIDEO_PATH = "/home/konrada/projects/Uni/ProjektAutonomesFahren/Behavioural_Cloning_Basic/data/Recordings/Video/ego_h264.mp4"

# Live-Modus:
# - 0 = erste Webcam
# - "/dev/video0" = Capture-Device unter Linux
# - "rtsp://..." = IP-Kamera
LIVE_SOURCE = 0

# gewünschte Video-Ausgabeauflösung (Breite, Höhe)
TARGET_SIZE: Optional[Tuple[int, int]] = (640, 360)

# gewünschte Ausgabe-FPS (nur zum Drosseln; bei Live oft None ok)
TARGET_FPS: Optional[float] = 60.0
# ================================================================= #


def main():
    # Quelle öffnen
    if USE_LIVE_SOURCE:
        print(f"[INFO] Öffne Live-Quelle: {LIVE_SOURCE}")
        cap = cv2.VideoCapture(LIVE_SOURCE, cv2.CAP_DSHOW)
    else:
        print(f"[INFO] Öffne Videodatei: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Video-/Livequelle konnte nicht geöffnet werden.")
        return

    # FPS bestimmen
    if USE_LIVE_SOURCE:
        # Live: FPS sind oft unzuverlässig -> wir nehmen TARGET_FPS oder keinen Sleep
        fps = TARGET_FPS if TARGET_FPS else cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        # Datei: FPS aus dem Container lesen
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0 or orig_fps > 200:
            orig_fps = 30.0
        fps = TARGET_FPS if TARGET_FPS else orig_fps

    frame_time = 1.0 / fps if fps > 0 else 0.0

    with NDIVideoOutput("Demo") as ndi:
        print(f"NDI Sender gestartet: Demo ({fps:.1f} FPS, target size={TARGET_SIZE})")

        while True:
            ret, frame = cap.read()
            if not ret:
                if USE_LIVE_SOURCE:
                    # Live: kurz warten und weiterprobieren
                    time.sleep(0.01)
                    continue
                else:
                    print("Video zu Ende.")
                    break

            # ggf. Auflösung anpassen
            if TARGET_SIZE is not None:
                frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)

            # an NDI senden (BGR)
            ndi.send(frame)

            # Preview
            cv2.imshow("Sender Preview", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("Abbruch durch Benutzer.")
                break

            # Zeitdrosselung:
            # - bei Datei: Simulation der Original-FPS
            # - bei Live: nur, wenn du wirklich hart auf TARGET_FPS kappen willst
            if frame_time > 0:
                time.sleep(frame_time)

    cap.release()
    cv2.destroyAllWindows()
    print("Sender beendet.")


if __name__ == "__main__":
    main()
