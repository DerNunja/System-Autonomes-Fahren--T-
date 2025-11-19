import time
import cv2
import numpy as np
import NDIlib as ndi

VIDEO_PATH = "/home/konrada/projects/Uni/ProjektAutonomesFahren/Behavioural_Cloning_Basic/data/Recordings/Video/ego_h264.mp4"
SOURCE_NAME = "Simu_Video_Feed"


def run_sender():
    if not ndi.initialize():
        print("NDI kann nicht initialisiert werden.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Fehler: Video {VIDEO_PATH} konnte nicht geöffnet werden.")
        ndi.destroy()
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / fps

    # --- NDI Sender konfigurieren (einfach) ---
    send_settings = ndi.SendCreate()
    send_settings.ndi_name = SOURCE_NAME
    ndi_sender = ndi.send_create(send_settings)

    if not ndi_sender:
        print("Fehler beim Erstellen des NDI-Senders.")
        cap.release()
        ndi.destroy()
        return

    # VideoFrame vorbereiten
    ndi_frame = ndi.VideoFrameV2()
    ndi_frame.xres = width
    ndi_frame.yres = height
    ndi_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA

    print(f"NDI Sender '{SOURCE_NAME}' gestartet. Sende Frames ({width}x{height} @ {fps:.2f} FPS)...")

    frame_idx = 0

    try:
        while True:
            start_time = time.time()
            ret, frame_bgr = cap.read()
            if not ret:
                print("[Sender] Ende des Videos erreicht. Loopen...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_idx += 1

            frame_bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
            frame_bgra = np.ascontiguousarray(frame_bgra)

            ndi_frame.data = frame_bgra

            if frame_idx <= 5:
                print(f"[Sender][Frame {frame_idx}] shape={frame_bgra.shape}, "
                      f"min={frame_bgra.min()}, max={frame_bgra.max()}")

            ndi.send_send_video_v2(ndi_sender, ndi_frame)

            elapsed = time.time() - start_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Sender abgebrochen (KeyboardInterrupt).")

    finally:
        cap.release()
        ndi.send_destroy(ndi_sender)
        ndi.destroy()
        print("Sender beendet.")


if __name__ == "__main__":
    run_sender()