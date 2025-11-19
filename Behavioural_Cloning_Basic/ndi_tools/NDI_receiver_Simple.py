import cv2
import numpy as np

from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.video_frame import VideoFrameSync

from LaneDetection.lanedetec_runner import init_lanedetector, process_frame

LANE_NET = None
LANE_CFG = None
LANE_TRANSFORMS = None
LANE_DEVICE = None

def init_models():
    """
    Initialisiert das LaneDetection-Modell einmalig.
    """
    global LANE_NET, LANE_CFG, LANE_TRANSFORMS, LANE_DEVICE
    print("[LaneDetec] Initialisiere LaneDetection-Modell...")
    LANE_NET, LANE_CFG, LANE_TRANSFORMS, LANE_DEVICE = init_lanedetector()
    print("[LaneDetec] Modell geladen.")


def run_perception_models(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Wendet das LaneDetection-Modell auf einen BGR-Frame an und
    gibt ein Visualisierungbild (z.B. mit eingezeichneten Lanes) zurück.

    Fallback: wenn das Modell nicht initialisiert ist, wird
    einfach das Originalbild zurückgegeben.
    """
    if any(v is None for v in (LANE_NET, LANE_CFG, LANE_TRANSFORMS, LANE_DEVICE)):
        print("[LaneDetec] WARNUNG: Modell nicht initialisiert, gebe Originalbild zurück.")
        return bgr_frame

    vis, lanes_xy, lanes_info = process_frame(
        bgr_frame, LANE_NET, LANE_CFG, LANE_TRANSFORMS, LANE_DEVICE
    )

    return vis


def main():
    init_models()

    finder = Finder()

    receiver = Receiver(
        color_format=RecvColorFormat.RGBX_RGBA,   # RGBA
        bandwidth=RecvBandwidth.highest,
    )

    video_frame = VideoFrameSync()
    receiver.frame_sync.set_video_frame(video_frame)

    source = None

    def on_finder_change():
        nonlocal source
        if finder is None:
            return

        ndi_source_names = finder.get_source_names()
        if len(ndi_source_names) == 0:
            print("Noch keine NDI-Quelle gefunden...")
            return

        if source is not None:
            # schon verbunden
            return

        first_name = ndi_source_names[0]
        print("Verbinde mit NDI-Quelle:", first_name)

        with finder.notify:
            source_obj = finder.get_source(first_name)
            source = source_obj
            receiver.set_source(source)

    finder.set_change_callback(on_finder_change)
    finder.open()

    cv2.namedWindow("NDI Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Perception Result", cv2.WINDOW_NORMAL)

    print("Warte auf Verbindung zu einer NDI-Quelle...")
    print("ESC oder 'q' zum Beenden.")

    while True:
        if receiver.is_connected():
            receiver.frame_sync.capture_video()

            if min(video_frame.xres, video_frame.yres) != 0:
                frame_rgba = video_frame.get_array()
                frame_rgba = frame_rgba.reshape(
                    video_frame.yres, video_frame.xres, 4
                )

                bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)

                result_img = run_perception_models(bgr)

                cv2.imshow("NDI Original", bgr)
                cv2.imshow("Perception Result", result_img)

        # Tastaturabfrage
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:  # q oder ESC
            break

    cv2.destroyAllWindows()
    if receiver.is_connected():
        receiver.disconnect()
    finder.close()
    print("Receiver beendet.")


if __name__ == "__main__":
    main()