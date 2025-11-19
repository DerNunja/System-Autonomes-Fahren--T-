import cv2
import numpy as np

from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.video_frame import VideoFrameSync


def run_perception_models(bgr_frame):
    """
    HIER später: dein Detection- / Segmentierungsmodell.
    Jetzt nur Dummy -> Graustufenbild.
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    return gray


def main():
    # 1) Finder und Receiver vorbereiten
    finder = Finder()

    # Receiver ohne Source anlegen
    receiver = Receiver(
        color_format=RecvColorFormat.RGBX_RGBA,   # RGBA
        bandwidth=RecvBandwidth.highest,
    )

    # VideoFrame-Objekt, in das NDI schreibt
    video_frame = VideoFrameSync()
    receiver.frame_sync.set_video_frame(video_frame)

    # Variable für die aktuell genutzte NDI-Quelle
    source = None

    # Callback, wenn sich die gefundenen NDI-Quellen ändern
    def on_finder_change():
        nonlocal source
        if finder is None:
            return

        ndi_source_names = finder.get_source_names()
        if len(ndi_source_names) == 0:
            print("Noch keine NDI-Quelle gefunden...")
            return

        if source is not None:
            # schon mit einer Quelle verbunden
            return

        # Wir nehmen einfach die erste gefundene Quelle
        first_name = ndi_source_names[0]
        print("Verbinde mit NDI-Quelle:", first_name)

        # Quelle holen und beim Receiver setzen
        with finder.notify:
            source_obj = finder.get_source(first_name)
            source = source_obj
            receiver.set_source(source)

    # Callback registrieren und Finder starten
    finder.set_change_callback(on_finder_change)
    finder.open()

    cv2.namedWindow("NDI Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Perception Result", cv2.WINDOW_NORMAL)

    print("Warte auf Verbindung zu einer NDI-Quelle...")
    print("ESC oder 'q' zum Beenden.")

    while True:
        # Nur wenn Receiver verbunden ist, Frames holen
        if receiver.is_connected():
            # Ein Video-Frame vom NDI-Stream holen
            receiver.frame_sync.capture_video()

            # Prüfen, ob Auflösung > 0 ist
            if min(video_frame.xres, video_frame.yres) != 0:
                # Rohdaten als NumPy-Array holen
                frame_rgba = video_frame.get_array()
                frame_rgba = frame_rgba.reshape(
                    video_frame.yres, video_frame.xres, 4
                )

                # RGBA -> BGR (OpenCV)
                bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)

                # Hier dein Modell aufrufen
                result_img = run_perception_models(bgr)

                # Anzeigen
                cv2.imshow("NDI Original", bgr)
                cv2.imshow("Perception Result", result_img)

        # Tastaturabfrage
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:  # q oder ESC
            break

    # Aufräumen
    cv2.destroyAllWindows()
    if receiver.is_connected():
        receiver.disconnect()
    finder.close()
    print("Receiver beendet.")


if __name__ == "__main__":
    main()
