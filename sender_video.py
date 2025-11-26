# sender_video.py
import cv2
import time

from visiongraph_ndi.NDIVideoOutput import NDIVideoOutput


def main():
    # Pfad zu deinem Video ANPASSEN:
    video_path = r"C:\Users\firas\Documents\Semester 5\Programmieren mobiler Systeme\run1\run1\2025-11-06 11-55-05.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Video konnte nicht geöffnet werden:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 200:
        fps = 30.0
    frame_time = 1.0 / fps

    # NDI-Stream-Name: "Demo"
    with NDIVideoOutput("Demo") as ndi:
        print("NDI Sender gestartet: Demo")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video zu Ende.")
                break

            # Frame im BGR-Format an NDI schicken
            ndi.send(frame)

            # Optional: Vorschau
            cv2.imshow("Sender Preview", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("Abbruch durch Benutzer.")
                break

            time.sleep(frame_time)

    cap.release()
    cv2.destroyAllWindows()
    print("Sender beendet.")


if __name__ == "__main__":
    main()
