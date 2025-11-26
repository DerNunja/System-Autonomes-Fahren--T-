import cv2
import numpy as np

from visiongraph_ndi.NDIVideoInput import NDIVideoInput


def run_perception_models(bgr_frame: np.ndarray) -> np.ndarray:
    """Hier später deine Detection / Segmentierung.
    Aktuell nur: Graustufen als Dummy."""
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    return gray


def main():
    # Nur zum Debug: gespeicherte Quellen anzeigen
    print("Suche NDI-Quellen...")
    sources = NDIVideoInput.find_sources(timeout=5.0)
    if not sources:
        print("Keine NDI-Quellen gefunden!")
    else:
        print("Gefundene Quellen:")
        for s in sources:
            print(" -", s.name)

    # WICHTIG: stream_name = NUR "Demo", NICHT "FIRAS (Demo)"
    print("Verbinde mit NDI-Stream: Demo")
    with NDIVideoInput(stream_name="Demo") as ndi:
        print("NDI Receiver verbunden, warte auf Frames... (ESC zum Beenden)")

        while ndi.is_connected:
            ts, frame = ndi.read()   # frame = BGR-Image (numpy)

            if frame is None:
                continue

            result = run_perception_models(frame)

            cv2.imshow("NDI Original", frame)
            cv2.imshow("Perception Result", result)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cv2.destroyAllWindows()
    print("Receiver beendet.")


if __name__ == "__main__":
    main()
