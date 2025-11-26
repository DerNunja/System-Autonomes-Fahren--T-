import cv2
import numpy as np
import time
import sys

from pathlib import Path
from typing import Tuple
from visiongraph_ndi.NDIVideoInput import NDIVideoInput

# Projekt-Root für Imports (LaneDetection) hinzufügen
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LaneDetection.lanedetec_runner import init_lanedetector, process_frame


def run_perception_models(
    bgr_frame: np.ndarray,
    net,
    cfg,
    img_transforms,
    device,
    loop_fps: float | None = None,
) -> Tuple[np.ndarray, float, int, float]:
    """
    Führt LaneDetection auf einem Frame aus und zeichnet FPS ins Bild.

    Args:
        bgr_frame: Eingangsframe (BGR).
        net, cfg, img_transforms, device: LaneDetektor-Objekte.
        loop_fps: instantane Loop-FPS (nur für Overlay).

    Returns:
        vis_bgr:   annotiertes Bild
        fps_inst:  Modell-FPS (instantan)
        n_lanes:   Anzahl detektierter Lanes
        model_dt:  Modell-Laufzeit in Sekunden
    """
    t0 = time.time()

    vis_bgr, lanes_xy, lanes_info = process_frame(bgr_frame, net, cfg, img_transforms, device)

    t1 = time.time()
    model_dt = t1 - t0
    fps_inst = 1.0 / model_dt if model_dt > 0 else 0.0
    n_lanes = len(lanes_xy)

    if loop_fps is not None and loop_fps > 0:
        overlay_text = f"model: {fps_inst:4.1f} FPS | loop: {loop_fps:4.1f} FPS"
    else:
        overlay_text = f"model: {fps_inst:4.1f} FPS"

    cv2.putText(
        vis_bgr,
        overlay_text,
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return vis_bgr, fps_inst, n_lanes, model_dt


def main():
    print("[INFO] Lane detector init...")
    net, cfg, img_transforms, device = init_lanedetector()
    print("[INFO] Lane detector ready!")

    print("[INFO] Suche NDI-Quellen...")
    sources = NDIVideoInput.find_sources(timeout=5.0)
    if not sources:
        print("[WARN] Keine NDI-Quellen gefunden!")
    else:
        print("[INFO] Gefundene Quellen:")
        for s in sources:
            print(" -", s.name)

    stream_name = "Demo"
    print(f"[INFO] Verbinde mit NDI-Stream: {stream_name}")
    with NDIVideoInput(stream_name=stream_name) as ndi:
        print("[INFO] NDI Receiver verbunden, warte auf Frames... (ESC/q zum Beenden)")

        total_frames = 0
        t_overall_start = time.time()

        # Stats / Breakdown
        sum_model_time = 0.0
        sum_display_time = 0.0
        sum_loop_time = 0.0

        # Video-FPS-Berechnung auf Basis einzigartiger Timestamps
        first_ts = None
        last_ts = None
        prev_ts_for_avg = None
        unique_ts_steps = 0  # Anzahl "echter" Timestamp-Updates (ts steigt)

        cv2.namedWindow("NDI Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("LaneDetection Stream", cv2.WINDOW_NORMAL)

        while ndi.is_connected:
            loop_t0 = time.time()

            ts, frame = ndi.read()   # ts in ms (Epoch), frame = BGR (numpy)

            if frame is None:
                continue

            # Video-Timestamps für avg_video_fps sammeln
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

                if prev_ts_for_avg is not None and ts > prev_ts_for_avg:
                    unique_ts_steps += 1
                prev_ts_for_avg = ts

            # ---------------- LaneDetection / Modell ----------------
            # loop_fps_inst berechnen nach der kompletten Loop; für Overlay reicht aber Näherung:
            # Wir berechnen loop_dt am Ende und geben hier einen Dummy rein, dann wird er im nächsten Frame "korrekt".
            # Einfacher: wir berechnen loop_fps_inst nach dem Modell-Aufruf:

            # zuerst Modell aufrufen ohne loop_fps
            vis_bgr, fps_inst, n_lanes, model_dt = run_perception_models(
                frame, net, cfg, img_transforms, device, loop_fps=None
            )
            sum_model_time += model_dt

            # Anzeige
            display_t0 = time.time()
            cv2.imshow("NDI Original", frame)
            cv2.imshow("LaneDetection Stream", vis_bgr)
            key = cv2.waitKey(1) & 0xFF
            display_t1 = time.time()
            display_dt = display_t1 - display_t0
            sum_display_time += display_dt

            loop_t1 = time.time()
            loop_dt = loop_t1 - loop_t0
            sum_loop_time += loop_dt
            loop_fps_inst = 1.0 / loop_dt if loop_dt > 0 else 0.0

            # Logging pro Frame (ohne Video-FPS-Kauderwelsch)
            print(
                f"[FRAME {total_frames:05d}] "
                f"ts={ts:13.3f} ms  "
                f"model_FPS={fps_inst:5.1f}  "
                f"loop_FPS={loop_fps_inst:5.1f}  "
                f"lanes={n_lanes}"
            )

            if key in (27, ord("q")):
                break

            total_frames += 1

        t_overall = time.time() - t_overall_start

        # ---------------- Statistik-Ausgabe ----------------
        if total_frames > 0:
            avg_loop_fps = total_frames / t_overall
            avg_model_ms = (sum_model_time / total_frames) * 1000.0
            avg_display_ms = (sum_display_time / total_frames) * 1000.0
            avg_loop_ms = (sum_loop_time / total_frames) * 1000.0

            print(
                f"\n[STATS] Processed {total_frames} frames in "
                f"{t_overall:.2f}s -> avg loop FPS = {avg_loop_fps:.2f}"
            )
            print(
                f"[BREAKDOWN] avg_model_time  = {avg_model_ms:6.2f} ms/frame "
                f"({avg_model_ms/avg_loop_ms*100:5.1f}% der Loop-Zeit)"
            )
            print(
                f"[BREAKDOWN] avg_display_time= {avg_display_ms:6.2f} ms/frame "
                f"({avg_display_ms/avg_loop_ms*100:5.1f}% der Loop-Zeit)"
            )
            print(
                f"[BREAKDOWN] avg_loop_time   = {avg_loop_ms:6.2f} ms/frame "
                f"(inkl. Model + Display + sonstiges)"
            )

            # globale durchschnittliche Video-FPS aus einzigartigen Timestamp-Steps
            if (
                first_ts is not None
                and last_ts is not None
                and last_ts > first_ts
                and unique_ts_steps > 0
            ):
                total_video_time_sec = (last_ts - first_ts) / 1000.0
                avg_video_fps = unique_ts_steps / total_video_time_sec
                print(
                    f"[VIDEO] avg_video_fps (unique ts steps) = {avg_video_fps:.2f} FPS"
                )

    cv2.destroyAllWindows()
    print("[INFO] Receiver beendet.")


if __name__ == "__main__":
    main()
