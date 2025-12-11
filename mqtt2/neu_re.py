import cv2
import numpy as np
import time
import sys
import json

from pathlib import Path
from typing import Tuple
from visiongraph_ndi.NDIVideoInput import NDIVideoInput

import paho.mqtt.client as mqtt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LaneDetection.lanedetec_runner import init_lanedetector, process_frame

# Optional: Torch für GPU-Synchronisation importieren
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# --- MQTT-Konfiguration ---
BROKER = "localhost"
TOPIC_LANESTATE = "sensor/lanestate"


def lanes_to_lanestate(lanes_xy, model_w, lane_width_m: float = 3.7) -> Tuple[float, float]:
    """
    Schätzt lane_center (in Metern) und curvature (Dummy) aus den erkannten Lanes.

    lanes_xy: Liste von Lanes, jede Lane = Liste von (x_model, y_canon)
    model_w:  Breite des Modell-Eingangs (cfg.train_width)
    lane_width_m: angenommene Spurbreite (≈ 3.7m)

    Rückgabe:
        lane_center: lateraler Versatz zur Spurmitte in Metern (+ rechts, - links)
        curvature:  momentan Dummy 0.0 (Krümmung)
    """

    # Mindestens 2 Lanes nötig (linke & rechte Fahrspurbegrenzung)
    if len(lanes_xy) < 2:
        return 0.0, 0.0

    lane_a = lanes_xy[0]
    lane_b = lanes_xy[1]

    def bottom_point(lane):
        if not lane:
            return None
        # Punkt mit größter y-Koordinate = am nächsten zur Kamera
        return max(lane, key=lambda p: p[1])

    pa = bottom_point(lane_a)
    pb = bottom_point(lane_b)
    if pa is None or pb is None:
        return 0.0, 0.0

    x_a, _ = pa
    x_b, _ = pb
    x_left, x_right = sorted([x_a, x_b])

    # Spurmitte im Modell-Koordinatensystem (Pixel)
    lane_center_x = 0.5 * (x_left + x_right)

    # Fahrzeugmitte = Bildmitte
    img_center_x = model_w / 2.0

    offset_px = lane_center_x - img_center_x
    lane_width_px = max(1e-3, abs(x_right - x_left))

    # Umrechnung Pixel -> Meter mithilfe der angenommenen Spurbreite
    meters_per_px = lane_width_m / lane_width_px
    lane_center_m = offset_px * meters_per_px

    curvature = 0.0  # aktuell noch nicht berechnet

    return lane_center_m, curvature


def run_perception_models(
    bgr_frame: np.ndarray,
    net,
    cfg,
    img_transforms,
    device,
    loop_fps: float | None = None,
) -> Tuple[np.ndarray, float, int, float, float, float]:
    """
    Führt LaneDetection auf einem Frame aus und zeichnet FPS ins Bild.

    Returns:
        vis_bgr:      annotiertes Bild (mit gezeichneten Lanes)
        fps_inst:     Modell-FPS (instantan)
        n_lanes:      Anzahl detektierter Lanes
        model_dt:     Modell-Laufzeit in Sekunden
        lane_center:  lateraler Versatz zur Spurmitte [m]
        curvature:    Spurkrümmung [1/m] (aktuell Dummy 0.0)
    """
    # Vor dem Modell synchronisieren, damit vorherige GPU-Operationen
    # nicht in unser Timing reinlaufen
    if HAS_TORCH and hasattr(device, "type") and device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.time()

    vis_bgr, lanes_xy, lanes_info = process_frame(bgr_frame, net, cfg, img_transforms, device)

    # Nach dem Modell wieder synchronisieren, damit wir die echte Modell-Laufzeit sehen
    if HAS_TORCH and hasattr(device, "type") and device.type == "cuda":
        torch.cuda.synchronize(device)

    t1 = time.time()
    model_dt = t1 - t0
    fps_inst = 1.0 / model_dt if model_dt > 0 else 0.0
    n_lanes = len(lanes_xy)

    # Lane-State aus Lanes berechnen
    lane_center_m, curvature = lanes_to_lanestate(
        lanes_xy,
        model_w=cfg.train_width,
        lane_width_m=3.7
    )

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

    return vis_bgr, fps_inst, n_lanes, model_dt, lane_center_m, curvature


def main():
    print("[INFO] Lane detector init...")
    net, cfg, img_transforms, device = init_lanedetector()
    print("[INFO] Lane detector ready!")

    # --- MQTT LaneState Client vorbereiten ---
    mqtt_client = mqtt.Client(client_id="lane-ndi-receiver")
    mqtt_client.connect(BROKER, 1883, keepalive=60)
    print(f"[INFO] MQTT verbunden mit {BROKER}, Topic: {TOPIC_LANESTATE}")

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
        sum_ndi_read_time = 0.0
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

            # -------- NDI / Netzwerk / Receive --------
            ndi_t0 = time.time()
            ts, frame = ndi.read()   # ts in ms (Epoch), frame = BGR (numpy)
            ndi_t1 = time.time()
            ndi_dt = ndi_t1 - ndi_t0
            sum_ndi_read_time += ndi_dt

            if frame is None:
                # hier wartet CPU „nur“ auf neue Daten
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
            vis_bgr, fps_inst, n_lanes, model_dt, lane_center_m, curvature = run_perception_models(
                frame, net, cfg, img_transforms, device, loop_fps=None
            )
            sum_model_time += model_dt

            # --- LaneState per MQTT publizieren ---
            lanestate_msg = {
                "lane_center": float(lane_center_m),
                "curvature": float(curvature)
            }
            mqtt_client.publish(
                TOPIC_LANESTATE,
                json.dumps(lanestate_msg),
                qos=1
            )

            # ---------------- Anzeige ----------------
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

            # Logging pro Frame
            print(
                f"[FRAME {total_frames:05d}] "
                f"ts={ts:13.3f} ms  "
                f"NDI={ndi_dt*1000:5.2f} ms  "
                f"model={model_dt*1000:5.2f} ms  "
                f"display={display_dt*1000:5.2f} ms  "
                f"loop={loop_dt*1000:5.2f} ms  "
                f"model_FPS={fps_inst:5.1f}  "
                f"loop_FPS={loop_fps_inst:5.1f}  "
                f"lanes={n_lanes}  "
                f"lane_center={lane_center_m:6.3f} m"
            )

            if key in (27, ord("q")):
                break

            total_frames += 1

        t_overall = time.time() - t_overall_start

        # ---------------- Statistik-Ausgabe ----------------
        if total_frames > 0:
            avg_loop_fps = total_frames / t_overall
            avg_ndi_ms = (sum_ndi_read_time / total_frames) * 1000.0
            avg_model_ms = (sum_model_time / total_frames) * 1000.0
            avg_display_ms = (sum_display_time / total_frames) * 1000.0
            avg_loop_ms = (sum_loop_time / total_frames) * 1000.0

            print(
                f"\n[STATS] Processed {total_frames} frames in "
                f"{t_overall:.2f}s -> avg loop FPS = {avg_loop_fps:.2f}"
            )

            # Anteile relativ zur Loop-Zeit
            def pct(part_ms: float, total_ms: float) -> float:
                return (part_ms / total_ms * 100.0) if total_ms > 0 else 0.0

            print(
                f"[BREAKDOWN] avg_ndi_read_time = {avg_ndi_ms:6.2f} ms/frame "
                f"({pct(avg_ndi_ms, avg_loop_ms):5.1f}% der Loop-Zeit)"
            )
            print(
                f"[BREAKDOWN] avg_model_time    = {avg_model_ms:6.2f} ms/frame "
                f"({pct(avg_model_ms, avg_loop_ms):5.1f}% der Loop-Zeit)"
            )
            print(
                f"[BREAKDOWN] avg_display_time  = {avg_display_ms:6.2f} ms/frame "
                f"({pct(avg_display_ms, avg_loop_ms):5.1f}% der Loop-Zeit)"
            )
            print(
                f"[BREAKDOWN] avg_loop_time     = {avg_loop_ms:6.2f} ms/frame "
                f"(inkl. NDI + Model + Display + sonstiges)"
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
    mqtt_client.disconnect()
    print("[INFO] Receiver beendet.")


if __name__ == "__main__":
    main()
