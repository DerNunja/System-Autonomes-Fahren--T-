import cv2
import numpy as np
import time
import sys
import math
import json
import paho.mqtt.client as mqtt

from pathlib import Path
from typing import Tuple
from visiongraph_ndi.NDIVideoInput import NDIVideoInput

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LaneDetection.lanedetec_runner import init_lanedetector, process_frame
from World.world_model import LaneDetResult, WorldModel
from drive.steering_controller import LateralController

# Optional: Torch für GPU-Synchronisation importieren
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# --- MQTT-Konfiguration ---
BROKER = "localhost"
TOPIC_CMD = "control/steering_cmd"
TOPIC_LANESTATE = "sensor/lanestate"  # <--- NEU: für mqtt_bridge/ui_world_monitor

mqtt_client = mqtt.Client(client_id="lane-steer-controller")
mqtt_client.connect(BROKER, 1883, keepalive=60)
mqtt_client.loop_start()


def draw_curvature_preview(vis_bgr: np.ndarray, ego_lane) -> np.ndarray:
    """
    Visualisiert curvature_preview als Pfeil:
    - Start: unten an der Ego-Mittelspur
    - Richtung: nach vorne (nach oben im Bild) und seitlich je nach Krümmung
      curvature_preview > 0 => Pfeil knickt nach rechts
      curvature_preview < 0 => Pfeil knickt nach links
    """
    if ego_lane is None or not ego_lane.has_ego_lane:
        return vis_bgr
    if not ego_lane.centerline_px:
        return vis_bgr

    H, W = vis_bgr.shape[:2]

    # Punkt am unteren Ende der Centerline als Anker
    bottom_center = max(ego_lane.centerline_px, key=lambda p: p[1])
    x0, y0 = int(bottom_center[0]), int(bottom_center[1])

    k = float(getattr(ego_lane, "curvature_preview", 0.0))

    # Krümmung etwas clampen, damit der Pfeil nicht komplett aus dem Bild fliegt
    k_clamped = max(-0.02, min(0.02, k))

    # Skalierung: wie stark die Krümmung den Pfeil seitlich ablenkt
    side_scale = 8000.0  # kannst du später feinjustieren
    dy = -80             # Pfeil zeigt nach "vorn" (nach oben, da y nach unten wächst)
    dx = int(k_clamped * side_scale)

    x1 = x0 + dx
    y1 = y0 + dy

    # Pfeil zeichnen
    cv2.arrowedLine(
        vis_bgr,
        (x0, y0),
        (x1, y1),
        (0, 0, 255),      # rot
        2,
        tipLength=0.25,
    )

    # Text mit Rohwert der Krümmung
    txt = f"curv_prev={k:+.4f}"
    cv2.putText(
        vis_bgr,
        txt,
        (10, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    return vis_bgr


def draw_steering_preview(
    vis_bgr: np.ndarray,
    origin: Tuple[int, int],
    steer_cmd,
    length_px: int = 120,
) -> np.ndarray:
    """
    Zeichnet einen Preview-Pfeil basierend auf dem Lenkwinkel.
    """
    cx, cy = origin
    angle = steer_cmd.steer_rad  # rad

    # Fahrzeugkoordinaten: x nach vorne, y nach links
    dx = math.sin(angle)
    dy = -math.cos(angle)   # fürs Bild: nach unten ist +v

    x2 = int(cx + dx * length_px)
    y2 = int(cy + dy * length_px)

    cv2.arrowedLine(
        vis_bgr,
        (cx, cy),
        (x2, y2),
        (0, 100, 200),
        3,
        tipLength=0.2,
    )
    return vis_bgr


def draw_ego_centerline(vis_bgr: np.ndarray, ego_lane) -> np.ndarray:
    """
    Zeichnet die Ego-Mittelspur (centerline_px) als gelbe Linie ins Bild,
    inkl. Text mit Lateraloffset.
    """
    if ego_lane is None or not ego_lane.has_ego_lane:
        return vis_bgr

    if not ego_lane.centerline_px:
        return vis_bgr

    pts = np.array(ego_lane.centerline_px, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        vis_bgr,
        [pts],
        isClosed=False,
        color=(0, 255, 255),  # gelb
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    # Marker am unteren Punkt
    bottom_pt = max(ego_lane.centerline_px, key=lambda p: p[1])
    cv2.circle(
        vis_bgr,
        (int(bottom_pt[0]), int(bottom_pt[1])),
        5,
        (0, 255, 255),
        -1,
        lineType=cv2.LINE_AA,
    )

    # Text mit Lateraloffset
    offset_px = ego_lane.lateral_offset_px
    txt = f"ego_offset={offset_px:+.1f}px"
    cv2.putText(
        vis_bgr,
        txt,
        (10, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return vis_bgr


def run_perception_models(
    bgr_frame: np.ndarray,
    net,
    cfg,
    img_transforms,
    device,
    loop_fps: float | None = None,
):
    """
    Führt LaneDetection auf einem Frame aus und zeichnet FPS ins Bild.

    Returns:
        vis_bgr:   annotiertes Bild
        fps_inst:  Modell-FPS (instantan)
        n_lanes:   Anzahl detektierter Lanes
        model_dt:  Modell-Laufzeit in Sekunden (inkl. GPU + CPU, per cuda.synchronize)
        lanes_xy:  Lane-Punkte im Modell-/Canon-Raum (x_model, y_canon)
        lanes_info: Meta-Infos pro Lane (lane_id, score, n_points)
    """
    if HAS_TORCH and hasattr(device, "type") and device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.time()

    vis_bgr, lanes_xy, lanes_info = process_frame(bgr_frame, net, cfg, img_transforms, device)

    if HAS_TORCH and hasattr(device, "type") and device.type == "cuda":
        torch.cuda.synchronize(device)

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

    return vis_bgr, fps_inst, n_lanes, model_dt, lanes_xy, lanes_info


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

        wm: WorldModel | None = None  # Weltmodell-Instanz (wird lazy initialisiert)
        controller: LateralController | None = None

        while ndi.is_connected:
            loop_t0 = time.time()

            #  NDI / Netzwerk / Receive
            ndi_t0 = time.time()
            ts, frame = ndi.read()   # ts in ms (Epoch), frame = BGR (numpy)
            ndi_t1 = time.time()
            ndi_dt = ndi_t1 - ndi_t0
            sum_ndi_read_time += ndi_dt

            if frame is None:
                continue

            # Bildgröße
            H, W = frame.shape[:2]

            # WorldModel beim ersten validen Frame initialisieren
            if wm is None:
                wm = WorldModel(img_width=W, img_height=H)
                controller = LateralController(
                    max_steer_rad=0.5,
                    k_stanley=1.0,
                    v_ref=20.0,
                    k_ff=8.0,
                    history_window_s=0.5,
                )

            # Video-Timestamps für avg_video_fps sammeln
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

                if prev_ts_for_avg is not None and ts > prev_ts_for_avg:
                    unique_ts_steps += 1
                prev_ts_for_avg = ts

            #  LaneDetection / Modell
            vis_bgr, fps_inst, n_lanes, model_dt, lanes_xy, lanes_info = run_perception_models(
                frame, net, cfg, img_transforms, device, loop_fps=None
            )
            sum_model_time += model_dt

            #  Weltmodell aktualisieren & Ego-Mittelspur einzeichnen
            lane_res = LaneDetResult(
                timestamp_ms=int(ts) if ts is not None else 0,
                img_width=W,
                img_height=H,
                lanes_model_xy=lanes_xy,
                lanes_info=lanes_info,
                model_width=cfg.train_width,
                canon_height=590,
            )
            wm_state = wm.update_from_lane_detection(lane_res)

            if wm_state.ego_lane is not None:
                vis_bgr = draw_ego_centerline(vis_bgr, wm_state.ego_lane)

            if wm_state.ego_lane and wm_state.ego_lane.has_ego_lane and controller is not None:
                ego = wm_state.ego_lane
                cmd = controller.update(
                    offset_m=ego.lateral_offset_m,
                    heading_error_rad=ego.heading_px_rad,
                    curvature_preview=ego.curvature_preview,
                )

                # Steering-Pfeil
                origin = (W // 2, int(H * 0.8))
                vis_bgr = draw_steering_preview(vis_bgr, origin, cmd)

                # Debug-Text
                txt3 = (
                    f"offset={ego.lateral_offset_m:+.2f} m, "
                    f"steer={cmd.steer_rad:+.3f} rad (norm={cmd.steer_norm:+.2f}), "
                    f"ff={cmd.ff_term:+.3f}"
                )
                cv2.putText(
                    vis_bgr,
                    txt3,
                    (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                )

                # --- EXISTIERENDER Steering-Command-Publish ---
                payload = {
                    "t_ms": int(ts) if ts is not None else 0,
                    "steer_rad": cmd.steer_rad,        # physischer Lenkwinkel
                    "steer_norm": cmd.steer_norm,      # -1..+1
                    "ff_term": cmd.ff_term,
                    "offset_m": ego.lateral_offset_m,
                    "heading_err_rad": ego.heading_px_rad,
                    "curvature": ego.curvature_preview,
                }
                mqtt_client.publish(TOPIC_CMD, json.dumps(payload))

                # --- NEU: LaneState für mqtt_bridge / UI ---
                lanestate_msg = {
                    "lane_center": float(ego.lateral_offset_m),
                    "curvature": float(ego.curvature_preview),
                }
                mqtt_client.publish(TOPIC_LANESTATE, json.dumps(lanestate_msg))

                # Krümmungs-Pfeil
                vis_bgr = draw_curvature_preview(vis_bgr, ego)

            #  Anzeige
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
                f"lanes={n_lanes}"
            )

            if key in (27, ord("q")):
                break

            total_frames += 1

        t_overall = time.time() - t_overall_start

        #  Statistik-Ausgabe
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
    print("[INFO] Receiver beendet.")


if __name__ == "__main__":
    main()