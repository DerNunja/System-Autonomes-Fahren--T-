import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ---- Pfade ----
video_path  = Path("Behavioural_Cloning_Basic/data/Recordings/Video/u_2025-05-23 10-47-10_h264.mp4")
labels_path = Path("Behavioural_Cloning_Basic/data/Processed/labels/labels_to_frames.csv")
out_path    = Path("Behavioural_Cloning_Basic/data/Processed/test_labels.mp4")

# ---- Parameter ----
fallback_fps = 60.0
max_len_px   = 180
center_ratio = (0.5, 0.8)

# Zeitfenster
csv_time_offset_sec = 0  #kann ignoriert werden
duration_sec = 120.0 

max_angle_deg = 35.0
max_angle_rad = np.deg2rad(max_angle_deg)

df = pd.read_csv(labels_path)
for need in ["frame_idx_rel", "wheel_position"]:
    assert need in df.columns, f"Spalte '{need}' fehlt in {labels_path}"


df = df.sort_values("frame_idx_rel").reset_index(drop=True)


cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Video konnte nicht geöffnet werden: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0.0 or np.isnan(fps):
    fps = fallback_fps
    print(f"[WARN] FPS nicht lesbar – Fallback auf {fps}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Auflösung: {w}x{h}, FPS: {fps}")


frames_offset = int(round(csv_time_offset_sec * fps))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

# ---- HUD ----
def draw_hud(img, steering_angle_rad, throttle, brakes):
    mag = float(np.clip(throttle - brakes, 0.0, 1.0))
    length = int(mag * max_len_px)
    cx = int(center_ratio[0] * w)
    cy = int(center_ratio[1] * h)

    # 0 rad = nach oben; rechts positiv:
    ang = float(steering_angle_rad)
    dx = int(length * np.sin(ang))
    dy = int(-length * np.cos(ang))
    end_pt = (cx + dx, cy + dy)

    cv2.circle(img, (cx, cy), 6, (255, 255, 255), -1)
    cv2.arrowedLine(img, (cx, cy), end_pt, (0, 255, 0), 6, tipLength=0.18)

    txt = f"steer={steering_angle_rad:+.2f} rad | thr={throttle:.2f} brk={brakes:.2f}"
    cv2.putText(img, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230,230,230), 2, cv2.LINE_AA)


max_frames = int(round(duration_sec * fps))
i = 0
ok, frame = cap.read()
while ok and i < max_frames:
    # passender Label-Index:
    label_idx = i + frames_offset
    if label_idx >= len(df):
        break
    row = df.iloc[label_idx]

    # wheel_position [-1,1] -> Winkel in rad
    steer_norm = float(np.clip(row["wheel_position"], -1.0, 1.0))
    steer_rad  = steer_norm * max_angle_rad

    thr = float(row.get("throttle", 0.0))
    brk = float(row.get("brakes",   0.0))

    draw_hud(frame, steer_rad, thr, brk)

    writer.write(frame)
    # Live-Preview (optional):
    # cv2.imshow("overlay", frame)
    # if cv2.waitKey(1) & 0xFF == 27: break

    i += 1
    ok, frame = cap.read()

cap.release()
writer.release()
print(f"[OK] Overlay gespeichert: {out_path}")