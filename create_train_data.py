import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ===================== Parameter =====================
# Eingaben
video_path  = Path("Behavioural_Cloning/data/Recordings/Video/u_2025-05-23 10-47-10_h264.mp4")
labels_path = Path("Behavioural_Cloning/data/Recordings/TabData/u_recording_2025_05_23__11_14_03.csv")

# Ausgaben
out_frames_dir = Path("Behavioural_Cloning/data/Processed/frames")
out_labels_csv = Path("Behavioural_Cloning/data/Processed/labels/labels_to_frames.csv")

# "timestamp"  -> CSV am nächstliegenden 'timestamp' zu csv_time_offset_sec schneiden
# "frame"      -> nur Video-Start über trim_video_start_sec setzen 
ALIGN_BY = "timestamp"  # oder "frame"

# Steuerung
fallback_fps          = 60.0
csv_time_offset_sec   = 23.111638     # richtiger start des videos
trim_video_start_sec  = 0.0           # schneidet cruden splashscreen weg
duration_sec          = 100           # wie viele minuten des videos verwendet werden sollen
sample_stride         = 1             # jeder x-te frame wird verwendet; 1=alle Frames

image_ext             = ".jpg"
jpeg_quality          = 95

# um crop fein abzustimmen verwende crop_finder.py
crop_box              = (482, 210, 960, 300)   
scale_percent         = 100     #skaliert das video 100%= keine skalierung bsp. 50% skalierung bei 1920x1080px = 960x540px                     

# double check der wichtigen features
required_cols = ["timestamp", "throttle", "brakes", "car0_velocity", "wheel_position"]
optional_cols = ["frame_idx", "t_sec"]


def apply_crop_and_scale(frame, crop_box, scale_percent):
    h, w = frame.shape[:2]

    if crop_box is not None:
        x, y, cw, ch = map(int, crop_box)
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        cw = max(1, min(cw, w - x))
        ch = max(1, min(ch, h - y))
        frame = frame[y:y+ch, x:x+cw]

    sp = float(scale_percent)
    if sp != 100.0:
        out_w = max(1, int(round(frame.shape[1] * sp / 100.0)))
        out_h = max(1, int(round(frame.shape[0] * sp / 100.0)))
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA if sp < 100 else cv2.INTER_LINEAR)

    out_h, out_w = frame.shape[:2]
    return frame, out_w, out_h

df = pd.read_csv(labels_path)

for col in required_cols:
    assert col in df.columns, f"Spalte '{col}' fehlt in {labels_path}"

needed = set(["timestamp"] + required_cols + [c for c in optional_cols if c in ["frame_idx","t_sec"]])
try:
    # Nur relevante Spalten laden, falls CSV Header diese hat – sonst alle laden
    preview = pd.read_csv(labels_path, nrows=0)
    usecols = [c for c in preview.columns if c in needed]
    if "timestamp" not in usecols:
        usecols = None  # falls Spaltennamen anders/unerwartet sind
except Exception:
    usecols = None

needed = set(["timestamp"] + required_cols + [c for c in optional_cols if c in ["frame_idx","t_sec"]])
try:
    preview = pd.read_csv(labels_path, nrows=0)
    usecols = [c for c in preview.columns if c in needed]
    if "timestamp" not in usecols:
        usecols = None
except Exception:
    usecols = None

df = pd.read_csv(labels_path, usecols=usecols, low_memory=False)

def coerce_float(s):
    return pd.to_numeric(
        pd.Series(s, copy=False)
          .astype(str)
          .str.replace(",", ".", regex=False)
          .str.replace(r"[^0-9\.\-\+eE]", "", regex=True),
        errors="coerce"
    )

for col in required_cols:
    assert col in df.columns, f"Spalte '{col}' fehlt in {labels_path}"

cols_to_float = ["timestamp"] + required_cols + [c for c in optional_cols if c in df.columns]
for c in cols_to_float:
    df[c] = coerce_float(df[c])

before = len(df)
df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
dropped = before - len(df)
if dropped > 0:
    print(f"[CLEAN] {dropped} Zeilen ohne gültigen 'timestamp' entfernt.")

# Timestamp-Einheit erkennen -> Sekunden
ts_med = float(df["timestamp"].median())
if ts_med > 1e12:
    df["timestamp"] = df["timestamp"] / 1e9
    print("[TIME] timestamp als Nanosekunden erkannt → in Sekunden umgerechnet.")
elif ts_med > 1e9:
    df["timestamp"] = df["timestamp"] / 1e3
    print("[TIME] timestamp als Millisekunden erkannt → in Sekunden umgerechnet.")

df = df.sort_values("timestamp").reset_index(drop=True)

if ALIGN_BY == "timestamp":
    # === Neue Variante: CSV an nächstliegendem timestamp schneiden ===
    diff = (df["timestamp"] - float(csv_time_offset_sec)).abs()
    nearest_idx = int(diff.idxmin())
    t0 = float(df.loc[nearest_idx, "timestamp"])
    delta = float(t0 - float(csv_time_offset_sec))
    print(f"[ALIGN/timestamp] Start im CSV: {t0:.6f}s (Δ={delta:+.6f}s zu Vorgabe {csv_time_offset_sec:.6f}s)")
    if abs(delta) > 0.5:
        print("[WARN] Versatz >0.5s – bitte prüfen.")

    df = df.loc[nearest_idx:].copy().reset_index(drop=True)

    # leichte Lücken in Pflichtspalten füllen
    df[required_cols] = df[required_cols].ffill().bfill()

    # Relativzeit ab gefundenem CSV-Start
    df["t_video"] = df["timestamp"] - t0

elif ALIGN_BY == "frame":
    # === Alte Variante: nur Startframe per trim_video_start_sec; CSV ungeschnitten ===
    # Relativzeit ab vorgegebenem csv_time_offset_sec
    df["t_video"] = df["timestamp"] - float(csv_time_offset_sec)

    # optional: negative Zeiten knapp unter 0 tolerieren/wegschneiden wie früher
    df = df[df["t_video"] >= -1e-6].copy().reset_index(drop=True)

    # leichte Lücken in Pflichtspalten füllen
    df[required_cols] = df[required_cols].ffill().bfill()

else:
    raise ValueError("ALIGN_BY muss 'timestamp' oder 'frame' sein.")

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Video konnte nicht geöffnet werden: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0.0 or np.isnan(fps):
    fps = float(fallback_fps)
    print(f"[WARN] FPS nicht lesbar – Fallback auf {fps}")

n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Video: {W0}x{H0}, FPS={fps}, Frames={n_total_frames}")



start_frame = int(round(trim_video_start_sec * fps))
end_frame_exclusive = n_total_frames if duration_sec is None else min(n_total_frames, start_frame + int(round(duration_sec * fps)))
if start_frame >= n_total_frames:
    raise ValueError("trim_video_start_sec liegt hinter dem Videoende.")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)



frame_indices = np.arange(start_frame, end_frame_exclusive, dtype=int)
t_videos = (frame_indices - start_frame) / fps  

frames_df = pd.DataFrame({"frame_idx_abs": frame_indices, "t_video": t_videos})
frames_df = frames_df.iloc[::sample_stride].reset_index(drop=True)



frames_df = frames_df.sort_values("t_video")
df = df.sort_values("t_video")
tolerance = 1.0 / (2.0 * fps)

aligned = pd.merge_asof(
    frames_df, df[["t_video"] + required_cols + [c for c in optional_cols if c in df.columns]],
    on="t_video", direction="nearest", tolerance=tolerance
)
aligned = aligned.ffill().bfill()



out_frames_dir.mkdir(parents=True, exist_ok=True)
out_labels_csv.parent.mkdir(parents=True, exist_ok=True)

max_angle_deg = 35.0
max_angle_rad = np.deg2rad(max_angle_deg)

records = []
export_count = 0
need_abs = set(frames_df["frame_idx_abs"].tolist())



pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
while pos < start_frame:
    ok, _ = cap.read()
    if not ok:
        break
    pos += 1



while True:
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if pos >= end_frame_exclusive:
        break
    ok, frame = cap.read()
    if not ok:
        break

    abs_idx = pos - 1
    if abs_idx in need_abs:
        row = aligned.loc[aligned["frame_idx_abs"] == abs_idx].iloc[0]

        frame_proc, out_w, out_h = apply_crop_and_scale(frame, crop_box, scale_percent)

        fname = f"frame_{export_count:06d}{image_ext}"
        fpath = out_frames_dir / fname
        if image_ext.lower() == ".jpg":
            cv2.imwrite(str(fpath), frame_proc, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        else:
            cv2.imwrite(str(fpath), frame_proc)

        steer_norm = float(np.clip(row["wheel_position"], -1.0, 1.0))
        steer_rad  = steer_norm * max_angle_rad

        rec = {
            "frame_idx_rel": export_count,
            "frame_idx_abs": int(abs_idx),
            "t_video": float(row["t_video"]),
            "filename": fname,
            "width": out_w,                
            "height": out_h,               
            "timestamp_src": float(row["timestamp"]),
            "throttle": float(row["throttle"]),
            "brakes": float(row["brakes"]),
            "car0_velocity": float(row["car0_velocity"]),
            "wheel_position": float(steer_norm),
            "steering_angle_rad": float(steer_rad),
        }
        if "frame_idx" in row: 
            rec["frame_idx_csv"] = int(row["frame_idx"])
        if "t_sec" in row:     
            rec["t_sec_csv"]   = float(row["t_sec"])

        records.append(rec)
        export_count += 1

cap.release()

labels_out_df = pd.DataFrame.from_records(records)
labels_out_df.to_csv(out_labels_csv, index=False)

print(f"[OK] Export fertig. Frames: {export_count} -> {out_frames_dir}")
print(f"[OK] Labels: {out_labels_csv}")
print(f"[OK] Größe nach Crop/Scale: {labels_out_df.iloc[0]['width']}x{labels_out_df.iloc[0]['height']}" if export_count>0 else "")