import cv2
import numpy as np
import pandas as pd

video_path = "Behavioural_Cloning/data/Recordings/Video/u_2025-05-23 10-47-10_h264.mp4"
csv_path   = "Behavioural_Cloning/data/Recordings/TabData/u_recording_2025_05_23__11_14_03.csv"


cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

video_offset = 0.0

video_times = np.arange(n_frames) / fps + video_offset
frames_df = pd.DataFrame({
    "frame_idx": np.arange(n_frames, dtype=int),
    "t_sec": video_times
})

df = pd.read_csv(csv_path, skiprows=[1])

if "timestamp" not in df.columns:
    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]
assert "timestamp" in df.columns, "Spalte 'timestamp' wurde nicht gefunden."

df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).sort_values("timestamp")


cols_keep = ["timestamp", "throttle", "brakes", "car0_velocity", "wheel_position"]
cols_keep = [c for c in cols_keep if c in df.columns]  
csv_sig = df[cols_keep].copy()

frames_df = frames_df.sort_values("t_sec")
csv_sig   = csv_sig.sort_values("timestamp")

aligned = pd.merge_asof(
    frames_df, csv_sig,
    left_on="t_sec", right_on="timestamp",
    direction="nearest",
    tolerance=1.0/(2*fps)
)

aligned = aligned.ffill().bfill()

aligned.to_csv("Behavioural_Cloning/data/Processed/labels.csv", index=False)
print("Gespeichert: labels_per_frame.csv")
print(aligned.head())
