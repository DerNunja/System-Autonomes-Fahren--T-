import cv2
from pathlib import Path
from typing import Generator
import time


def frame_stream_from_video(
    video_path: str | Path,
    loop: bool = False,
    simulate_realtime: bool = True,
    target_size: tuple[int, int] | None = None,   # NEW: (W, H)
) -> Generator[tuple[int, float, "cv2.Mat"], None, None]:
    """
    Simulates a video stream with optional downscaling.

    Args:
        target_size: (width, height) to resize each frame.
                     Example: (640, 360) or (320, 180)
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 60.0

    frame_duration = 1.0 / fps
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                start_time = time.time()
                continue
            break

        # Optional lower resolution
        if target_size is not None:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

        ts = frame_idx * frame_duration

        if simulate_realtime:
            now = time.time() - start_time
            sleep_time = ts - now
            if sleep_time > 0:
                time.sleep(sleep_time)

        yield frame_idx, ts, frame
        frame_idx += 1

    cap.release()