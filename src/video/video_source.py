from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass
class VideoSourceConfig:
    use_live_source: bool = False
    video_path: str = "/home/konrada/projects/Uni/ProjektAutonomesFahren/src/data/Recordings/Video/output_h264.mp4"
    live_source: int = 0


class VideoSource:
    def __init__(self, config: VideoSourceConfig):
        self.config = config
        self.capture = None

    def open(self) -> None:
        if self.config.use_live_source:
            print(f"[INFO] Öffne Live-Quelle: {self.config.live_source}")
            self.capture = cv2.VideoCapture(self.config.live_source, cv2.CAP_DSHOW)
        else:
            print(f"[INFO] Öffne Videodatei: {self.config.video_path}")
            self.capture = cv2.VideoCapture(self.config.video_path)

        if not self.capture.isOpened():
            raise RuntimeError("Video-/Livequelle konnte nicht geöffnet werden.")

    def read(self):
        if self.capture is None:
            raise RuntimeError("VideoSource wurde nicht geöffnet.")
        return self.capture.read()

    def fps(self, target_fps: float | None) -> float:
        if target_fps:
            return target_fps

        if self.capture is None:
            return 30.0

        source_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or source_fps > 200:
            return 30.0
        return source_fps

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
