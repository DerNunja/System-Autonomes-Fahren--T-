from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Tuple

import cv2

from runtime.stats import RuntimeStats, pct
from video.video_source import VideoSource, VideoSourceConfig
from video.video_transport import VideoStreamSender


@dataclass
class VideoSenderConfig:
    stream_name: str = "Demo"
    use_live_source: bool = False
    video_path: str = "/home/konrada/projects/Uni/ProjektAutonomesFahren/src/data/Recordings/Video/output_h264.mp4"
    live_source: int = 0
    target_size: Optional[Tuple[int, int]] = (640, 360)
    target_fps: Optional[float] = 60.0
    preview_window: str = "Sender Preview"


class VideoSenderApp:
    def __init__(self, config: VideoSenderConfig):
        self.config = config
        source_config = VideoSourceConfig(
            use_live_source=config.use_live_source,
            video_path=config.video_path,
            live_source=config.live_source,
        )
        self.source = VideoSource(source_config)
        self.stats = RuntimeStats()

    def run(self) -> None:
        try:
            self.source.open()
            fps = self.source.fps(self.config.target_fps)
            frame_time = 1.0 / fps if fps > 0 else 0.0
            next_send_time = time.time()

            with VideoStreamSender(self.config.stream_name) as stream:
                print(
                    f"[INFO] Video Sender gestartet: {self.config.stream_name} "
                    f"({fps:.1f} FPS, target size={self.config.target_size})"
                )
                next_send_time = self._run_loop(stream, frame_time, next_send_time)
        finally:
            self.source.close()
            cv2.destroyAllWindows()
            self._print_summary()
            print("[INFO] Sender beendet.")

    def _run_loop(self, stream: VideoStreamSender, frame_time: float, next_send_time: float) -> float:
        while True:
            with self.stats.measure("read") as read_dt:
                ret, frame = self.source.read()

            if not ret:
                if self.config.use_live_source:
                    time.sleep(0.01)
                    continue
                print("[INFO] Video zu Ende.")
                break

            with self.stats.measure("resize") as resize_dt:
                if self.config.target_size is not None:
                    frame = cv2.resize(frame, self.config.target_size, interpolation=cv2.INTER_AREA)

            with self.stats.measure("send") as send_dt:
                stream.send(frame)

            self.stats.increment_frame()

            with self.stats.measure("preview") as preview_dt:
                cv2.imshow(self.config.preview_window, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("[INFO] Abbruch durch Benutzer.")
                    break

            sleep_dt = self._sleep_until_next_frame(frame_time, next_send_time)
            next_send_time = sleep_dt["next_send_time"]
            self.stats.add("sleep", sleep_dt["elapsed_s"])

            self._print_frame_log(read_dt[0], resize_dt[0], send_dt[0], preview_dt[0], sleep_dt["elapsed_s"])

        return next_send_time

    def _sleep_until_next_frame(self, frame_time: float, next_send_time: float) -> dict[str, float]:
        if frame_time <= 0:
            return {"elapsed_s": 0.0, "next_send_time": next_send_time}

        next_send_time += frame_time
        now = time.time()
        sleep_time = next_send_time - now
        if sleep_time <= 0:
            return {"elapsed_s": 0.0, "next_send_time": now}

        start = time.time()
        time.sleep(sleep_time)
        return {"elapsed_s": time.time() - start, "next_send_time": next_send_time}

    def _print_frame_log(
        self,
        read_dt: float,
        resize_dt: float,
        send_dt: float,
        preview_dt: float,
        sleep_dt: float,
    ) -> None:
        loop_dt = read_dt + resize_dt + send_dt + preview_dt + sleep_dt
        fps_inst = 1.0 / loop_dt if loop_dt > 0 else 0.0
        print(
            f"[SEND {self.stats.frame_count:05d}] "
            f"read={read_dt*1000:5.2f} ms  "
            f"resize={resize_dt*1000:5.2f} ms  "
            f"send={send_dt*1000:5.2f} ms  "
            f"preview={preview_dt*1000:5.2f} ms  "
            f"sleep={sleep_dt*1000:5.2f} ms  "
            f"eff_FPS={fps_inst:5.1f}"
        )

    def _print_summary(self) -> None:
        if self.stats.frame_count <= 0:
            return

        elapsed = self.stats.elapsed_s()
        print(
            f"\n[STATS] Sent {self.stats.frame_count} frames in "
            f"{elapsed:.2f}s -> effective send FPS = {self.stats.average_fps():.2f}"
        )

        total_ms = elapsed / self.stats.frame_count * 1000.0 if elapsed > 0 else 0.0
        print("[BREAKDOWN SENDER] Durchschnitt pro Frame:")
        for stage in ("read", "resize", "send", "preview", "sleep"):
            avg_ms = self.stats.average_ms(stage)
            print(f"  {stage:<9} = {avg_ms:6.2f} ms ({pct(avg_ms, total_ms):5.1f}% der Zeit)")
        print(f"  total     = {total_ms:6.2f} ms (gemittelte Frame-Dauer)")
