from __future__ import annotations

from dataclasses import dataclass
import time

import cv2

from World.world_model import LaneDetResult, WorldModel
from perception.lane_detector import LaneDetector
from perception.lanestate import LaneStatePublisher, LaneStatePublisherConfig
from perception.visualization import draw_curvature_preview, draw_ego_centerline, show_perception_windows
from runtime.stats import RuntimeStats, pct
from video.video_transport import VideoStreamReceiver


@dataclass
class PerceptionReceiverConfig:
    stream_name: str = "Demo"
    broker: str = "localhost"
    mqtt_port: int = 1883
    lanestate_topic: str = "sensor/lanestate"
    source_discovery_timeout_s: float = 5.0


class PerceptionReceiverApp:
    def __init__(self, config: PerceptionReceiverConfig):
        self.config = config
        self.detector = LaneDetector()
        self.publisher = LaneStatePublisher(
            LaneStatePublisherConfig(
                broker=config.broker,
                port=config.mqtt_port,
                topic=config.lanestate_topic,
            )
        )
        self.stats = RuntimeStats()
        self.world_model: WorldModel | None = None
        self.first_ts = None
        self.last_ts = None
        self.prev_ts_for_avg = None
        self.unique_ts_steps = 0

    def run(self) -> None:
        self.publisher.start()
        try:
            self._print_sources()
            cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Lane Perception", cv2.WINDOW_NORMAL)

            print(f"[INFO] Verbinde mit Video-Stream: {self.config.stream_name}")
            with VideoStreamReceiver(self.config.stream_name) as stream:
                print("[INFO] Receiver verbunden, warte auf Frames... (ESC/q zum Beenden)")
                self._run_loop(stream)
        finally:
            self.publisher.stop()
            cv2.destroyAllWindows()
            self._print_summary()
            print("[INFO] Receiver beendet.")

    def _run_loop(self, stream: VideoStreamReceiver) -> None:
        while stream.is_connected:
            loop_t0 = time.time()

            with self.stats.measure("receive") as receive_dt:
                ts, frame = stream.read()

            if frame is None:
                continue

            height, width = frame.shape[:2]
            if self.world_model is None:
                self.world_model = WorldModel(img_width=width, img_height=height)

            self._update_video_timestamp_stats(ts)
            detection = self.detector.process(frame)
            self.stats.add("model", detection.model_dt)

            world_state = self._update_world_model(ts, width, height, detection)
            detection.vis_bgr = self._draw_world_overlays(detection.vis_bgr, world_state.ego_lane)

            timestamp_ms = int(ts) if ts is not None else 0
            self.publisher.publish(timestamp_ms, world_state.ego_lane)

            with self.stats.measure("display") as display_dt:
                key = show_perception_windows(frame, detection.vis_bgr)

            loop_dt = time.time() - loop_t0
            self.stats.add("loop", loop_dt)
            self.stats.increment_frame()
            self._print_frame_log(ts, receive_dt[0], detection.model_dt, display_dt[0], loop_dt, detection)

            if key in (27, ord("q")):
                break

    def _print_sources(self) -> None:
        print("[INFO] Suche Video-Quellen...")
        sources = VideoStreamReceiver.find_sources(timeout=self.config.source_discovery_timeout_s)
        if not sources:
            print("[WARN] Keine Video-Quellen gefunden!")
            return

        print("[INFO] Gefundene Quellen:")
        for source in sources:
            print(" -", source.name)

    def _update_world_model(self, ts, width: int, height: int, detection):
        if self.world_model is None:
            raise RuntimeError("WorldModel wurde nicht initialisiert.")

        lane_res = LaneDetResult(
            timestamp_ms=int(ts) if ts is not None else 0,
            img_width=width,
            img_height=height,
            lanes_model_xy=detection.lanes_xy,
            lanes_info=detection.lanes_info,
            model_width=self.detector.cfg.train_width,
            canon_height=590,
        )
        return self.world_model.update_from_lane_detection(lane_res)

    def _draw_world_overlays(self, vis_bgr, ego_lane):
        vis_bgr = draw_ego_centerline(vis_bgr, ego_lane)
        if ego_lane and ego_lane.has_ego_lane:
            vis_bgr = draw_curvature_preview(vis_bgr, ego_lane)
        return vis_bgr

    def _update_video_timestamp_stats(self, ts) -> None:
        if ts is None:
            return
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts

        if self.prev_ts_for_avg is not None and ts > self.prev_ts_for_avg:
            self.unique_ts_steps += 1
        self.prev_ts_for_avg = ts

    def _print_frame_log(self, ts, receive_dt: float, model_dt: float, display_dt: float, loop_dt: float, detection) -> None:
        loop_fps_inst = 1.0 / loop_dt if loop_dt > 0 else 0.0
        ts_text = f"{ts:13.3f}" if ts is not None else "         None"
        print(
            f"[FRAME {self.stats.frame_count:05d}] "
            f"ts={ts_text} ms  "
            f"receive={receive_dt*1000:5.2f} ms  "
            f"model={model_dt*1000:5.2f} ms  "
            f"display={display_dt*1000:5.2f} ms  "
            f"loop={loop_dt*1000:5.2f} ms  "
            f"model_FPS={detection.fps_inst:5.1f}  "
            f"loop_FPS={loop_fps_inst:5.1f}  "
            f"lanes={detection.n_lanes}"
        )

    def _print_summary(self) -> None:
        if self.stats.frame_count <= 0:
            return

        elapsed = self.stats.elapsed_s()
        print(
            f"\n[STATS] Processed {self.stats.frame_count} frames in "
            f"{elapsed:.2f}s -> avg loop FPS = {self.stats.average_fps():.2f}"
        )

        avg_loop_ms = self.stats.average_ms("loop")
        for stage, label in (("receive", "avg_receive_time"), ("model", "avg_model_time"), ("display", "avg_display_time")):
            avg_ms = self.stats.average_ms(stage)
            print(f"[BREAKDOWN] {label:<17} = {avg_ms:6.2f} ms/frame ({pct(avg_ms, avg_loop_ms):5.1f}% der Loop-Zeit)")
        print(f"[BREAKDOWN] avg_loop_time     = {avg_loop_ms:6.2f} ms/frame (inkl. Receive + Model + Display + sonstiges)")

        if self.first_ts is not None and self.last_ts is not None and self.last_ts > self.first_ts and self.unique_ts_steps > 0:
            total_video_time_sec = (self.last_ts - self.first_ts) / 1000.0
            avg_video_fps = self.unique_ts_steps / total_video_time_sec
            print(f"[VIDEO] avg_video_fps (unique ts steps) = {avg_video_fps:.2f} FPS")
