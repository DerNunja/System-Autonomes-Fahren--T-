from __future__ import annotations

from dataclasses import dataclass
import time

import cv2

from LaneDetection.lanedetec_runner import init_lanedetector, process_frame

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class LaneDetectionFrameResult:
    vis_bgr: object
    fps_inst: float
    n_lanes: int
    model_dt: float
    lanes_xy: list
    lanes_info: list


class LaneDetector:
    def __init__(self):
        print("[INFO] Lane detector init...")
        self.net, self.cfg, self.img_transforms, self.device = init_lanedetector()
        print("[INFO] Lane detector ready!")

    def process(self, bgr_frame, loop_fps: float | None = None) -> LaneDetectionFrameResult:
        self._synchronize_cuda()
        t0 = time.time()

        vis_bgr, lanes_xy, lanes_info = process_frame(
            bgr_frame,
            self.net,
            self.cfg,
            self.img_transforms,
            self.device,
        )

        self._synchronize_cuda()
        model_dt = time.time() - t0
        fps_inst = 1.0 / model_dt if model_dt > 0 else 0.0

        self._draw_fps(vis_bgr, fps_inst, loop_fps)
        return LaneDetectionFrameResult(
            vis_bgr=vis_bgr,
            fps_inst=fps_inst,
            n_lanes=len(lanes_xy),
            model_dt=model_dt,
            lanes_xy=lanes_xy,
            lanes_info=lanes_info,
        )

    def _synchronize_cuda(self) -> None:
        if HAS_TORCH and hasattr(self.device, "type") and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _draw_fps(self, vis_bgr, fps_inst: float, loop_fps: float | None) -> None:
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
