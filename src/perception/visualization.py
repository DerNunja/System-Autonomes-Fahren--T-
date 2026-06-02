from __future__ import annotations

import cv2
import numpy as np


def draw_curvature_preview(vis_bgr: np.ndarray, ego_lane) -> np.ndarray:
    if ego_lane is None or not ego_lane.has_ego_lane:
        return vis_bgr
    if not ego_lane.centerline_px:
        return vis_bgr

    bottom_center = max(ego_lane.centerline_px, key=lambda p: p[1])
    x0, y0 = int(bottom_center[0]), int(bottom_center[1])

    k = float(getattr(ego_lane, "curvature_preview", 0.0))
    k_clamped = max(-0.02, min(0.02, k))

    side_scale = 8000.0
    dy = -80
    dx = int(k_clamped * side_scale)

    cv2.arrowedLine(
        vis_bgr,
        (x0, y0),
        (x0 + dx, y0 + dy),
        (0, 0, 255),
        2,
        tipLength=0.25,
    )

    cv2.putText(
        vis_bgr,
        f"curv_prev={k:+.4f}",
        (10, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    return vis_bgr


def draw_ego_centerline(vis_bgr: np.ndarray, ego_lane) -> np.ndarray:
    if ego_lane is None or not ego_lane.has_ego_lane:
        return vis_bgr
    if not ego_lane.centerline_px:
        return vis_bgr

    pts = np.array(ego_lane.centerline_px, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        vis_bgr,
        [pts],
        isClosed=False,
        color=(0, 255, 255),
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    bottom_pt = max(ego_lane.centerline_px, key=lambda p: p[1])
    cv2.circle(
        vis_bgr,
        (int(bottom_pt[0]), int(bottom_pt[1])),
        5,
        (0, 255, 255),
        -1,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        vis_bgr,
        f"ego_offset={ego_lane.lateral_offset_px:+.1f}px",
        (10, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis_bgr


def show_perception_windows(original_bgr, annotated_bgr) -> int:
    cv2.imshow("Original Video", original_bgr)
    cv2.imshow("Lane Perception", annotated_bgr)
    return cv2.waitKey(1) & 0xFF
