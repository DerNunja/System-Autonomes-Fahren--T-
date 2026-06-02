from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    from .steering_controller import LateralController, SteeringCommand
except ImportError:
    try:
        from drive.steering_controller import LateralController, SteeringCommand
    except ImportError:
        from steering_controller import LateralController, SteeringCommand


@dataclass
class LaneState:
    t_ms: int
    has_ego_lane: bool
    offset_m: float
    heading_error_rad: float
    curvature_preview: float
    quality: float


@dataclass
class DriveControllerResult:
    command: SteeringCommand
    valid: bool
    quality: float
    reason: str
    t_ms: int


def _neutral_command() -> SteeringCommand:
    return SteeringCommand(
        steer_rad=0.0,
        steer_norm=0.0,
        error_offset_m=0.0,
        d_offset_dt=0.0,
        ff_term=0.0,
    )


class DriveController:
    def __init__(
        self,
        max_steer_rad: float,
        k_stanley: float,
        v_ref: float,
        k_ff: float,
        history_window_s: float,
        min_quality: float = 0.3,
    ):
        self.min_quality = min_quality
        self.controller = LateralController(
            max_steer_rad=max_steer_rad,
            k_stanley=k_stanley,
            v_ref=v_ref,
            k_ff=k_ff,
            history_window_s=history_window_s,
        )

    def update_from_lanestate(
        self,
        payload: dict[str, Any],
        t: Optional[float] = None,
    ) -> DriveControllerResult:
        try:
            lane = self._parse_lanestate(payload)
        except (TypeError, ValueError, KeyError) as exc:
            return DriveControllerResult(
                command=_neutral_command(),
                valid=False,
                quality=0.0,
                reason=f"invalid_payload:{exc}",
                t_ms=0,
            )

        if not lane.has_ego_lane:
            return self._invalid_result(lane, "no_ego_lane")

        if lane.quality < self.min_quality:
            return self._invalid_result(lane, "low_quality")

        command = self.controller.update(
            offset_m=lane.offset_m,
            heading_error_rad=lane.heading_error_rad,
            curvature_preview=lane.curvature_preview,
            t=t,
        )
        return DriveControllerResult(
            command=command,
            valid=True,
            quality=lane.quality,
            reason="ok",
            t_ms=lane.t_ms,
        )

    def timeout_result(self) -> DriveControllerResult:
        return DriveControllerResult(
            command=_neutral_command(),
            valid=False,
            quality=0.0,
            reason="timeout",
            t_ms=0,
        )

    def _invalid_result(self, lane: LaneState, reason: str) -> DriveControllerResult:
        return DriveControllerResult(
            command=_neutral_command(),
            valid=False,
            quality=lane.quality,
            reason=reason,
            t_ms=lane.t_ms,
        )

    def _parse_lanestate(self, payload: dict[str, Any]) -> LaneState:
        return LaneState(
            t_ms=int(payload.get("t_ms", 0)),
            has_ego_lane=bool(payload["has_ego_lane"]),
            offset_m=float(payload["offset_m"]),
            heading_error_rad=float(payload["heading_error_rad"]),
            curvature_preview=float(payload["curvature_preview"]),
            quality=float(payload.get("quality", 0.0)),
        )
