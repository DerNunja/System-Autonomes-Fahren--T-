from __future__ import annotations

import json
from dataclasses import dataclass

import paho.mqtt.client as mqtt


@dataclass
class LaneStatePublisherConfig:
    broker: str = "localhost"
    port: int = 1883
    topic: str = "sensor/lanestate"
    client_id: str = "lane-perception"


def build_lanestate_payload(timestamp_ms: int, ego_lane) -> dict:
    if ego_lane is None or not ego_lane.has_ego_lane:
        return {
            "t_ms": timestamp_ms,
            "has_ego_lane": False,
            "offset_m": 0.0,
            "heading_error_rad": 0.0,
            "curvature_preview": 0.0,
            "quality": 0.0,
            "lane_center": 0.0,
            "curvature": 0.0,
        }

    offset_m = float(ego_lane.lateral_offset_m)
    curvature = float(ego_lane.curvature_preview)
    return {
        "t_ms": timestamp_ms,
        "has_ego_lane": True,
        "offset_m": offset_m,
        "heading_error_rad": float(ego_lane.heading_px_rad),
        "curvature_preview": curvature,
        "quality": float(ego_lane.quality),
        "lateral_offset_px": float(ego_lane.lateral_offset_px),
        "lane_width_px": float(ego_lane.lane_width_px),
        "lane_center": offset_m,
        "curvature": curvature,
    }


class LaneStatePublisher:
    def __init__(self, config: LaneStatePublisherConfig):
        self.config = config
        self.client = mqtt.Client(client_id=config.client_id)

    def start(self) -> None:
        self.client.connect(self.config.broker, self.config.port, keepalive=60)
        self.client.loop_start()

    def publish(self, timestamp_ms: int, ego_lane) -> dict:
        payload = build_lanestate_payload(timestamp_ms, ego_lane)
        self.client.publish(self.config.topic, json.dumps(payload))
        return payload

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
