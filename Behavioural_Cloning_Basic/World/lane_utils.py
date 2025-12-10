# angepasste version von firas lane_detection_client.py
from __future__ import annotations
from typing import List, Tuple

LaneXY = List[Tuple[float, float]]  # (x_model, y_canon)


def lanes_to_lanestate(lanes_xy, model_w, lane_width_m=3.7):
    """
    Schätzt lane_center (in Metern) und curvature (Dummy) aus den erkannten Lanes.

    lanes_xy: Liste von Lanes, jede Lane = Liste von (x_model, y_canon)
    model_w:  Breite des Modell-Eingangs (cfg.train_width)
    lane_width_m: angenommene Spurbreite (ca. 3.7m)
    """

    # Mindestens 2 Lanes nötig (linke & rechte Spurbegrenzung)
    if len(lanes_xy) < 2:
        return 0.0, 0.0

    lane_a = lanes_xy[0]
    lane_b = lanes_xy[1]

    def bottom_point(lane):
        if not lane:
            return None
        # Punkt mit größter y-Koordinate = am nächsten zur Kamera
        return max(lane, key=lambda p: p[1])

    pa = bottom_point(lane_a)
    pb = bottom_point(lane_b)
    if pa is None or pb is None:
        return 0.0, 0.0

    x_a, _ = pa
    x_b, _ = pb

    # sortieren: links / rechts
    x_left, x_right = sorted([x_a, x_b])

    # Spurmitte in Modellpix koord.
    lane_center_x = 0.5 * (x_left + x_right)

    # Fahrzeugmitte = Bildmitte
    img_center_x = model_w / 2.0

    offset_px = lane_center_x - img_center_x
    lane_width_px = max(1e-3, abs(x_right - x_left))

    meters_per_px = lane_width_m / lane_width_px
    lane_center_m = offset_px * meters_per_px

    curvature = 0.0  # vorerst Dummy

    return lane_center_m, curvature