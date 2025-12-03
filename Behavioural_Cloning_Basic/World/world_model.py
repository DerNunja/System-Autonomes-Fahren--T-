# world_model.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

Point2D = Tuple[float, float]  # (x, y)

@dataclass
class LaneDetResult:
    """
    Roh-Ergebnis aus der LaneDetection für EIN Frame.
    - lanes_model_xy: Punkte im "gemischten" Raum (x_model, y_canon).
    - lanes_info: Liste von Dicts aus pred2coords_mixed (lane_id, score, n_points).
    """
    timestamp_ms: int
    img_width: int
    img_height: int

    lanes_model_xy: List[List[Point2D]]     # [(x_model, y_canon), ...] pro Lane
    lanes_info: List[dict]

    model_width: int                        # cfg.train_width
    canon_height: int = 590                 # CANON_H = 590 für CULane


# ---------- Interne Repräsentation im Bildraum ---------- #

@dataclass
class LanePixelPolyline:
    """Lane in Bildpixeln (u,v)."""
    lane_id: int
    score: float
    points: List[Point2D]        # [(u, v), ...] von vorne nach hinten (v klein->groß)


@dataclass
class EgoLaneState:
    """Aggregierte Infos zur aktuellen Fahrspur (noch in Pixelkoordinaten)."""
    has_ego_lane: bool
    centerline_px: List[Point2D]   # Mittelspur als Pixel-Polyline
    lateral_offset_px: float       # Abweichung der Spurmitte von Bildmitte (u - W/2)
    heading_px_rad: float          # grobe Richtung (aus den unteren Punkten)
    quality: float                 # 0..1, z.B. mittlere Score der Spurbegrenzungen


@dataclass
class WorldModelState:
    timestamp_ms: int
    lanes_px: List[LanePixelPolyline] = field(default_factory=list)
    ego_lane: Optional[EgoLaneState] = None


# ---------- Weltmodell ---------- #

class WorldModel:
    """
    Einfaches Weltmodell, das nur LaneDetection in 2D (Bildpixel) verwaltet.
    """

    def __init__(self, img_width: int, img_height: int):
        self.img_width = img_width
        self.img_height = img_height
        self.state: Optional[WorldModelState] = None

    # ----- Public API ----- #

    def update_from_lane_detection(self, det: LaneDetResult) -> WorldModelState:
        """
        Nimmt ein LaneDetResult und aktualisiert den Weltmodell-Zustand.
        """
        lanes_px = self._project_lanes_to_pixel(det)
        ego_lane = self._estimate_ego_lane(lanes_px)

        self.state = WorldModelState(
            timestamp_ms=det.timestamp_ms,
            lanes_px=lanes_px,
            ego_lane=ego_lane,
        )
        return self.state

    # ----- Interne Helfer ----- #

    def _project_lanes_to_pixel(self, det: LaneDetResult) -> List[LanePixelPolyline]:
        """
        Überführt (x_model, y_canon) -> Bildpixel (u,v).
        Nutzt die gleichen Skalierungen wie draw_lanes_mixed.
        """
        sx = det.img_width / det.model_width
        sy = det.img_height / det.canon_height

        lanes_px: List[LanePixelPolyline] = []

        for lane_pts, info in zip(det.lanes_model_xy, det.lanes_info):
            pts_px: List[Point2D] = []
            for (x_model, y_canon) in lane_pts:
                u = x_model * sx
                v = y_canon * sy
                pts_px.append((u, v))

            # zur Sicherheit nach v sortieren (oben->unten)
            pts_px.sort(key=lambda p: p[1])

            lanes_px.append(
                LanePixelPolyline(
                    lane_id=info["lane_id"],
                    score=info["score"],
                    points=pts_px,
                )
            )

        return lanes_px

    def _estimate_ego_lane(self, lanes_px: List[LanePixelPolyline]) -> Optional[EgoLaneState]:
        """
        Sehr einfache Heuristik:
          - nehme für jede Lane den "untersten" Punkt (max v)
          - klassifiziere nach links/rechts relativ zum Bildzentrum
          - wähle die beste linke & rechte Lane nach Score
          - baue eine Mittelspur daraus
        Alles noch in Pixeln.
        """
        if not lanes_px:
            return EgoLaneState(
                has_ego_lane=False,
                centerline_px=[],
                lateral_offset_px=0.0,
                heading_px_rad=0.0,
                quality=0.0,
            )

        u_center = self.img_width / 2.0

        # Unterste Punkte suchen
        lane_bottom_points = []
        for lane in lanes_px:
            if not lane.points:
                continue
            bottom_pt = max(lane.points, key=lambda p: p[1])  # max v
            lane_bottom_points.append((lane, bottom_pt))

        if not lane_bottom_points:
            return EgoLaneState(
                has_ego_lane=False,
                centerline_px=[],
                lateral_offset_px=0.0,
                heading_px_rad=0.0,
                quality=0.0,
            )

        left_candidates = []
        right_candidates = []

        for lane, (u, v) in lane_bottom_points:
            if u < u_center:
                left_candidates.append((lane, u, v))
            else:
                right_candidates.append((lane, u, v))

        def pick_best(candidates):
            if not candidates:
                return None
            # Bevorzugt: hoher Score, und nah an der Bildmitte
            return max(
                candidates,
                key=lambda t: (t[0].score, -abs(t[1] - u_center))
            )

        left_pick = pick_best(left_candidates)
        right_pick = pick_best(right_candidates)

        if not left_pick or not right_pick:
            # Keine klare Fahrspur erkennbar
            return EgoLaneState(
                has_ego_lane=False,
                centerline_px=[],
                lateral_offset_px=0.0,
                heading_px_rad=0.0,
                quality=0.0,
            )

        left_lane = left_pick[0]
        right_lane = right_pick[0]

        # Mittelspur als Mittelwert der beiden Lane-Polylines in gemeinsamen v-Bereichen
        centerline_px: List[Point2D] = self._build_centerline_from_lr(left_lane, right_lane)

        if not centerline_px:
            return EgoLaneState(
                has_ego_lane=False,
                centerline_px=[],
                lateral_offset_px=0.0,
                heading_px_rad=0.0,
                quality=0.0,
            )

        # Lateraloffset: Differenz zwischen Spurmitte am unteren Bildrand und Bildmitte
        bottom_center = max(centerline_px, key=lambda p: p[1])
        lateral_offset_px = bottom_center[0] - u_center

        # Heading grob: Steigung zwischen unterstem und z.B. Punkt in der Mitte der Spur
        if len(centerline_px) >= 2:
            p1 = bottom_center
            p2 = min(centerline_px, key=lambda p: abs(p[1] - (self.img_height * 0.5)))
            dx = p2[0] - p1[0]
            dy = p1[1] - p2[1]  # Bildschirmkoordinaten: v nach unten
            heading_rad = 0.0
            if abs(dx) > 1e-3 and abs(dy) > 1e-3:
                # Richtung: wie stark "dreht" die Spur
                heading_rad = - (dx / dy)  # super grobe Proxy, kein echter Winkel
        else:
            heading_rad = 0.0

        quality = 0.5 * (left_lane.score + right_lane.score)

        return EgoLaneState(
            has_ego_lane=True,
            centerline_px=centerline_px,
            lateral_offset_px=float(lateral_offset_px),
            heading_px_rad=float(heading_rad),
            quality=float(quality),
        )

    def _build_centerline_from_lr(
        self,
        left_lane: LanePixelPolyline,
        right_lane: LanePixelPolyline,
    ) -> List[Point2D]:
        """
        Baut eine grobe Mittelspur, indem für gemeinsame v-Bereiche der Mittelwert von u genommen wird.
        Annahme: beide Polylines sind nach v sortiert.
        """
        left_pts = left_lane.points
        right_pts = right_lane.points

        if not left_pts or not right_pts:
            return []

        # Wir gehen einfach über v in einem groben Raster von unten nach oben
        v_min = max(min(p[1] for p in left_pts), min(p[1] for p in right_pts))
        v_max = min(max(p[1] for p in left_pts), max(p[1] for p in right_pts))

        if v_max <= v_min:
            return []

        num_samples = 20
        vs = [v_min + i * (v_max - v_min) / (num_samples - 1) for i in range(num_samples)]

        def interp_lane_at_v(lane_pts: List[Point2D], v_query: float) -> Optional[float]:
            # einfache lineare Interpolation in v
            pts = lane_pts
            # passende Segmente suchen
            for i in range(len(pts) - 1):
                v0 = pts[i][1]
                v1 = pts[i + 1][1]
                if (v0 <= v_query <= v1) or (v1 <= v_query <= v0):
                    t = 0.0 if abs(v1 - v0) < 1e-6 else (v_query - v0) / (v1 - v0)
                    u = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
                    return u
            return None

        centerline: List[Point2D] = []
        for v in vs:
            u_left = interp_lane_at_v(left_pts, v)
            u_right = interp_lane_at_v(right_pts, v)
            if u_left is not None and u_right is not None:
                u_center = 0.5 * (u_left + u_right)
                centerline.append((u_center, v))

        return centerline
