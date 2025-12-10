from __future__ import annotations
from collections import deque
import numpy as np


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
    """Aggregierte Infos zur aktuellen Fahrspur (noch in Pixelkoordinaten + grob in Metern)."""
    has_ego_lane: bool
    centerline_px: List[Point2D]   # Mittelspur als Pixel-Polyline
    lateral_offset_px: float       # Abweichung der Spurmitte von Bildmitte (u - W/2)
    heading_px_rad: float          # grobe Richtung (aus den unteren Punkten)
    quality: float                 # 0..1, z.B. mittlere Score der Spurbegrenzungen
    lane_width_px: float = 0.0     # geschätzte Spurbreite im unteren Bereich (Pixel)
    lateral_offset_m: float = 0.0  # Abweichung in Metern (via lane_width_px -> m)
    curvature_preview: float = 0.0 # >0: Spur biegt nach rechts, <0: nach links


@dataclass
class WorldModelState:
    timestamp_ms: int
    lanes_px: List[LanePixelPolyline] = field(default_factory=list)
    ego_lane: Optional[EgoLaneState] = None


# ---------- Weltmodell ---------- #

class WorldModel:
    def __init__(self, img_width: int, img_height: int, lane_width_m: float = 3.7,
                 ema_alpha: float = 0.2):
        self.img_width = img_width
        self.img_height = img_height
        self.lane_width_m = lane_width_m
        self.state: Optional[WorldModelState] = None

        # History/Glättung
        self.ema_alpha = ema_alpha
        self._ema_offset_m: Optional[float] = None
        self._ema_heading: Optional[float] = None

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

    def _estimate_curvature_from_centerline(
        self,
        centerline_px: List[Point2D],
    ) -> float:
        """
        Sehr einfache Krümmungs-Heuristik aus der Mittelspur.
        Sign:
          kappa > 0  => Spur biegt nach rechts
          kappa < 0  => Spur biegt nach links
        """
        if len(centerline_px) < 3:
            return 0.0

        # nach Bild-y sortieren: von oben (kleine v) nach unten (große v)
        cl_sorted = sorted(centerline_px, key=lambda p: p[1])

        # ein Fenster über die unteren ~10 Punkte nehmen
        bottom = cl_sorted[-1]
        window_size = min(10, len(cl_sorted))
        top = cl_sorted[-window_size]

        u_top, v_top = top
        u_bot, v_bot = bottom

        du = u_top - u_bot          # seitliche Änderung
        dv = v_bot - v_top          # „vorwärts“ im Bild

        if dv < 1e-3:
            return 0.0

        # einfache Normierung: Pixel-„Krümmung“
        kappa_pix = du / dv   # >0: Spur wandert nach rechts, <0: nach links

        # auf einen für den Regler sinnvollen Bereich skalieren
        # (Faktor musst du ggf. anpassen)
        kappa = 0.1 * kappa_pix

        # Clamp gegen Ausreißer
        kappa = float(np.clip(kappa, -0.02, 0.02))
        return kappa

    def _estimate_ego_lane(self, lanes_px: List[LanePixelPolyline]) -> Optional[EgoLaneState]:
        """
        Sehr einfache Heuristik:
          - nehme für jede Lane den "untersten" Punkt (max v)
          - klassifiziere nach links/rechts relativ zum Bildzentrum
          - wähle die beste linke & rechte Lane nach Score
          - baue eine Mittelspur daraus
          - schätze Spurbreite in Pixeln und lateral_offset in Metern
        """
        if not lanes_px:
            return EgoLaneState(
                has_ego_lane=False,
                centerline_px=[],
                lateral_offset_px=0.0,
                heading_px_rad=0.0,
                quality=0.0,
                lane_width_px=0.0,
                lateral_offset_m=0.0,
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
                lane_width_px=0.0,
                lateral_offset_m=0.0,
            )

        left_candidates = []
        right_candidates = []

        for lane, (u, v) in lane_bottom_points:
            if u < u_center:
                left_candidates.append((lane, u, v))
            else:
                right_candidates.append((lane, u, v))

        MIN_LANE_SCORE = 0.3 

        def pick_best(candidates):
            if not candidates:
                return None
            best = max(
                candidates,
                key=lambda t: (t[0].score, -abs(t[1] - u_center))
            )
            if best[0].score < MIN_LANE_SCORE:
                return None
            return best

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
                lane_width_px=0.0,
                lateral_offset_m=0.0,
            )

        left_lane, u_left_bottom, v_left_bottom = left_pick
        right_lane, u_right_bottom, v_right_bottom = right_pick

        # Mittelspur als Mittelwert der beiden Lane-Polylines in gemeinsamen v-Bereichen
        centerline_px: List[Point2D] = self._build_centerline_from_lr(left_lane, right_lane)

        if not centerline_px:
            return EgoLaneState(
                has_ego_lane=False,
                centerline_px=[],
                lateral_offset_px=0.0,
                heading_px_rad=0.0,
                quality=0.0,
                lane_width_px=0.0,
                lateral_offset_m=0.0,
            )


        # Lateraloffset: Differenz zwischen Spurmitte am unteren Bildrand und Bildmitte
        bottom_center = max(centerline_px, key=lambda p: p[1])
        lateral_offset_px = bottom_center[0] - u_center

        # Spurbreite: Differenz der untersten Punkte links/rechts
        lane_width_px = max(1e-6, abs(u_right_bottom - u_left_bottom))

        # px -> m: angenommene lane_width_m / lane_width_px
        meters_per_px = self.lane_width_m / lane_width_px
        lateral_offset_m = lateral_offset_px * meters_per_px
        curvature_preview = self._estimate_curvature_from_centerline(
            centerline_px=centerline_px,
        )

        # Heading grob: Steigung zwischen unterstem und Punkt etwa Bildmitte
        if len(centerline_px) >= 2:
            p1 = bottom_center
            p2 = min(centerline_px, key=lambda p: abs(p[1] - (self.img_height * 0.5)))
            dx = p2[0] - p1[0]
            dy = p1[1] - p2[1]  # Bildschirmkoordinaten: v nach unten
            if abs(dx) > 1e-3 and abs(dy) > 1e-3:
                heading_rad = - (dx / dy)  # grober Proxy
            else:
                heading_rad = 0.0
        else:
            heading_rad = 0.0

        quality = 0.5 * (left_lane.score + right_lane.score)

        # --- EMA-Glättung ---
        alpha = self.ema_alpha

        if self._ema_offset_m is None:
            self._ema_offset_m = lateral_offset_m
        else:
            self._ema_offset_m = (1 - alpha) * self._ema_offset_m + alpha * lateral_offset_m

        if self._ema_heading is None:
            self._ema_heading = heading_rad
        else:
            self._ema_heading = (1 - alpha) * self._ema_heading + alpha * heading_rad

        smooth_offset_m = self._ema_offset_m
        smooth_heading  = self._ema_heading

        if quality < 0.1 or len(centerline_px) < 5:
            return EgoLaneState(
                has_ego_lane=False,
                centerline_px=[],
                lateral_offset_px=0.0,
                heading_px_rad=0.0,
                quality=float(quality),
                lane_width_px=float(lane_width_px),
                lateral_offset_m=0.0,
            )

        # --- Steering-Preview: einfache Krümmungsabschätzung der Centerline ---
        # Wir nehmen den untersten und einen weiter oben liegenden Punkt und schauen,
        # wie stark die Spur "nach rechts" oder "nach links" wandert.
        """
        curvature_preview = 0.0
        if len(centerline_px) >= 3:
            # sortiert nach v (Bild y): von oben (kleine v) nach unten (große v)
            cl_sorted = sorted(centerline_px, key=lambda p: p[1])
            top = cl_sorted[0]
            mid = cl_sorted[len(cl_sorted) // 2]
            bottom = cl_sorted[-1]

            # Wir interpretieren v nach unten als "nach vorne" (weg vom Auto)
            # delta_u > 0 => Spur geht nach rechts, delta_u < 0 => nach links
            du = top[0] - bottom[0]   # Unterschied in x von unten nach oben
            dv = bottom[1] - top[1]   # "Vorwärts"-Distanz in Pixel

            if dv > 1e-3:
                curvature_preview = du / dv   # einfache Normierung

        # (Optional: EMA auch auf curvature_preview anwenden, falls nötig)
        """
        return EgoLaneState(
            has_ego_lane=True,
            centerline_px=centerline_px,
            lateral_offset_px=float(lateral_offset_px),
            heading_px_rad=float(smooth_heading),
            quality=float(quality),
            lane_width_px=float(lane_width_px),
            lateral_offset_m=float(smooth_offset_m),
            curvature_preview=float(curvature_preview),
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
