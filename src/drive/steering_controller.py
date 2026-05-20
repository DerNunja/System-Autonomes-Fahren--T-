from dataclasses import dataclass
from collections import deque
from typing import Optional, Deque, Tuple
import time
import math

@dataclass
class SteeringCommand:
    steer_rad: float      # Lenkwinkel in rad
    steer_norm: float     # normiert -1..+1
    error_offset_m: float
    d_offset_dt: float
    ff_term: float        # reiner Feed-Forward-Beitrag


class LateralController:
    """
    Stanley-ähnlicher Lateralregler mit Feed-Forward über Spurkrümmung.

      δ = δ_ff + δ_fb
      δ_ff  ~ K_ff * kappa
      δ_fb  ~ heading_error + atan2(k * e, v + v0)
    """

    def __init__(
        self,
        max_steer_rad: float = 0.5,   # Sättigung (ca. 30°)
        k_stanley: float = 1.0,       # Gain für Querfehler
        v_ref: float = 20.0,          # "virtuelle" Geschwindigkeit [m/s]
        k_ff: float = 8.0,            # Feed-Forward Gain (abhängig von Fahrzeug)
        history_window_s: float = 0.5
    ):
        self.max_steer_rad = max_steer_rad
        self.k_stanley = k_stanley
        self.v_ref = v_ref
        self.k_ff = k_ff
        self.history_window_s = history_window_s
        self._history: Deque[Tuple[float, float]] = deque()   # (t, offset_m)

    def _update_history(self, offset_m: float, t: Optional[float] = None) -> float:
        if t is None:
            t = time.time()
        self._history.append((t, offset_m))
        while self._history and (t - self._history[0][0] > self.history_window_s):
            self._history.popleft()

        if len(self._history) < 2:
            return 0.0

        t0, o0 = self._history[0]
        t1, o1 = self._history[-1]
        dt = max(1e-3, t1 - t0)
        return (o1 - o0) / dt    # m/s

    def update(
        self,
        offset_m: float,
        heading_error_rad: float,
        curvature_preview: float,
        t: Optional[float] = None,
    ) -> SteeringCommand:

        d_offset_dt = self._update_history(offset_m, t)

        # --- Feed-Forward: Spurkrümmung -> Lenkwinkel ---
        # kappa > 0 (Linkskurve) => positiver Lenkwinkel
        steer_ff = self.k_ff * curvature_preview

        # --- Stanley-Feedback ---
        # Querfehler e = offset_m (rechts positiv, links negativ)
        e = offset_m
        v = self.v_ref

        # klassisch: δ_fb = θ_e + atan2(k*e, v)
        steer_fb = heading_error_rad + math.atan2(self.k_stanley * e, max(0.1, v))

        steer_rad = steer_ff + steer_fb

        # Sättigung
        steer_rad = max(-self.max_steer_rad, min(self.max_steer_rad, steer_rad))
        steer_norm = steer_rad / self.max_steer_rad

        return SteeringCommand(
            steer_rad=float(steer_rad),
            steer_norm=float(steer_norm),
            error_offset_m=float(e),
            d_offset_dt=float(d_offset_dt),
            ff_term=float(steer_ff),
        )
