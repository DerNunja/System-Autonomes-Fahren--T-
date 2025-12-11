# Weltmodell.py
import time
import uuid
from typing import Dict, Any, List, Optional

class Weltmodell:
    """
    Lightweight world model for highway scenarios.
    Tracks:
      - ego state (position, speed, heading)
      - static objects
      - traffic signs
      - traffic lights
      - lane information (updated externally)
    """

    def __init__(self, lane_detector: Optional[Any] = None):
        # Ego vehicle state
        self.ego = {"position": (0.0, 0.0), "geschwindigkeit": 0.0, "richtung": 0.0, "lenkwinkel": 0.0}
        # Static infrastructure
        self.statische_objekte: Dict[str, Dict[str, Any]] = {}
        # Signs and traffic lights
        self.schilder: Dict[str, Dict[str, Any]] = {}
        self.ampeln: Dict[str, Dict[str, Any]] = {}
        # Lane information
        self.lane_info: Dict[str, Any] = {
            'curvature_m': None,
            'lateral_offset_m': None,
            'confidence': 0.0,
            'left_fit': None,
            'right_fit': None,
            'last_update': None
        }
        self.lane_detector = lane_detector

    # -----------------------
    # --- Helpers ---
    # -----------------------
    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def add_schild(self, position, label: str, score: float = 1.0):
        obj_id = self._new_id("schild")
        self.schilder[obj_id] = {
            'id': obj_id,
            'typ': 'schild',
            'label': label,
            'position': position,
            'score': float(score),
            'timestamp': time.time()
        }
        return obj_id

    def add_ampel(self, position, state: str, score: float = 1.0):
        obj_id = self._new_id("ampel")
        self.ampeln[obj_id] = {
            'id': obj_id,
            'typ': 'ampel',
            'state': state,
            'position': position,
            'score': float(score),
            'timestamp': time.time()
        }
        return obj_id

    def update_lane_info(self, lane_result: Optional[Any]):
        if lane_result is None:
            return
        self.lane_info.update({
            'curvature_m': float(getattr(lane_result, 'curvature_m', None)),
            'lateral_offset_m': float(getattr(lane_result, 'lateral_offset_m', None)),
            'confidence': float(getattr(lane_result, 'confidence', 0.0)),
            'left_fit': getattr(lane_result, 'left_fit', None),
            'right_fit': getattr(lane_result, 'right_fit', None),
            'last_update': time.time()
        })

    def update_from_vision(self, detections: List[Dict[str, Any]]):
        """
        Update world model with pre-computed detections.
        Each detection: {'box':[x1,y1,x2,y2], 'score':float, 'class_name':str}
        """
        for d in detections:
            cname = d.get('class_name', '').lower()
            box = d.get('box', [0, 0, 0, 0])
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            position = (float(cx), float(cy))
            score = float(d.get('score', 0.0))

            if 'light' in cname or 'ampel' in cname:
                state = 'unknown'
                if 'red' in cname: state='red'
                elif 'green' in cname: state='green'
                elif 'yellow' in cname or 'amber' in cname: state='yellow'
                self.add_ampel(position=position, state=state, score=score)
            elif 'sign' in cname or 'schild' in cname or 'speed' in cname:
                self.add_schild(position=position, label=cname, score=score)

    # -----------------------
    # --- Utilities ---
    # -----------------------
    def get_summary(self) -> Dict[str, Any]:
        return {
            'ego': self.ego.copy(),
            'lane_info': self.lane_info.copy(),
            'anzahl_schilder': len(self.schilder),
            'anzahl_ampeln': len(self.ampeln),
            'schilder': list(self.schilder.values()),
            'ampeln': list(self.ampeln.values())
        }

    def print_state(self):
        print("=== 🌍 Weltmodell-Status ===")
        print(f"Ego-Fahrzeug: {self.ego}")
        print(f"Lane confidence: {self.lane_info.get('confidence')}, curvature: {self.lane_info.get('curvature_m')}, offset: {self.lane_info.get('lateral_offset_m')}")
        print(f"Anzahl Schilder: {len(self.schilder)}")
        for o in self.schilder.values(): 
            print(f"  - {o['label']} @ {o['position']} (score: {o['score']:.2f})")
        print(f"Anzahl Ampeln: {len(self.ampeln)}")
        for o in self.ampeln.values(): 
            print(f"  - {o['state']} @ {o['position']} (score: {o['score']:.2f})")
        print("==============================================")
