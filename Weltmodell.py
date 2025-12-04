# Updated Weltmodell.py
# Integrated with LaneDetector (from weltmodell_lane.py) and YOLOv5 (torch.hub or custom .pt)
# Simplified for highway use: REMOVE other vehicles — only traffic signs ("schilder") and traffic lights ("ampeln") plus lanes
import os
import math
import time
import uuid
from typing import Dict, Any, List, Optional
import numpy as np

# Optional imports for vision modules (import guarded)
try:
    import torch
except Exception:
    torch = None
try:
    import cv2
except Exception:
    cv2 = None

# Attempt to import LaneDetector (user must have weltmodell_lane.py in PYTHONPATH)
try:
    from weltmodell_lane import LaneDetector, TemporalSmoother, LaneResult
except Exception:
    LaneDetector = None
    TemporalSmoother = None
    LaneResult = None


class SimpleYOLOv5Wrapper:
    """
    Minimal wrapper for YOLOv5 using torch.hub or custom weights.
    Designed to detect only traffic signs and traffic lights (filtering done by class_map).
    This wrapper will NOT download models automatically if no internet — provide a local .pt path.
    """
    def __init__(self, model_path: str = "yolov5s", device: Optional[str] = None, class_map: Optional[Dict[int,str]] = None):
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        self.model = None
        self.backend = None
        self.class_map = class_map or {}  # e.g. {0:"person", 1:"car", 2:"traffic_light", 3:"sign"}
        if torch is None:
            print("Torch not available — Detection wrapper disabled.")
            return
        try:
            # If model_path points to a local .pt, try to load via hub 'custom' entrypoint
            if os.path.exists(model_path):
                # this uses ultralytics/yolov5 repo hubconf which supports custom weights
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
                self.backend = "yolov5_hub_custom"
            else:
                # model_path is a model name like 'yolov5s' available via hub
                self.model = torch.hub.load('ultralytics/yolov5', model_path, pretrained=True)
                self.backend = "yolov5_hub_pretrained"
            # set device
            try:
                self.model.to(self.device)
            except Exception:
                pass
        except Exception as e:
            print("Failed to load YOLOv5 model via torch.hub:", e)
            self.model = None

    def predict(self, img_rgb: np.ndarray, conf_thresh: float = 0.25) -> List[Dict[str,Any]]:
        """
        img_rgb: HxWx3 numpy RGB uint8
        Returns list of detections: {'box':[x1,y1,x2,y2], 'score':float, 'class_id':int, 'class_name':str}
        """
        if self.model is None:
            return []
        # torch.hub yolov5 expects images in either numpy or list
        try:
            results = self.model(img_rgb)
            # results.xyxy[0] -> tensor of detections
            xyxy = results.xyxy[0].cpu().numpy() if hasattr(results, 'xyxy') else results.xyxy[0].numpy()
            dets = []
            for row in xyxy:
                x1, y1, x2, y2, conf, cid = row.tolist()
                if conf < conf_thresh:
                    continue
                cid = int(cid)
                cname = self.class_map.get(cid, str(cid))
                dets.append({'box':[float(x1), float(y1), float(x2), float(y2)], 'score':float(conf), 'class_id':cid, 'class_name':cname})
            return dets
        except Exception as e:
            print("YOLO predict error:", e)
            return []


class Weltmodell:
    """
    Lightweight world model specialized for highway-only scenarios without other vehicles.
    Tracks:
      - ego state (position, speed, heading)
      - static objects (infrastructure)
      - traffic signs (schilder)
      - traffic lights (ampeln)
      - lane information (from LaneDetector)
    """
    def __init__(self, lane_detector: Optional[LaneDetector] = None, detection_wrapper: Optional[SimpleYOLOv5Wrapper] = None):
        # Ego vehicle state
        self.ego = {
            "position": (0.0, 0.0),
            "geschwindigkeit": 0.0,
            "richtung": 0.0,  # heading in degrees
            "lenkwinkel": 0.0
        }

        # Static infrastructure: id -> object dict
        self.statische_objekte: Dict[str, Dict[str,Any]] = {}

        # Traffic signs and lights (separate collections for convenience)
        self.schilder: Dict[str, Dict[str,Any]] = {}
        self.ampeln: Dict[str, Dict[str,Any]] = {}

        # Lane information (single lane pair assumed for highway)
        self.lane_info: Dict[str, Any] = {
            'curvature_m': None,
            'lateral_offset_m': None,
            'confidence': 0.0,
            'left_fit': None,
            'right_fit': None,
            'last_update': None
        }

        # Vision modules (injected)
        self.lane_detector = lane_detector
        self.detector = detection_wrapper

    # -----------------------
    # --- Object helpers ---
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
            'state': state,  # e.g., 'red', 'green', 'yellow', or 'unknown'
            'position': position,
            'score': float(score),
            'timestamp': time.time()
        }
        return obj_id

    def update_lane_info(self, lane_result: LaneResult):
        if lane_result is None:
            return
        self.lane_info.update({
            'curvature_m': float(lane_result.curvature_m) if lane_result.curvature_m is not None else None,
            'lateral_offset_m': float(lane_result.lateral_offset_m) if lane_result.lateral_offset_m is not None else None,
            'confidence': float(lane_result.confidence),
            'left_fit': lane_result.left_fit.tolist() if lane_result.left_fit is not None else None,
            'right_fit': lane_result.right_fit.tolist() if lane_result.right_fit is not None else None,
            'last_update': time.time()
        })

    # -----------------------
    # --- Vision update API ---
    # -----------------------
    def update_from_vision(self, frame_rgb: np.ndarray):
        """
        Run lane detection and object detection on a single RGB frame and update Weltmodell state.
        The detector is expected to be YOLOv5 or similar with class_map where traffic lights & signs have known IDs.
        We filter detections to only update signs & lights.
        """
        # 1) lane detection
        if self.lane_detector is not None:
            try:
                lane_res = self.lane_detector.process(frame_rgb)
                self.update_lane_info(lane_res)
            except Exception as e:
                print("Lane detection failed:", e)

        # 2) object detection (YOLO)
        if self.detector is not None:
            try:
                dets = self.detector.predict(frame_rgb, conf_thresh=0.25)
                for d in dets:
                    cname = d.get('class_name', str(d.get('class_id', -1))).lower()
                    box = d.get('box', [0,0,0,0])
                    # Estimate position simplistically as box center in image coords
                    x1,y1,x2,y2 = box
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    position = (float(cx), float(cy))
                    score = float(d.get('score', 0.0))

                    # classify as light or sign
                    if "traffic" in cname or "light" in cname or "ampel" in cname:
                        state = 'unknown'
                        if 'red' in cname:
                            state = 'red'
                        elif 'green' in cname:
                            state = 'green'
                        elif 'yellow' in cname or 'amber' in cname:
                            state = 'yellow'
                        self.add_ampel(position=position, state=state, score=score)

                    elif "sign" in cname or "schild" in cname or "speed" in cname:
                        label = cname
                        self.add_schild(position=position, label=label, score=score)

                    # ignore everything else (cars, pedestrians, etc.)
            except Exception as e:
                print("Detection wrapper error:", e)

    # -----------------------
    # --- Public utilities ---
    # -----------------------
    def get_summary(self) -> Dict[str,Any]:
        summary = {
            "ego": self.ego.copy(),
            "lane_info": self.lane_info.copy(),
            "anzahl_schilder": len(self.schilder),
            "anzahl_ampeln": len(self.ampeln),
            "schilder": list(self.schilder.values()),
            "ampeln": list(self.ampeln.values())
        }
        return summary

    def print_state(self):
        print("\n=== 🌍 Weltmodell-Status (Highway mode) ===")
        print(f"Ego-Fahrzeug: {self.ego}")
        print(f"Lane confidence: {self.lane_info.get('confidence')}, curvature: {self.lane_info.get('curvature_m')}, offset: {self.lane_info.get('lateral_offset_m')}")
        print(f"Anzahl Schilder: {len(self.schilder)}")
        for o in self.schilder.values():
            print(f"  - {o['label']} @ {o['position']} (score: {o['score']:.2f})")
        print(f"Anzahl Ampeln: {len(self.ampeln)}")
        for o in self.ampeln.values():
            print(f"  - {o['state']} @ {o['position']} (score: {o['score']:.2f})")
        print("==============================================")

# --------------------------
# --- Example usage -----
# --------------------------
if __name__ == '__main__':
    # Example showing how to instantiate and run one frame through the Weltmodell updated pipeline.
    # NOTE: This demo will NOT download heavy models automatically. Provide local yolov5 weights if available.
    # Adjust image_path and model_path to your environment.
    image_path = 'test.jpg'
    model_path = 'yolov5s'  # or path to custom weights like '/path/to/best.pt'

    # instantiate lane detector if module is available
    lane = None
    if LaneDetector is not None:
        lane = LaneDetector(image_shape=(480,640), smoother=TemporalSmoother(maxlen=5))

    detector = None
    if torch is not None:
        detector = SimpleYOLOv5Wrapper(model_path=model_path, device=None, class_map={
            0:'person', 1:'bicycle', 2:'car', 3:'motorbike', 4:'traffic_light', 5:'sign'
        })

    wm = Weltmodell(lane_detector=lane, detection_wrapper=detector)

    if os.path.exists(image_path):
        import cv2 as _cv2
        img = _cv2.cvtColor(_cv2.imread(image_path), _cv2.COLOR_BGR2RGB)
        wm.update_from_vision(img)
        wm.print_state()
    else:
        print('Place an RGB test image at', image_path, 'and run this script.')
