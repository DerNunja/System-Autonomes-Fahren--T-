from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2
from PIL import Image
from time import perf_counter

import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import onnxruntime as ort

# ================= Timer =================
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class TickTock:
    def __init__(self): self.t = {}
    def start(self, k): self.t[k] = perf_counter()
    def stop(self, k):  return perf_counter() - self.t[k]

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] CUDA available:", torch.cuda.is_available())

# ================= Road via SegFormer =================
MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
segformer = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(device).eval()

@torch.inference_mode()
def infer_road_mask(img_pil: Image.Image, tt: Optional[TickTock] = None) -> np.ndarray:
    W, H = img_pil.size
    if tt: tt.start("prep")
    inputs = processor(images=img_pil, return_tensors="pt").to(device)
    _sync(); prep = tt.stop("prep") if tt else None

    if tt: tt.start("forward")
    outputs = segformer(**inputs)
    _sync(); fwd = tt.stop("forward") if tt else None

    if tt: tt.start("upsample")
    logits = torch.nn.functional.interpolate(outputs.logits, size=(H, W),
                                             mode="bilinear", align_corners=False)[0]
    _sync(); up = tt.stop("upsample") if tt else None

    id2label = segformer.config.id2label
    road_id = {v.lower(): int(k) for k, v in id2label.items()}.get("road")
    pred = logits.argmax(0).detach().cpu().numpy().astype(np.uint8)
    road = (pred == road_id).astype(np.uint8)
    if tt: print(f"[TIMING] segformer: prep={prep*1000:.1f}ms | forward={fwd*1000:.1f}ms | upsample={up*1000:.1f}ms")
    return road

# ================= UFLD (ONNX) =================
# Standard-UFLD-Inputgröße (abhängig vom Checkpoint):
#  - TuSimple:  800x288, griding_num=100, num_rows≈56, num_lanes=4
#  - CULane:    800x288, griding_num=200, num_rows≈72, num_lanes=4
UFLD_W, UFLD_H = 800, 288  # (width, height)

def ufld_preprocess_bgr(img_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[float,float]]:
    h, w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (UFLD_W, UFLD_H), interpolation=cv2.INTER_LINEAR)
    # BGR->RGB, normalize nach UFLD-Konvention (ImageNet)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    chw = rgb.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)
    sx, sy = w / UFLD_W, h / UFLD_H
    return chw.astype(np.float32), (sx, sy)

def _ufld_build_col_samples(griding_num: int, width: int) -> np.ndarray:
    """gleichmäßig verteilte Spaltenpositionen über die Bildbreite (Netz-Koordinaten)."""
    return np.linspace(0, width - 1, griding_num)

def _polyline_to_mask(points: List[Tuple[int,int]], shape: Tuple[int,int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(mask, points[i], points[i+1], 1, 5)  # 5px Strichdicke
    return mask

def load_ufld_onnx(onnx_path: Path) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    return sess

def parse_ufld_outputs(outputs: List[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns (cls, exist):
      cls   : (B, num_lanes, num_rows, griding_num) ODER (B, num_lanes, griding_num, num_rows)  (wir normalisieren)
      exist : (B, num_lanes) oder None
    """
    # Heuristik: das größte 4D-Array ist die Klassifizierung (Gridding), 2D optional lane existence
    cls, exist = None, None
    for o in outputs:
        if o.ndim == 4:
            cls = o
        elif o.ndim == 2:
            exist = o
    if cls is None:
        # evtl. einziger Output ist 3D -> expand
        for o in outputs:
            if o.ndim == 3:
                cls = o[None, ...]  # (1, C, R, G) oder (1,C,G,R)
                break
    if cls is None:
        raise RuntimeError("UFLD: kein passender Klassifizierungs-Output im ONNX gefunden.")

    # Bringe in Form (B, L, R, G)
    B, C, A, Bdim = cls.shape  # zwei der dims sind (rows, griding)
    # Entscheide, welches rows ist: typischerweise R << G
    if A < Bdim:
        # (B, L, R, G) already if C==L
        cls_norm = cls  # (B,C,R,G)
    else:
        cls_norm = np.transpose(cls, (0, 1, 3, 2))  # (B, C, R, G)

    return cls_norm, exist

def ufld_infer_lanes(img_bgr: np.ndarray, sess: ort.InferenceSession,
                     griding_num: int = 100,
                     num_rows: Optional[int] = None,
                     num_lanes: Optional[int] = None) -> Tuple[np.ndarray, List[List[Tuple[int,int]]]]:
    """
    Führt UFLD aus und rekonstruiert pro Lane eine Liste von (x,y)-Punkten in Originalbild-Koordinaten.
    Returns (lane_mask_01, lane_points_per_lane)
    """
    blob, (sx, sy) = ufld_preprocess_bgr(img_bgr)
    # I/O-Namen
    in_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {in_name: blob})
    cls_raw, exist = parse_ufld_outputs(ort_out)  # (1, L, R, G)

    B, L, R, G = cls_raw.shape
    if num_rows is not None and R != num_rows:
        # Okay – wir vertrauen dem Modelloutput mehr als dem Parameter
        pass
    if num_lanes is not None and L != num_lanes:
        pass

    # Argmax über griding → Spaltenindex pro Row
    idx = cls_raw.argmax(axis=-1)[0]  # (L, R) Werte in [0..G-1]

    # col-samples im Netzraum (0..W-1 im ufld-Input)
    cols = _ufld_build_col_samples(griding_num=G, width=UFLD_W)

    # Row-Anker: gleichmäßig über die Höhe verteilen (Fallback),
    # viele Checkpoints haben feste row_anchors; ohne cfg nutzen wir uniform:
    rows = np.linspace(0, UFLD_H - 1, R).astype(np.float32)

    H, W = img_bgr.shape[:2]
    lane_mask = np.zeros((H, W), np.uint8)
    lanes_pts: List[List[Tuple[int,int]]] = []

    for l in range(L):
        # Existenzprüfung, falls vorhanden
        if exist is not None and exist.shape[-1] == L:
            if exist[0, l] < 0.5:  # Heuristik: <0.5 = nicht vorhanden
                lanes_pts.append([]); continue

        pts = []
        for r in range(R):
            c = idx[l, r]
            # UFLD nutzt manchmal c==G als 'keine Spur in dieser Zeile'
            if c <= 0 or c >= G:
                continue
            x_net = cols[c]
            y_net = rows[r]
            x = int(round(x_net * sx))
            y = int(round(y_net * sy))
            pts.append((x, y))

        # polyline zeichnen
        lanes_pts.append(pts)
        if len(pts) >= 2:
            cv2.polylines(lane_mask, [np.array(pts, np.int32)], isClosed=False, color=1, thickness=5)

    return lane_mask, lanes_pts

# ================= Pipeline =================
def process_image(img_path: Path, out_dir: Path, onnx_path: Path,
                  warmup: int = 1, use_intersection=True):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bild
    img = Image.open(img_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Modelle
    ufld_sess = load_ufld_onnx(onnx_path)

    # Warmup
    for _ in range(max(0, warmup)):
        _ = infer_road_mask(img)
        _ = ufld_infer_lanes(img_bgr, ufld_sess)
    _sync()

    tt = TickTock(); tt.start("total")

    # 1) Road
    road01 = infer_road_mask(img, tt)
    cv2.imwrite(str(out_dir / f"{img_path.stem}_road.png"), (road01*255).astype(np.uint8))

    # 2) UFLD lanes
    lane01, lanes_pts = ufld_infer_lanes(img_bgr, ufld_sess)
    # optional: nur innerhalb der Straße
    if use_intersection:
        lane01 &= road01

    cv2.imwrite(str(out_dir / f"{img_path.stem}_lane.png"), (lane01*255).astype(np.uint8))

    # 3) Overlay
    color = np.zeros_like(img_bgr)
    color[road01.astype(bool)] = (0,255,0)
    color[lane01.astype(bool)] = (0,255,255)
    overlay = (0.5*img_bgr + 0.5*color).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{img_path.stem}_overlay.png"), overlay)

    _sync()
    total = tt.stop("total")
    print(f"[TIMING] TOTAL={total*1000:.1f} ms  ({1.0/total:.2f} FPS)")

# ================= Beispiel =================
if __name__ == "__main__":
    inp = Path("Behavioural_Cloning/data/Processed/frames/frame_002568.jpg")
    out = Path("Behavioural_Cloning/data/Processed/seg_masks")
    onnx = Path("weights/ufld_tusimple.onnx")   # <- hier deinen UFLD-ONNX-Pfad setzen
    process_image(inp, out, onnx_path=onnx, warmup=2, use_intersection=True)