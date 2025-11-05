import argparse
import os
import sys
import glob
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from typing import List, Tuple, Dict

import importlib.util
from utils.common import get_model

def build_transform(cfg):
    return transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def load_weights_to_model(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    compatible = {}
    for k, v in state.items():
        compatible[k[7:]] = v if k.startswith('module.') else v
    model.load_state_dict(compatible, strict=False)
    model.to(device)
    model.eval()
    return model


def load_cfg_from_file(cfg_path):
    spec = importlib.util.spec_from_file_location("ufld_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sort_points_for_lane(kind, pts):
    # row-Lanes verlaufen vertikal -> entlang y sortieren
    # col-Lanes verlaufen horizontal -> entlang x sortieren
    if kind == 'row':
        return sorted(pts, key=lambda p: (p[1], p[0]))
    else:
        return sorted(pts, key=lambda p: (p[0], p[1]))


def map_row_anchor_y(val, *, resize_h, crop_y0, train_h, anchor_base_h, mode):
    """
    val: row_anchor[k] (0..1)
    return: y_resized (Koordinate im resized Bild vor Rückskalierung)
    """
    if mode == 'base':
        return crop_y0 + float(val) * float(anchor_base_h)
    elif mode == 'train':
        return crop_y0 + float(val) * float(train_h)
    elif mode == 'flip':
        return crop_y0 + (1.0 - float(val)) * float(train_h)
    else:
        # fallback: 'base'
        return crop_y0 + float(val) * float(anchor_base_h)


def pred2coords_with_conf(
    pred: Dict[str, torch.Tensor],
    row_anchor: List[float],
    col_anchor: List[float] | None = None,
    local_width: int = 1,
    original_image_width: int = 1640,
    original_image_height: int = 590,
    resize_h: int = 0,
    crop_y0: int = 0,
    train_height: int = 0,
    anchor_base_h: int = 288,
    anchor_mode: str = 'train',
    debug: bool = True,
    exist_thresh: float = 0.45,      # höherer Schwellwert
    bottom_frac: float = 0.8,       # nur untere 50% der row_anchors
):
    """
    Robuste, minimalistische Extraktion NUR aus dem row-Branch:
    - keine Fallbacks (kein 'nimm alle')
    - nur Anker im unteren Bildbereich (bottom_frac)
    - x aus Argmax (ohne lokalen Softmax-Mittel)
    """
    assert 'loc_row' in pred and 'exist_row' in pred, "Vorhersage enthält keinen row-Branch."
    B, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    if len(row_anchor) != num_cls_row:
        raise RuntimeError(f"row_anchor len={len(row_anchor)} passt nicht zu num_cls_row={num_cls_row}")

    # Argmax über Grid und Existenz-Wahrscheinlichkeit (nur Lane-Existenz=1)
    max_idx_row = pred['loc_row'].argmax(1).cpu()             # (B, num_cls_row, num_lane_row)
    exist_row_prob = pred['exist_row'].softmax(1).cpu()[:, 1, :, :]  # (B, num_cls_row, num_lane_row)

    # nur die beiden mittleren Fahrspuren versuchen
    lane_indices = list(range(num_lane_row))

    # Indizes der unteren Anchors (row_anchor ist 0..1, 1 = oben ODER unten → daher anchor_mode beachten)
    # Wir filtern anhand der "train"-Interpretation: val > 1-bottom_frac ⇒ unten
    # (bei 'flip' invertieren wir entsprechend)
    bottom_mask = []
    for a in row_anchor:
        a_train = (1.0 - a) if anchor_mode == 'flip' else a
        bottom_mask.append(a_train >= (1.0 - bottom_frac))
    bottom_mask = torch.tensor(bottom_mask, dtype=torch.bool)

    lanes = []
    total_points = 0

    for lane_i in lane_indices:
        pts, probs = [], []

        # nur valide + unten
        valid = (exist_row_prob[0, :, lane_i] >= exist_thresh) & bottom_mask
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()

        if debug:
            vm_sum = int(valid.sum())
            p_mean = float(exist_row_prob[0, :, lane_i][bottom_mask].mean())
            print(f"[DEBUG] lane {lane_i}: valid_bottom={vm_sum}/{int(bottom_mask.sum())}, mean_p_bottom={p_mean:.3f}")

        if vm_sum == 0:
            continue  # keine Punkte -> Lane skippen

        for k in valid_idx.tolist():
            # reiner Argmax (stabiler als lokaler Softmax-Mittel)
            g = int(max_idx_row[0, k, lane_i].item())
            out_tmp = float(g) + 0.5

            # X zurück in Originalbreite
            x = int((out_tmp / max(1, (num_grid_row - 1))) * original_image_width)

            # Y-Mapping
            a = float(row_anchor[k])
            if anchor_mode == 'flip':
                y_resized = crop_y0 + (1.0 - a) * float(train_height)
            elif anchor_mode == 'base':
                y_resized = crop_y0 + a * float(anchor_base_h)
            else:  # 'train'
                y_resized = crop_y0 + a * float(train_height)
            y = int((y_resized / max(1.0, float(resize_h))) * float(original_image_height))

            pts.append((x, y))
            probs.append(float(exist_row_prob[0, k, lane_i]))

        if pts:
            lanes.append({'points': pts, 'probs': probs, 'kind': 'row', 'num_classes': num_cls_row})
            total_points += len(pts)

            if debug:
                xs = [p[0] for p in pts]
                print(f"[DEBUG] lane {lane_i}: x-range={min(xs)}..{max(xs)}, mean={np.mean(xs):.1f}, n={len(xs)}")

    if debug:
        print(f"[DEBUG] resize_h={resize_h}, crop_y0={crop_y0}, train_h={train_height}, total_row_pts={total_points}")

    return lanes



def classify_line_style(points: List[Tuple[int, int]], num_classes: int, coverage_thresh=0.7, max_gap_fraction=0.08):
    """
    Heuristik:
      - coverage = #Punkte / #Anker
      - Größtes Lückenmaß relativ zur Ankerzahl
      -> 'dashed' wenn coverage klein ODER große Lücken (typisch gestrichelt)
    """
    if not points:
        return 'dashed'

    coverage = len(points) / float(num_classes)

    # Lücken-Messung: sortiere entlang y (abwärts) und nimm durchschnittlichen Abstand
    pts_sorted = sorted(points, key=lambda p: (p[1], p[0]))
    gaps = []
    for a, b in zip(pts_sorted[:-1], pts_sorted[1:]):
        dy = abs(b[1] - a[1])
        dx = abs(b[0] - a[0])
        gaps.append((dx + dy) * 0.5)  # einfacher Pixelabstand
    max_gap = max(gaps) if gaps else 0.0

    # Normiere grob mit Bildhöhe/Ankeranzahl (Pixel pro Anker ~ span)
    # Hier einfache Proxy: wenn sehr wenige Punkte -> gestrichelt; sonst prüfe große Lücken
    if coverage < coverage_thresh:
        return 'dashed'
    # Wenn die größte Lücke deutlich größer als "üblicher" Punktabstand ist, ebenfalls gestrichelt
    # Die Schwelle max_gap_fraction bezieht sich auf die Bildhöhe grob; ohne Bildhöhe hier heuristisch:
    # Wenn <10 Punkte, wenig Info -> eher solid, außer große Lücke > 60 Pixel
    if max_gap > 60 and coverage < 0.9:
        return 'dashed'
    return 'solid'


def draw_polyline_solid(img, pts, thickness=4, color=(0, 255, 0)):
    if len(pts) >= 2:
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [arr], False, color, thickness, cv2.LINE_AA)
    elif len(pts) == 1:
        cv2.circle(img, pts[0], thickness, color, -1)


def draw_polyline_dashed(img, pts, dash_len=18, gap_len=12, thickness=4, color=(0, 255, 255)):
    # Zeichne kleine Segmente entlang der Polyline
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        draw_dashed_segment(img, p1, p2, dash_len, gap_len, thickness, color)


def draw_dashed_segment(img, p1, p2, dash_len, gap_len, thickness, color):
    import math
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    vx, vy = dx / dist, dy / dist
    t = 0.0
    while t < dist:
        a, b = t, min(t + dash_len, dist)
        sx, sy = int(x1 + vx * a), int(y1 + vy * a)
        ex, ey = int(x1 + vx * b), int(y1 + vy * b)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        t += dash_len + gap_len



def draw_polyline_dashed(img, pts, dash_len=18, gap_len=12, thickness=4, color=(0, 255, 255)):
    """Zeichnet eine gestrichelte Linie über Punkte."""
    for i in range(len(pts) - 1):
        draw_dashed_segment(img, pts[i], pts[i + 1], dash_len, gap_len, thickness, color)


def np_int32(pts: List[Tuple[int, int]]):
    import numpy as np
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def overlay_confidence(img, pts, probs, color=(255, 255, 255)):
    if not pts or not probs:
        return
    mid = len(pts) // 2
    p = max(0.0, min(1.0, sum(probs) / len(probs)))
    text = f"{p:.2f}"
    x, y = pts[mid]
    y = max(20, y - 10)
    cv2.putText(img, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_points(img, pts, kind):
    color = (0, 255, 0) if kind == 'row' else (255, 0, 0)
    for (x, y) in pts:
        cv2.circle(img, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)


def infer_image_file(cfg, net, device, image_path, out_path, show_conf=False, hide_col=False, points_only=False, anchor_mode='base'):
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Konnte Bild nicht lesen: {image_path}")
    H, W = img_bgr.shape[:2]

    # --- PREPROCESS exakt wie im Training ---
    resize_h = int(cfg.train_height / cfg.crop_ratio)
    # resize nach (W=cfg.train_width, H=resize_h)
    img_resized = cv2.resize(img_bgr, (cfg.train_width, resize_h), interpolation=cv2.INTER_LINEAR)

    # Bottom-Crop: die unteren train_height Pixel
    crop_y0 = max(0, resize_h - cfg.train_height)
    crop_y1 = resize_h
    img_cropped = img_resized[crop_y0:crop_y1, :, :]  # H = train_height, W = train_width

    # -> Tensor
    img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    pil_input = transforms.functional.to_pil_image(img_rgb)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tensor = tfm(pil_input).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = net(tensor)

    lanes = pred2coords_with_conf(
        pred,
        cfg.row_anchor,
        cfg.col_anchor,
        original_image_width=W,
        original_image_height=H,
        resize_h=resize_h,
        crop_y0=crop_y0,
        train_height=cfg.train_height,
        anchor_base_h=getattr(cfg, 'anchor_base_height', getattr(cfg, 'anchor_base_h', 288)),
        anchor_mode=anchor_mode,   
    )


    vis = img_bgr.copy()
    for lane in lanes:
        if hide_col and lane['kind'] == 'col':
            continue

        pts = lane['points']
        probs = lane['probs']
        kind = lane['kind']  # 'row' oder 'col'

        # Sortieren hilft nur fürs Linienzeichnen; für Punkte egal
        # pts = sort_points_for_lane(kind, pts)

        if len(pts) == 0:
            continue

        if points_only:
            draw_points(vis, pts, kind)
            if show_conf:
                overlay_confidence(vis, pts, probs)
            continue

        # --- Linienmodus (falls points_only False) ---
        pts = sort_points_for_lane(kind, pts)
        if len(pts) < 2:
            draw_points(vis, pts, kind)  # fallback: einzelner Punkt
            if show_conf:
                overlay_confidence(vis, pts, probs)
            continue

        style = classify_line_style(pts, lane['num_classes'])
        color = (0, 255, 0) if kind == 'row' else (255, 0, 0)

        if style == 'solid':
            draw_polyline_solid(vis, pts, thickness=4, color=color)
        else:
            draw_polyline_dashed(vis, pts, dash_len=18, gap_len=12, thickness=4, color=color)

        if show_conf:
            overlay_confidence(vis, pts, probs)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = cv2.imwrite(out_path, vis)
    if not ok:
        raise RuntimeError(f"Konnte Ausgabedatei nicht schreiben: {out_path}")


def collect_images(input_path: str) -> Tuple[str, list]:
    """
    Gibt (root_dir, [bildpfade]) zurück.
    - Datei: root_dir = Ordner der Datei
    - Ordner: root_dir = selbst
    """
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.webp')
    paths = []
    if os.path.isfile(input_path):
        root = os.path.dirname(os.path.abspath(input_path))
        paths = [input_path]
    else:
        root = os.path.abspath(input_path)
        for e in exts:
            paths.extend(glob.glob(os.path.join(root, e)))
        # auch rekursiv?
        if not paths:
            for e in exts:
                paths.extend(glob.glob(os.path.join(root, '**', e), recursive=True))
    paths = sorted(paths)
    return root, paths


def parse_args():
    p = argparse.ArgumentParser(description="UFLDv2 – Batch-Bildinferenz mit Stil-Klassifikation & optionaler Confidence")
    p.add_argument('config', type=str, help="Pfad zur *.py Config (z.B. configs/culane_res18.py)")
    p.add_argument('--test_model', required=True, type=str, help="Pfad zu den Gewichten (.pth)")
    p.add_argument('--input', required=True, type=str, help="Bilddatei oder Ordner mit Bildern")
    p.add_argument('--output_dir', type=str, default=None, help="Zielordner (Default: <input_root>/result_ulfd)")
    p.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None, help="Gerät erzwingen")
    p.add_argument('--show_conf', action='store_true', help="Confidence-Scores einblenden")
    p.add_argument('--hide_col', action='store_true', help='col-Lanes (blau) nicht zeichnen')
    p.add_argument('--points_only', action='store_true', help='Keine Linien zeichnen, nur Marker wie im Original-Repo')
    p.add_argument('--anchor_mode', choices=['base', 'train', 'flip'], default='base', help="Mapping der row_anchor: 'base'=anchor_base_h, 'train'=train_height, 'flip'=(1-anchor)*train_height")

    return p.parse_args()


def main():
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    assert os.path.isfile(args.config), f"Config nicht gefunden: {args.config}"
    assert os.path.isfile(args.test_model), f"Gewichte nicht gefunden: {args.test_model}"
    in_root, image_paths = collect_images(args.input)
    if not image_paths:
        print("Keine Bilder gefunden.", file=sys.stderr)
        sys.exit(1)

    cfg = load_cfg_from_file(args.config)
    if len(cfg.row_anchor) == cfg.num_row + 1:
        cfg.row_anchor = cfg.row_anchor[:-1]   # letzten Eintrag droppen
    if len(cfg.row_anchor) != cfg.num_row:
        raise RuntimeError(f"row_anchor len={len(cfg.row_anchor)} != num_row={cfg.num_row}")
    cfg.batch_size = 1
    if not hasattr(cfg, 'row_anchor') or not hasattr(cfg, 'col_anchor'):
        raise RuntimeError("Config muss row_anchor und col_anchor enthalten.")

    device = torch.device(
        args.device if args.device in ('cpu', 'cuda') else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    net = get_model(cfg)
    net = load_weights_to_model(net, args.test_model, device)

    out_dir = args.output_dir or os.path.join(in_root, 'result_ulfd')
    os.makedirs(out_dir, exist_ok=True)

    for ip in image_paths:
        stem, ext = os.path.splitext(os.path.basename(ip))
        out_path = os.path.join(out_dir, f"{stem}_Lane{ext}")
        try:
            infer_image_file(cfg, net, device, ip, out_path,
                             show_conf=args.show_conf,
                             hide_col=args.hide_col,
                             points_only=args.points_only,
                             anchor_mode=args.anchor_mode)
            
            print(f"[OK] {ip} -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {ip}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()