from PIL import Image
import torch
import cv2
import torchvision.transforms as transforms
from pathlib import Path
from .model.model_culane import parsingNet as LaneNet
from .configs.culane_res34 import (
    num_row, num_col, train_width, train_height,
    num_cell_row, num_cell_col, num_lanes, backbone, fc_norm,
)
import numpy as np

CANON_H = 590
VIS_CROP_TOP = -0.7
VIS_CROP_RANGE = 1.0 - VIS_CROP_TOP


def init_lanedetector():
    """Initialisiert Netz, Config, Transforms und Device einmalig."""
    row_anchor = np.linspace(0.42, 1.0, num_row)
    col_anchor = np.linspace(0.0, 1.0, num_col)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = LaneNet(
        pretrained=True,
        backbone=backbone,
        num_grid_row=num_cell_row,
        num_cls_row=num_row,
        num_grid_col=num_cell_col,
        num_cls_col=num_col,
        num_lane_on_row=num_lanes,
        num_lane_on_col=num_lanes,
        use_aux=False,
        input_height=train_height,
        input_width=train_width,
        fc_norm=fc_norm,
    ).to(device)
    net.eval()

    weight_path = Path(__file__).resolve().parent / "weights" / "culane_res34.pth"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at '{weight_path}'. "
            f"Make sure culane_res34.pth is in LaneDetection/weights/"
        )
    try:
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from '{weight_path}': {e}") from e

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state = checkpoint['model']
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise RuntimeError(
            f"Unexpected checkpoint format at '{weight_path}'. "
            f"Expected a dict with a 'model' key, got {type(checkpoint).__name__}"
        )
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)

    img_transforms = transforms.Compose([
        transforms.Resize((train_height, train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # cfg-like object for downstream code that expects cfg.train_width etc.
    cfg = type('Cfg', (), {
        'train_width': train_width,
        'train_height': train_height,
        'num_row': num_row,
        'num_col': num_col,
        'row_anchor': row_anchor,
        'col_anchor': col_anchor,
    })()

    return net, cfg, img_transforms, device


def pred2coords_mixed(pred, row_anchor, model_w, topk_lanes=2):
    """
    Wandelt Model-Output in Lane-Punkte um (x in Modell-Breite, y in CULane-Höhe 590).
    Wählt top-k Lanes nach Score, filtert unsichere Punkte.
    """
    _, G_r, C_r, L = pred['loc_row'].shape
    loc_row = pred['loc_row'][0].cpu()
    max_idx_row = pred['loc_row'].argmax(1)[0].cpu()
    exist_row_p = pred['exist_row'].softmax(1)[0, 1].cpu()

    lane_scores = exist_row_p.mean(0)
    picked = torch.argsort(lane_scores, descending=True)[:min(topk_lanes, L)].tolist()

    lanes_xy = []
    lanes_info = []

    for lane in picked:
        xs, ys = [], []
        for k in range(C_r):
            if float(exist_row_p[k, lane]) < 0.8:
                continue

            center = int(max_idx_row[k, lane])
            left = max(0, center - 2)
            right = min(G_r - 1, center + 2)
            inds = torch.arange(left, right + 1)

            probs = loc_row[inds, k, lane].softmax(0)
            x_hat = (probs * inds.float()).sum() + 0.5

            x_conf = float(probs.max())
            point_conf = float(exist_row_p[k, lane]) * x_conf

            if point_conf < 0.25:
                continue

            x = float(x_hat) / (G_r - 1) * model_w
            y = (VIS_CROP_TOP + float(row_anchor[k]) * VIS_CROP_RANGE) * CANON_H

            xs.append(x)
            ys.append(y)

        # Median-Smoothing
        if len(xs) >= 5:
            xs_np = np.array(xs, dtype=np.float32)
            radius = 2
            for i in range(len(xs_np)):
                l = max(0, i - radius)
                r = min(len(xs_np), i + radius + 1)
                xs_np[i] = np.median(xs_np[l:r])
            xs = xs_np.tolist()

        lane_pts = list(zip(xs, ys))
        lanes_xy.append(lane_pts)
        lanes_info.append({
            "lane_id": lane,
            "score": float(lane_scores[lane]),
            "n_points": len(lane_pts),
        })

    return lanes_xy, lanes_info


def process_frame(frame_bgr, net, cfg, img_transforms, device):
    """
    Verarbeitet einen BGR-Frame (numpy) und gibt (annotiertes_BGR, lanes_xy, lanes_info) zurück.
    lanes_xy: Liste von [(x_model, y_canon), ...] pro Lane
    lanes_info: Liste von Dicts mit lane_id, score, n_points
    """
    H, W = frame_bgr.shape[:2]
    MODEL_W = cfg.train_width

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    image_tensor = img_transforms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = net(image_tensor)

    lanes_xy, lanes_info = pred2coords_mixed(
        pred, cfg.row_anchor, model_w=MODEL_W, topk_lanes=2
    )

    # Scaling factors
    sx_model_w = W / MODEL_W
    sy_canon_h = H / CANON_H

    vis = frame_bgr.copy()
    vis = _draw_lanes(vis, lanes_xy, sx_model_w, sy_canon_h)

    return vis, lanes_xy, lanes_info


def _draw_lanes(vis_bgr, lanes_xy, sx_model_w, sy_canon_h):
    """Zeichnet erkannte Lanes als grüne Linien auf das Bild."""
    H, W = vis_bgr.shape[:2]

    for lane in lanes_xy:
        pts = []
        for (x_model, y_canon) in lane:
            xi = int(round(x_model * sx_model_w))
            yi = int(round(y_canon * sy_canon_h))
            if 0 <= xi < W and 0 <= yi < H:
                pts.append([xi, yi])

        if len(pts) >= 2:
            cv2.polylines(vis_bgr, [np.array(pts, dtype=np.int32)],
                          isClosed=False, color=(0, 255, 0), thickness=3,
                          lineType=cv2.LINE_AA)
        elif len(pts) == 1:
            cv2.circle(vis_bgr, tuple(pts[0]), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    return vis_bgr
