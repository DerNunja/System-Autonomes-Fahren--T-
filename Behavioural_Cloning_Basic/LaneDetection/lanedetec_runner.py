from PIL import Image
import torch
import cv2
import torchvision.transforms as transforms
from .model.model_culane import parsingNet as LaneNet
from .configs import culane_res34 as lane_cfg
import numpy as np

DEBUG = True  # globale Debug-Flag

def init_lanedetector():
    """Initialisiert Netz, Config, Transforms und Device nur einmal."""
    torch.backends.cudnn.benchmark = True
    cfg = lane_cfg

    CANON_W, CANON_H = 1640, 590

    cfg.dataset = 'CULane'
    cfg.batch_size = 1
    print('DATASET =', cfg.dataset)

    if not hasattr(cfg, "row_anchor") or not hasattr(cfg, "col_anchor"):
        if cfg.dataset == 'CULane':
            cfg.row_anchor = np.linspace(0.42, 1.0, cfg.num_row)
            cfg.col_anchor = np.linspace(0.0, 1.0, cfg.num_col)
        elif cfg.dataset == 'Tusimple':
            cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / 720.0
            cfg.col_anchor = np.linspace(0.0, 1.0, cfg.num_col)
        elif cfg.dataset == 'CurveLanes':
            cfg.row_anchor = np.linspace(0.4, 1.0, cfg.num_row)
            cfg.col_anchor = np.linspace(0.0, 1.0, cfg.num_col)

    if not getattr(cfg, "test_model", ""):
        cfg.test_model = "LaneDetection/weights/culane_res34.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = LaneNet(
        pretrained=True,
        backbone=cfg.backbone,
        num_grid_row=cfg.num_cell_row,
        num_cls_row=cfg.num_row,
        num_grid_col=cfg.num_cell_col,
        num_cls_col=cfg.num_col,
        num_lane_on_row=cfg.num_lanes,
        num_lane_on_col=cfg.num_lanes,
        use_aux=cfg.use_aux,
        input_height=cfg.train_height,
        input_width=cfg.train_width,
        fc_norm=cfg.fc_norm,
    ).to(device)
    net.eval()

    state = torch.load(cfg.test_model, map_location=device)['model']
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)

    img_transforms = transforms.Compose([
        transforms.Resize((cfg.train_height, cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    return net, cfg, img_transforms, device

def pred2coords_mixed(pred, row_anchor, model_w, cfg,
                      thr_row=0.8, local_width=2, topk_lanes=2, smooth_kernel=5):
    """
    Erzeugt Lane-Punkte im 'gemischten' Raum:
      x:   in MODEL_W-Einheiten (cfg.train_width)
      y:   in CULane-Höhe (590px), inkl. crop_offset (crop_ratio)
    """
    _, G_r, C_r, L = pred['loc_row'].shape
    loc_row      = pred['loc_row'][0].cpu()                 
    max_idx_row  = pred['loc_row'].argmax(1)[0].cpu()       
    exist_row_p  = pred['exist_row'].softmax(1)[0,1].cpu()  

    lane_scores = exist_row_p.mean(0)                       # [L]
    picked = torch.argsort(lane_scores, descending=True)[:min(topk_lanes, L)].tolist()

    VIS_CROP_TOP = -0.7   # lane detection zurück korrigieren nur wichtig für vis
    VIS_CROP_BOTTOM = 1.0
    VIS_CROP_RANGE = VIS_CROP_BOTTOM - VIS_CROP_TOP
    CANON_H = 590.0


    lanes_xy = []
    lanes_info = []  # für Debug

    for lane in picked:
        xs, ys = [], []
        for k in range(C_r):

            if float(exist_row_p[k, lane]) < thr_row:
                continue

            center = int(max_idx_row[k, lane])
            left  = max(0, center - local_width)
            right = min(G_r - 1, center + local_width)
            inds = torch.arange(left, right + 1)

            probs = loc_row[inds, k, lane].softmax(0)
            x_hat = (probs * inds.float()).sum() + 0.5

            x_conf = float(probs.max())                      # schmeißt unsichere punkte raus
            point_conf = float(exist_row_p[k, lane]) * x_conf

            if point_conf < 0.25:   
                continue

            x = float(x_hat) / (G_r - 1) * model_w

            y = (VIS_CROP_TOP + float(row_anchor[k]) * VIS_CROP_RANGE) * CANON_H   

            xs.append(x)
            ys.append(y)

        if len(xs) >= smooth_kernel:
            xs_np = np.array(xs, dtype=np.float32)
            radius = smooth_kernel // 2
            for i in range(len(xs_np)):
                l = max(0, i - radius)
                r_ = min(len(xs_np), i + radius + 1)
                xs_np[i] = np.median(xs_np[l:r_])
            xs = xs_np.tolist()

        lane_pts = list(zip(xs, ys))
        lanes_xy.append(lane_pts)
        lanes_info.append({
            "lane_id": lane,
            "score": float(lane_scores[lane]),
            "n_points": len(lane_pts),
        })

    if DEBUG:
        print("  [DEBUG] Lane-Auswahl:")
        for info in lanes_info:
            print(f"    lane_idx={info['lane_id']}  score={info['score']:.3f}  points={info['n_points']}")

    return lanes_xy, lanes_info

def draw_lanes_mixed(vis_bgr, lanes_xy, sx_model_w, sy_canon_h,
                     color=(0,255,0), thickness=3, debug=False, lanes_info=None):
    """
    Zeichnet Lanes auf das Bild:
      - Linien (polylines)
      - optional Kreise auf jedem Punkt
      - optional Text-Overlay mit Lane-Infos
    """
    H, W = vis_bgr.shape[:2]
    pts_count = []

    for idx, lane in enumerate(lanes_xy):
        pts = []
        for (x_model, y_canon) in lane:
            xi = int(round(x_model * sx_model_w))   # skaliere x von MODEL_W -> W
            yi = int(round(y_canon * sy_canon_h))   # skaliere y von 590 -> H
            if 0 <= xi < W and 0 <= yi < H:
                pts.append([xi, yi])

        pts_count.append(len(pts))

        if len(pts) >= 2:
            cv2.polylines(vis_bgr, [np.array(pts, dtype=np.int32)],
                          isClosed=False, color=color, thickness=thickness,
                          lineType=cv2.LINE_AA)
        elif len(pts) == 1:
            cv2.circle(vis_bgr, tuple(pts[0]), 3, color, -1, lineType=cv2.LINE_AA)

        if debug:
            for (x_model, y_canon) in lane:
                xi = int(round(x_model * sx_model_w))
                yi = int(round(y_canon * sy_canon_h))
                if 0 <= xi < W and 0 <= yi < H:
                    cv2.circle(vis_bgr, (xi, yi), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    if debug:
        txt = f"lanes={len(lanes_xy)} | pts={pts_count}"
        if lanes_info:
            scores = [f"{info['score']:.2f}" for info in lanes_info]
            txt += f" | scores={scores}"
        cv2.putText(vis_bgr, txt, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    return vis_bgr

def process_frame(frame_bgr, net, cfg, img_transforms, device):
    """
    Nimmt ein BGR-Frame (numpy, z.B. aus OpenCV/NDI) und gibt ein
    annotiertes BGR-Frame zurück.
    """
    # Originalgröße
    H, W = frame_bgr.shape[:2]

    MODEL_W, MODEL_H = cfg.train_width, cfg.train_height
    CANON_W, CANON_H = 1640, 590

    # BGR (OpenCV) -> RGB (PIL) für die Transforms
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    image_tensor = img_transforms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = net(image_tensor)

    lanes_xy, lanes_info = pred2coords_mixed(
        pred,
        cfg.row_anchor,
        model_w=MODEL_W,
        cfg=cfg,
        thr_row=0.8,
        local_width=2,
        topk_lanes=2,
        smooth_kernel=5
    )

    # Scaling Originalbild ↔ Modell-/CULane-Raum
    sx_model_w = W / MODEL_W          # x: MODEL_W -> W
    sy_canon_h = H / CANON_H          # y: 590 -> H

    if DEBUG:
        r_crop  = float(getattr(cfg, "crop_ratio", 1.0))
        crop_top = 1.0 - r_crop
        crop_y_top_img = int(round(crop_top * CANON_H * sy_canon_h))
        crop_y_bottom_img = int(round(CANON_H * sy_canon_h))

        print(f"\n[DEBUG] Frame:")
        print(f"  orig_wh=({W}x{H})  model_wh=({MODEL_W}x{MODEL_H})  canon_h={CANON_H}")
        print(f"  sx_model_w={sx_model_w:.3f}  sy_canon_h={sy_canon_h:.3f}")
        print(f"  row_anchor max={max(cfg.row_anchor):.3f}")
        print(f"  crop_ratio={r_crop:.3f}  -> crop_top={crop_top:.3f}")
        print(f"  crop_y_top_img={crop_y_top_img}  crop_y_bottom_img={crop_y_bottom_img}")
        print(f"  lanes_detected={len(lanes_xy)}")

    # vis-Bild ist einfach eine Kopie des Originalframes
    vis = frame_bgr.copy()

    if DEBUG:
        r_crop  = float(getattr(cfg, "crop_ratio", 1.0))
        crop_top = 1.0 - r_crop
        crop_y_top_img = int(round(crop_top * CANON_H * sy_canon_h))
        crop_y_bottom_img = int(round(CANON_H * sy_canon_h))

        cv2.line(vis, (0, crop_y_top_img), (W-1, crop_y_top_img), (255, 0, 0), 2)
        cv2.putText(vis, "CULane crop top", (10, crop_y_top_img - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        overlay = vis.copy()
        cv2.rectangle(overlay, (0, crop_y_top_img),
                      (W-1, min(H-1, crop_y_bottom_img)),
                      (255, 0, 0), thickness=-1)
        alpha = 0.1
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

    vis = draw_lanes_mixed(
        vis, lanes_xy,
        sx_model_w=sx_model_w,
        sy_canon_h=sy_canon_h,
        color=(0,255,0),
        thickness=3,
        debug=DEBUG,
        lanes_info=lanes_info
    )

    return vis, lanes_xy, lanes_info