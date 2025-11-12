from PIL import Image
import torch, os, cv2
from utils.common import merge_config, get_model
import torchvision.transforms as transforms
from model.model_culane import parsingNet as LaneNet
import numpy as np


def pred2coords_mixed(pred, row_anchor, model_w, cfg,
                      thr_row=0.75, local_width=2, topk_lanes=2, smooth_kernel=5):
    _, G_r, C_r, L = pred['loc_row'].shape
    loc_row = pred['loc_row'][0].cpu()
    max_idx_row = pred['loc_row'].argmax(1)[0].cpu()
    exist_row_p  = pred['exist_row'].softmax(1)[0,1].cpu()

    lane_scores = exist_row_p.mean(0)
    picked = torch.argsort(lane_scores, descending=True)[:min(topk_lanes, L)].tolist()

    CANON_H = 590.0

    lanes_xy = []
    for lane in picked:
        xs, ys = [], []
        for k in range(C_r):
            if float(exist_row_p[k, lane]) < thr_row:
                continue
            center = int(max_idx_row[k, lane])
            left  = max(0, center - local_width)
            right = min(G_r - 1, center + local_width)
            inds = torch.arange(left, right+1)
            x_hat = (loc_row[inds, k, lane].softmax(0) * inds.float()).sum() + 0.5

            
            x = float(x_hat) / (G_r - 1) * model_w             # x im MODEL_W-Raum
            r = float(getattr(cfg, "crop_ratio", 1.0))     # z.B. 0.6 bei CULane
            crop_top = 1.0 - r                             # oberer Versatz im 0..1-Raum
            y = (crop_top + float(row_anchor[k]) * r) * CANON_H   # 590 = CANON_H
            xs.append(x)
            ys.append(y)

        if len(xs) >= smooth_kernel:
            xs_np = np.array(xs, dtype=np.float32)
            r = smooth_kernel // 2
            for i in range(len(xs_np)):
                l = max(0, i - r); r_ = min(len(xs_np), i + r + 1)
                xs_np[i] = np.median(xs_np[l:r_])
            xs = xs_np.tolist()

        lanes_xy.append(list(zip(xs, ys)))
    return lanes_xy


def draw_lanes_mixed(vis_bgr, lanes_xy, sx_model_w, sy_canon_h, color=(0,255,0), thickness=3):
    H, W = vis_bgr.shape[:2]
    for lane in lanes_xy:
        pts = []
        for (x_model, y_canon) in lane:
            xi = int(round(x_model * sx_model_w))  # skaliere x mit MODEL_W
            yi = int(round(y_canon * sy_canon_h))  # skaliere y mit CANON_H
            if 0 <= xi < W and 0 <= yi < H:
                pts.append([xi, yi])
        if len(pts) >= 2:
            cv2.polylines(vis_bgr, [np.array(pts, dtype=np.int32)],
                          isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        elif len(pts) == 1:
            cv2.circle(vis_bgr, tuple(pts[0]), 3, color, -1, lineType=cv2.LINE_AA)
    return vis_bgr

def process_image(image_path, net, cfg, img_transforms, device):
    global canon_h_for_anchor  # wird aus main übernommen
    pil_img = Image.open(image_path).convert('RGB')
    W, H = pil_img.size
    MODEL_W, MODEL_H = cfg.train_width, cfg.train_height
    CANON_W, CANON_H = 1640, 590

    # Inferenz im Netzraum
    image_tensor = img_transforms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = net(image_tensor)

    # Koordinaten im Netzraum holen
    
    lanes_xy = pred2coords_mixed(
        pred,
        cfg.row_anchor,
        model_w=MODEL_W,
        cfg=cfg,  # <- die erkannte Skala
        thr_row=0.72, local_width=2, topk_lanes=2, smooth_kernel=5
    )
    
    # Auf Originalbild skalieren + zeichnen
    sx_model_w = W / MODEL_W
    sy_canon_h = (H / CANON_H)
    print(CANON_W, CANON_H)
    print(sx_model_w, sy_canon_h)

    print(f"Image {os.path.basename(image_path)}:")
    print(f"  row_anchor max={max(cfg.row_anchor):.3f}")
    print(f"  canon_h_for_anchor={canon_h_for_anchor}")
    print(f"  scale x={W/MODEL_W:.3f}, y={H/CANON_H:.3f}")
    print("crop_ratio =", cfg.crop_ratio, "→ crop_top =", 1 - cfg.crop_ratio)
    print("sx =", W / cfg.train_width, "sy =", H / 590)

    vis = cv2.imread(image_path)
    vis = draw_lanes_mixed(vis, lanes_xy, sx_model_w, sy_canon_h, color=(0,255,0), thickness=3)
    return vis
    

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()
    CANON_W, CANON_H = 1640, 590

    # --- Heuristik: sind row_anchor Werte normiert oder in Pixeln ---
    row_anchor_np = np.array(cfg.row_anchor, dtype=float)
    row_anchor_are_ratio = float(row_anchor_np.max()) <= 1.5
    print(f"row_anchor max={row_anchor_np.max():.3f} ->",
          "RATIO" if row_anchor_are_ratio else "PIXELS@590")

    canon_h_for_anchor = CANON_H if row_anchor_are_ratio else 1.0

    cfg.dataset = 'CULane'
    cfg.batch_size = 1
    print('DATASET =', cfg.dataset)

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
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    in_dir = "/home/konrada/projects/Uni/ProjektAutonomesFahren/Behavioural_Cloning_Basic/data/Processed/frames"
    out_dir = "/home/konrada/projects/Uni/ProjektAutonomesFahren/Behavioural_Cloning_Basic/data/Processed/LaneDetection"
    os.makedirs(out_dir, exist_ok=True)

    # --- Hier Parameter übergeben ---
    for fname in os.listdir(in_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(in_dir, fname)
            dst = os.path.join(out_dir, fname)
            vis = process_image(src, net, cfg, img_transforms, device)
            cv2.imwrite(dst, vis)
            print(f"Processed {fname} -> {dst}")