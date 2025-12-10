def process_frame(frame_bgr, net, cfg, img_transforms, device):
    """
    Nimmt ein BGR-Frame (numpy, z.B. aus OpenCV/NDI) und gibt zurück:
      vis        -> BGR-Frame mit eingezeichneten Lanes (für Debug/Anzeige)
      lanes_xy   -> Liste von Lanes, jede Lane ist eine Liste von (x_model, y_canon)
      lanes_info -> Zusatzinfos (Score, Anzahl Punkte, Lane-Index, ...)
    """

    # Originalgröße des Eingangsbilds
    H, W = frame_bgr.shape[:2]

    MODEL_W, MODEL_H = cfg.train_width, cfg.train_height
    CANON_H = 590.0  # CULane-Referenzhöhe

    # BGR (OpenCV) -> RGB (PIL), weil deine Transforms mit PIL arbeiten
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # Normalisieren / Resizen wie beim Training
    image_tensor = img_transforms(pil_img).unsqueeze(0).to(device)

    # Vorwärtslauf durch das Netz
    with torch.no_grad():
        pred = net(image_tensor)

    # Lane-Punkte im "gemischten Raum" holen
    lanes_xy, lanes_info = pred2coords_mixed(
        pred,
        cfg.row_anchor,
        model_w=MODEL_W,
        cfg=cfg,
        thr_row=0.72,
        local_width=2,
        topk_lanes=2,
        smooth_kernel=5
    )

    # Skalierungsfaktoren: Modell-Raum → Bild-Raum
    sx_model_w = W / float(MODEL_W)    # x: MODEL_W -> W
    sy_canon_h = H / float(CANON_H)    # y: 590 -> H

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

    # vis-Bild ist eine Kopie des Originals
    vis = frame_bgr.copy()

    # Optional: CULane-Crop-Bereich markieren (nur Debug)
    if DEBUG:
        r_crop  = float(getattr(cfg, "crop_ratio", 1.0))
        crop_top = 1.0 - r_crop
        crop_y_top_img = int(round(crop_top * CANON_H * sy_canon_h))
        crop_y_bottom_img = int(round(CANON_H * sy_canon_h))

        cv2.line(vis, (0, crop_y_top_img), (W-1, crop_y_top_img), (255, 0, 0), 2)
        cv2.putText(vis, "CULane crop top", (10, crop_y_top_img - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        overlay = vis.copy()
        cv2.rectangle(
            overlay,
            (0, crop_y_top_img),
            (W-1, min(H-1, crop_y_bottom_img)),
            (255, 0, 0),
            thickness=-1
        )
        alpha = 0.1
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

    # Lanes in das Bild malen
    vis = draw_lanes_mixed(
        vis,
        lanes_xy,
        sx_model_w=sx_model_w,
        sy_canon_h=sy_canon_h,
        color=(0, 255, 0),
        thickness=3,
        debug=DEBUG,
        lanes_info=lanes_info
    )

    # WICHTIG: genau diese drei Werte zurückgeben
    return vis, lanes_xy, lanes_info
