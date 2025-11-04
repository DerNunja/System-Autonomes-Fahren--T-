import cv2
import numpy as np
from pathlib import Path


video_path = Path("Behavioural_Cloning/data/Recordings/Video/u_2025-05-23 10-47-10_h264.mp4")
fallback_fps = 60.0  

# Fenstername
WIN = "Crop Tuner (m=Mouse to crop, s=print, r=reset, q=quit)"


def clamp(v, lo, hi): return max(lo, min(int(v), hi))

def read_frame_at_time(cap, t_sec, fps):
    frame_idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)))
        ok, frame = cap.read()
    return ok, frame

def apply_crop_and_scale(img, x, y, w, h, scale_percent):
    H, W = img.shape[:2]
    x = clamp(x, 0, W-1)
    y = clamp(y, 0, H-1)
    w = clamp(w, 1, W - x)
    h = clamp(h, 1, H - y)
    crop = img[y:y+h, x:x+w].copy()
    sp = max(1, int(scale_percent))
    if sp != 100:
        out_w = max(1, int(round(crop.shape[1] * sp / 100.0)))
        out_h = max(1, int(round(crop.shape[0] * sp / 100.0)))
        interp = cv2.INTER_AREA if sp < 100 else cv2.INTER_LINEAR
        crop = cv2.resize(crop, (out_w, out_h), interpolation=interp)
    else:
        out_w, out_h = crop.shape[1], crop.shape[0]
    return crop, (out_w, out_h)


def main():
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Kann Video nicht öffnen: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or fallback_fps
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps if fps > 0 else 0
    print(f"[INFO] Video: {W}x{H} @ {fps:.3f} FPS, Dauer ≈ {duration:.2f} s, Frames={n_frames}")

    x0, y0, w0, h0 = 0, 0, W, H
    scale0 = 100  
    t0 = 0

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("x", WIN, x0, W-1, lambda v: None)
    cv2.createTrackbar("y", WIN, y0, H-1, lambda v: None)
    cv2.createTrackbar("w", WIN, w0, W,   lambda v: None)
    cv2.createTrackbar("h", WIN, h0, H,   lambda v: None)
    cv2.createTrackbar("scale %", WIN, scale0, 200, lambda v: None)  # bis 200%
    cv2.createTrackbar("t (s)", WIN, int(t0), max(1, int(duration)), lambda v: None)

    last_params = None
    while True:
        x = cv2.getTrackbarPos("x", WIN)
        y = cv2.getTrackbarPos("y", WIN)
        w = cv2.getTrackbarPos("w", WIN)
        h = cv2.getTrackbarPos("h", WIN)
        sp = max(1, cv2.getTrackbarPos("scale %", WIN))
        t = cv2.getTrackbarPos("t (s)", WIN)

        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        params = (x, y, w, h, sp, t)
        if last_params is None or params[-1] != last_params[-1]:
            ok, frame = read_frame_at_time(cap, t, fps)
            if not ok:
                ok, frame = cap.read()
                if not ok:
                    print("[WARN] Kein Frame verfügbar.")
                    break
            base = frame.copy()
        preview = base.copy()
        cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(preview, f"ROI: x={x} y={y} w={w} h={h} | scale={sp}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(preview, f"t={t}s  out~{int(w*sp/100)}x{int(h*sp/100)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,255,180), 2, cv2.LINE_AA)

        crop_scaled, (ow, oh) = apply_crop_and_scale(base, x, y, w, h, sp)
        max_thumb_w = 640
        thumb = crop_scaled
        if thumb.shape[1] > max_thumb_w:
            th = int(round(thumb.shape[0] * (max_thumb_w / thumb.shape[1])))
            thumb = cv2.resize(thumb, (max_thumb_w, th), interpolation=cv2.INTER_AREA)
        pad_h = max(preview.shape[0], thumb.shape[0])
        canvas = np.zeros((pad_h, preview.shape[1] + thumb.shape[1] + 10, 3), dtype=np.uint8)
        canvas[:preview.shape[0], :preview.shape[1]] = preview
        canvas[:thumb.shape[0], preview.shape[1]+10:preview.shape[1]+10+thumb.shape[1]] = thumb
        cv2.putText(canvas, f"OUTPUT {ow}x{oh}", (preview.shape[1]+10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2, cv2.LINE_AA)

        cv2.imshow(WIN, canvas)
        last_params = params

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):  
            break
        elif key == ord('s'):
            print("\n=== Crop/Scale Einstellungen ===")
            print(f"crop_box       = ({x}, {y}, {w}, {h})")
            print(f"scale_percent  = {sp}")
            print(f"preview_size   ≈ {ow}x{oh}")
            print(f"at time (sec)  = {t}")
            print("================================\n")
        elif key == ord('r'):
            cv2.setTrackbarPos("x", WIN, 0)
            cv2.setTrackbarPos("y", WIN, 0)
            cv2.setTrackbarPos("w", WIN, W)
            cv2.setTrackbarPos("h", WIN, H)
            cv2.setTrackbarPos("scale %", WIN, 100)
        elif key == ord('m'):
            tmp = base.copy()
            roi = cv2.selectROI("selectROI (ENTER=OK, c=cancel)", tmp, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("selectROI (ENTER=OK, c=cancel)")
            rx, ry, rw, rh = map(int, roi)
            if rw > 0 and rh > 0:
                cv2.setTrackbarPos("x", WIN, rx)
                cv2.setTrackbarPos("y", WIN, ry)
                cv2.setTrackbarPos("w", WIN, rw)
                cv2.setTrackbarPos("h", WIN, rh)
        elif key == ord('a'):  
            cv2.setTrackbarPos("t (s)", WIN, max(0, t-1))
        elif key == ord('d'):  
            cv2.setTrackbarPos("t (s)", WIN, min(int(duration), t+1))
        elif key == ord('z'):  
            cv2.setTrackbarPos("t (s)", WIN, max(0, t-5))
        elif key == ord('c'): 
            cv2.setTrackbarPos("t (s)", WIN, min(int(duration), t+5))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()