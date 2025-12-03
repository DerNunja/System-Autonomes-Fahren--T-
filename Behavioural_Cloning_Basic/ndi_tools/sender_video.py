import cv2
import time
from typing import Optional, Tuple

from visiongraph_ndi.NDIVideoOutput import NDIVideoOutput

# ========================== USER CONFIG ========================== #
USE_LIVE_SOURCE = False

VIDEO_PATH = "/home/konrada/projects/Uni/ProjektAutonomesFahren/Behavioural_Cloning_Basic/data/Recordings/Video/ego_h264.mp4"
LIVE_SOURCE = 0

TARGET_SIZE: Optional[Tuple[int, int]] = (640, 360)
TARGET_FPS: Optional[float] = 60.0
# ================================================================= #


def main():
    # Quelle öffnen
    if USE_LIVE_SOURCE:
        print(f"[INFO] Öffne Live-Quelle: {LIVE_SOURCE}")
        cap = cv2.VideoCapture(LIVE_SOURCE, cv2.CAP_DSHOW)
    else:
        print(f"[INFO] Öffne Videodatei: { VIDEO_PATH }")
        cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Video-/Livequelle konnte nicht geöffnet werden.")
        return

    # FPS bestimmen
    if USE_LIVE_SOURCE:
        fps = TARGET_FPS if TARGET_FPS else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    else:
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0 or orig_fps > 200:
            orig_fps = 30.0
        fps = TARGET_FPS if TARGET_FPS else orig_fps

    frame_time = 1.0 / fps if fps > 0 else 0.0

    with NDIVideoOutput("Demo") as ndi:
        print(f"[INFO] NDI Sender gestartet: Demo ({fps:.1f} FPS, target size={TARGET_SIZE})")

        total_frames = 0
        t_start = time.time()
        next_send_time = t_start

        # Profiling-Summen
        sum_read_time = 0.0
        sum_resize_time = 0.0
        sum_send_time = 0.0
        sum_preview_time = 0.0
        sum_sleep_time = 0.0

        while True:
            # -------- Frame aus Quelle lesen --------
            read_t0 = time.time()
            ret, frame = cap.read()
            read_t1 = time.time()
            read_dt = read_t1 - read_t0
            sum_read_time += read_dt

            if not ret:
                if USE_LIVE_SOURCE:
                    time.sleep(0.01)
                    continue
                else:
                    print("[INFO] Video zu Ende.")
                    break

            # -------- Resize --------
            resize_t0 = time.time()
            if TARGET_SIZE is not None:
                frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            resize_t1 = time.time()
            resize_dt = resize_t1 - resize_t0
            sum_resize_time += resize_dt

            # -------- NDI-Senden --------
            send_t0 = time.time()
            ndi.send(frame)
            send_t1 = time.time()
            send_dt = send_t1 - send_t0
            sum_send_time += send_dt

            total_frames += 1

            # -------- Preview (optional) --------
            preview_t0 = time.time()
            cv2.imshow("Sender Preview", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("[INFO] Abbruch durch Benutzer.")
                preview_t1 = time.time()
                sum_preview_time += (preview_t1 - preview_t0)
                break
            preview_t1 = time.time()
            preview_dt = preview_t1 - preview_t0
            sum_preview_time += preview_dt

            # -------- Framerate drosseln --------
            sleep_dt = 0.0
            if frame_time > 0:
                next_send_time += frame_time
                now = time.time()
                sleep_time = next_send_time - now
                if sleep_time > 0:
                    sleep_t0 = time.time()
                    time.sleep(sleep_time)
                    sleep_t1 = time.time()
                    sleep_dt = sleep_t1 - sleep_t0
                else:
                    next_send_time = now
            sum_sleep_time += sleep_dt

            # Optional: pro Frame loggen
            loop_dt = read_dt + resize_dt + send_dt + preview_dt + sleep_dt
            eff_fps_inst = 1.0 / loop_dt if loop_dt > 0 else 0.0
            print(
                f"[SEND {total_frames:05d}] "
                f"read={read_dt*1000:5.2f} ms  "
                f"resize={resize_dt*1000:5.2f} ms  "
                f"send={send_dt*1000:5.2f} ms  "
                f"preview={preview_dt*1000:5.2f} ms  "
                f"sleep={sleep_dt*1000:5.2f} ms  "
                f"eff_FPS={eff_fps_inst:5.1f}"
            )

        t_total = time.time() - t_start
        if total_frames > 0 and t_total > 0:
            eff_fps = total_frames / t_total
            print(f"\n[STATS] Sent {total_frames} frames in {t_total:.2f}s -> effective send FPS = {eff_fps:.2f}")

            avg_read_ms = (sum_read_time / total_frames) * 1000.0
            avg_resize_ms = (sum_resize_time / total_frames) * 1000.0
            avg_send_ms = (sum_send_time / total_frames) * 1000.0
            avg_preview_ms = (sum_preview_time / total_frames) * 1000.0
            avg_sleep_ms = (sum_sleep_time / total_frames) * 1000.0

            total_ms = (t_total / total_frames) * 1000.0

            def pct(part_ms: float) -> float:
                return (part_ms / total_ms * 100.0) if total_ms > 0 else 0.0

            print("[BREAKDOWN SENDER] Durchschnitt pro Frame:")
            print(f"  read      = {avg_read_ms:6.2f} ms ({pct(avg_read_ms):5.1f}% der Zeit)")
            print(f"  resize    = {avg_resize_ms:6.2f} ms ({pct(avg_resize_ms):5.1f}% der Zeit)")
            print(f"  send      = {avg_send_ms:6.2f} ms ({pct(avg_send_ms):5.1f}% der Zeit)")
            print(f"  preview   = {avg_preview_ms:6.2f} ms ({pct(avg_preview_ms):5.1f}% der Zeit)")
            print(f"  sleep     = {avg_sleep_ms:6.2f} ms ({pct(avg_sleep_ms):5.1f}% der Zeit)")
            print(f"  total     = {total_ms:6.2f} ms (gemittelte Frame-Dauer)")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Sender beendet.")


if __name__ == "__main__":
    main()
