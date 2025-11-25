import time
from pathlib import Path

import cv2
from LaneDetection.lanedetec_runner import init_lanedetector, process_frame
from simulate_videostream import frame_stream_from_video


def main():
    print("[INFO] Initializing lane detector...")
    net, cfg, img_transforms, device = init_lanedetector()
    print("[INFO] Lane detector ready.")

    video_path = Path("/home/konrada/projects/Uni/ProjektAutonomesFahren/Behavioural_Cloning_Basic/data/Recordings/Video/ego_h264.mp4")  # <- anpassen

    total_frames = 0
    t_overall_start = time.time()

    cv2.namedWindow("LaneDetection Stream", cv2.WINDOW_NORMAL)

    for frame_idx, ts, frame_bgr in frame_stream_from_video(
        video_path,
        loop=False,          
        simulate_realtime=True,
        target_size=(1280, 720)
    ):
        t0 = time.time()

        vis_bgr, lanes_xy, lanes_info = process_frame(
            frame_bgr, net, cfg, img_transforms, device
        )

        t1 = time.time()
        dt = t1 - t0
        fps_inst = 1.0 / dt if dt > 0 else 0.0

        print(
            f"[FRAME {frame_idx:05d}] ts={ts:6.3f}s  "
            f"process_time={dt*1000:6.1f} ms  "
            f"FPS_inst={fps_inst:5.1f}  lanes={len(lanes_xy)}"
        )

        overlay_text = f"{fps_inst:4.1f} FPS"
        cv2.putText(
            vis_bgr,
            overlay_text,
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("LaneDetection Stream", vis_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        total_frames += 1

    t_overall = time.time() - t_overall_start
    if total_frames > 0:
        avg_fps = total_frames / t_overall
        print(
            f"\n[STATS] Processed {total_frames} frames in "
            f"{t_overall:.2f}s -> avg FPS = {avg_fps:.2f}"
        )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
