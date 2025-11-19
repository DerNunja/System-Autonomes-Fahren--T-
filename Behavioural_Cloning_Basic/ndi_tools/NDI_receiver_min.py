import numpy as np
import cv2

from cyndilib.wrapper.ndi_recv import RecvColorFormat
from cyndilib.finder import Finder
from cyndilib.receiver import Receiver, ReceiveFrameType
from cyndilib.video_frame import VideoRecvFrame

SOURCE_NAME = "Simu_Video_Feed"


def run_receiver():
    # 1) Quellen suchen
    finder = Finder()
    finder.open()
    print(f"Suche nach NDI-Quelle, die '{SOURCE_NAME}' enthält...")

    source = None
    try:
        while source is None:
            finder.wait_for_sources(2.0)
            names = finder.update_sources()
            print("Gefundene Quellen:", names)

            for name in names:
                if SOURCE_NAME in name:
                    source = finder.get_source(name)
                    break
    finally:
        finder.close()

    if source is None:
        print("Keine passende NDI-Quelle gefunden.")
        return

    print("Quelle gewählt:", source.name)

    # 2) Receiver anlegen
    recv = Receiver(
        source_name=source.name,
        color_format=RecvColorFormat.BGRX_BGRA,
    )

    # explizit mit der Quelle verbinden
    try:
        recv.connect_to(source)
        print("[NDI] connect_to(source) aufgerufen.")
    except Exception as e:
        print(f"[NDI] connect_to(source) Fehler: {e}")

    # Status ausgeben
    try:
        connected = recv.is_connected()
    except Exception as e:
        connected = f"Fehler bei is_connected(): {e}"
    try:
        num_conns = recv.get_num_connections()
    except Exception as e:
        num_conns = f"Fehler bei get_num_connections(): {e}"

    print(f"[NDI] is_connected: {connected}")
    print(f"[NDI] get_num_connections: {num_conns}")

    # 3) Video-Frame-Container registrieren
    vf = VideoRecvFrame()
    recv.set_video_frame(vf)

    max_debug_frames = 15
    num_video_frames = 0
    saved_frame = False

    try:
        for i in range(max_debug_frames):
            ft = recv.receive(ReceiveFrameType.recv_video, 2000)
            print(f"[NDI] receive() call {i+1} -> FrameType: {ft} (int={int(ft)})")

            if ft == ReceiveFrameType.nothing:
                print("  Kein Frame (timeout).")
                continue

            if ft & ReceiveFrameType.recv_video:
                frame_arr = vf.current_frame_data
                if frame_arr is None:
                    print("  current_frame_data ist None.")
                    continue

                num_video_frames += 1

                print(
                    f"  raw frame_arr.shape={frame_arr.shape}, "
                    f"ndim={frame_arr.ndim}, size={frame_arr.size}, dtype={frame_arr.dtype}"
                )

                # Auflösung und ggf. Kanäle bestimmen
                try:
                    xres, yres = vf.get_resolution()  # (width, height)
                except Exception as e:
                    print(f"  get_resolution() Fehler: {e}")
                    continue

                print(f"  reported resolution: xres={xres}, yres={yres}")

                if xres <= 0 or yres <= 0:
                    print("  Ungültige Auflösung, breche diesen Frame ab.")
                    continue

                pixels = xres * yres
                if frame_arr.size % pixels != 0:
                    print("  frame_arr.size passt nicht zu xres*yres, breche ab.")
                    continue

                channels = frame_arr.size // pixels
                print(f"  abgeleitete channels={channels}")

                if channels not in (1, 3, 4):
                    print("  Unerwartete Kanalanzahl, breche ab.")
                    continue

                # reshapen
                frame_reshaped = frame_arr.reshape((yres, xres, channels)) if channels > 1 else frame_arr.reshape((yres, xres))
                min_val = int(frame_reshaped.min())
                max_val = int(frame_reshaped.max())
                print(f"  reshaped min={min_val}, max={max_val}")

                if not saved_frame and frame_reshaped.size > 0:
                    # Konvertiere ggf. BGRA -> BGR für Save
                    if channels == 4:
                        bgr = cv2.cvtColor(frame_reshaped, cv2.COLOR_BGRA2BGR)
                    elif channels == 3:
                        bgr = frame_reshaped
                    else:
                        bgr = frame_reshaped

                    out_path = "ndi_debug_frame.png"
                    try:
                        cv2.imwrite(out_path, bgr)
                        print(f"  Erste Frame-Kopie gespeichert als {out_path}")
                        saved_frame = True
                    except Exception as e:
                        print(f"  Fehler beim Speichern von {out_path}: {e}")

        try:
            perf = recv.get_performance_data()
            print(f"[NDI] Performance: {perf}")
        except Exception as e:
            print(f"[NDI] get_performance_data() Fehler: {e}")

    finally:
        if hasattr(recv, "disconnect"):
            recv.disconnect()
        if hasattr(recv, "destroy"):
            recv.destroy()
        print("Receiver beendet.")


if __name__ == "__main__":
    run_receiver()
