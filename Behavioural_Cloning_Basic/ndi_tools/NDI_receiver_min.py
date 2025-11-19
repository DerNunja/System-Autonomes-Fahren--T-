import NDIlib as ndi
import numpy as np
import time

SOURCE_NAME = "Simu_Video_Feed"
MAX_FRAMES = 10


def main():
    print("[NDI] initialize() ...")
    if not ndi.initialize():
        print("NDI konnte nicht initialisiert werden.")
        return

    print("[NDI] find_create_v2() ...")
    finder = ndi.find_create_v2()
    if not finder:
        print("Fehler beim Erstellen des Finders.")
        ndi.destroy()
        return

    source = None
    try:
        print(f"[NDI] Suche nach Quellen, die '{SOURCE_NAME}' enthalten ...")
        while source is None:
            ndi.find_wait_for_sources(finder, 2000)
            sources = ndi.find_get_current_sources(finder) or []
            names = [s.ndi_name for s in sources]
            print("[NDI] Gefundene Quellen:", names)

            for s in sources:
                if SOURCE_NAME in s.ndi_name:
                    source = s
                    break

        print(f"[NDI] Quelle gewählt: {source.ndi_name}")
    finally:
        print("[NDI] find_destroy()")
        ndi.find_destroy(finder)

    print("[NDI] recv_create_v3() ...")
    recv_create = ndi.RecvCreateV3()
    recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
    # Bandbreite hier NICHT anfassen, das Feld gibt's in V3 nicht
    recv = ndi.recv_create_v3(recv_create)

    if not recv:
        print("Fehler beim Erstellen des NDI-Receivers.")
        ndi.destroy()
        return

    print("[NDI] recv_connect() ...")
    ndi.recv_connect(recv, source)

    print(f"[NDI] Empfange bis zu {MAX_FRAMES} Video-Frames (ohne Anzeige) ...")
    frame_count = 0
    start_all = time.time()

    try:
        while frame_count < MAX_FRAMES:
            t, v_frame, a_frame, m_frame = ndi.recv_capture_v2(recv, 5000)

            print(f"[NDI] recv_capture_v2 -> t={t}")

            if t == ndi.FRAME_TYPE_VIDEO:
                frame_count += 1

                xres = v_frame.xres
                yres = v_frame.yres
                frame_bgra = np.copy(v_frame.data)

                print(
                    f"[NDI][Frame {frame_count}] xres={xres}, yres={yres}, "
                    f"shape={frame_bgra.shape}, dtype={frame_bgra.dtype}, "
                    f"min={frame_bgra.min()}, max={frame_bgra.max()}"
                )

                ndi.recv_free_video_v2(recv, v_frame)

            elif t == ndi.FRAME_TYPE_NONE:
                print("[NDI] Kein Frame (FRAME_TYPE_NONE) – Timeout ohne Daten.")
            elif t == ndi.FRAME_TYPE_STATUS_CHANGE:
                print("[NDI] STATUS_CHANGE (Verbindung/Format geändert)")
            elif t == ndi.FRAME_TYPE_AUDIO:
                print("[NDI] AUDIO-Frame (ignoriert)")
                ndi.recv_free_audio_v2(recv, a_frame)

        dur = time.time() - start_all
        print(f"[NDI] {frame_count} Video-Frames in {dur:.2f}s empfangen.")

    finally:
        print("[NDI] Aufräumen ...")
        ndi.recv_destroy(recv)
        ndi.destroy()
        print("[NDI] Fertig.")


if __name__ == "__main__":
    main()
