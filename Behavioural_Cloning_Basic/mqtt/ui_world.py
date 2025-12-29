# ui_world_monitor.py
import json
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk

import paho.mqtt.client as mqtt

BROKER = "localhost"
TOPIC_WORLD_STATE = "sensor/lanestate"

# Queue für Thread-sichere Übergabe MQTT → GUI
message_queue = queue.Queue()


# ========== MQTT-Callbacks ==========

def on_connect(client, userdata, flags, rc):
    print("MQTT verbunden mit Code:", rc)
    client.subscribe(TOPIC_WORLD_STATE, qos=0)
    print(f"Abonniere {TOPIC_WORLD_STATE}")


def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", errors="ignore")
    message_queue.put((msg.topic, payload))


def start_mqtt():
    client = mqtt.Client(client_id="ui-world-monitor")
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, 1883, keepalive=60)
    client.loop_start()
    return client


# ========== Tkinter-UI ==========

class WorldMonitorUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("World State Monitor")
        self.geometry("700x600")

        # interner Zustand
        self.world = {
            "objects": [],
            "lanestate": {"lane_center": 0.0, "curvature": 0.0},
            "vehicle_state": {},             # <- neu
            "last_update_ts": None
        }

        self._build_widgets()

        # Polling der MQTT-Queue alle 100 ms
        self.after(100, self.process_queue)

    def _build_widgets(self):
        # --- Status-Zeile ---
        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(status_frame, text="World State Topic:").pack(side="left")
        self.topic_label = ttk.Label(status_frame, text=TOPIC_WORLD_STATE)
        self.topic_label.pack(side="left", padx=5)

        self.time_label = ttk.Label(status_frame, text="Letztes Update: –")
        self.time_label.pack(side="right")

        # --- Objekte ---
        objects_frame = ttk.LabelFrame(self, text="Erkannte Objekte")
        objects_frame.pack(fill="both", expand=True, padx=10, pady=5)

        columns = ("type", "distance", "speed_or_status")
        self.objects_tree = ttk.Treeview(
            objects_frame,
            columns=columns,
            show="headings",
            height=6
        )
        self.objects_tree.heading("type", text="Typ")
        self.objects_tree.heading("distance", text="Distanz [m]")
        self.objects_tree.heading("speed_or_status", text="Speed/Status")

        self.objects_tree.column("type", width=120)
        self.objects_tree.column("distance", width=120)
        self.objects_tree.column("speed_or_status", width=150)

        self.objects_tree.pack(fill="both", expand=True, padx=5, pady=5)

        # Kurze Zusammenfassung (z. B. nächstes Auto)
        self.summary_label = ttk.Label(objects_frame, text="Nähestes Objekt: –")
        self.summary_label.pack(anchor="w", padx=5, pady=5)

        # --- Lane State ---
        lane_frame = ttk.LabelFrame(self, text="Lane State")
        lane_frame.pack(fill="x", padx=10, pady=5)

        self.lane_info_label = ttk.Label(lane_frame, text="lane_center: –, curvature: –")
        self.lane_info_label.pack(anchor="w", padx=5, pady=5)

        # Visualisierung: Lane-Center
        self.lane_canvas = tk.Canvas(lane_frame, width=300, height=60, bg="white")
        self.lane_canvas.pack(padx=5, pady=5)

        # Mittellinie
        self.lane_canvas.create_line(150, 0, 150, 60, dash=(4, 2))
        # Fahrzeug-Rechteck
        self.car_rect = self.lane_canvas.create_rectangle(140, 25, 160, 45, fill="blue")

        # --- Vehicle State (Ego-Fahrzeug) ---
        vehicle_frame = ttk.LabelFrame(self, text="Vehicle State (Ego-Fahrzeug)")
        vehicle_frame.pack(fill="x", padx=10, pady=5)

        # StringVars für Werte
        self.throttle_var = tk.StringVar(value="–")
        self.brakes_var = tk.StringVar(value="–")
        self.wheel_pos_var = tk.StringVar(value="–")
        self.engine_rpm_var = tk.StringVar(value="–")
        self.brakes_vol_var = tk.StringVar(value="–")
        self.timestamp_var = tk.StringVar(value="–")

        # Layout in Grid
        row = 0
        ttk.Label(vehicle_frame, text="Timestamp [s]:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(vehicle_frame, textvariable=self.timestamp_var).grid(row=row, column=1, sticky="w", padx=5, pady=2)

        row += 1
        ttk.Label(vehicle_frame, text="Throttle [%]:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(vehicle_frame, textvariable=self.throttle_var).grid(row=row, column=1, sticky="w", padx=5, pady=2)

        row += 1
        ttk.Label(vehicle_frame, text="Brakes [%]:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(vehicle_frame, textvariable=self.brakes_var).grid(row=row, column=1, sticky="w", padx=5, pady=2)

        row += 1
        ttk.Label(vehicle_frame, text="Wheel Position [rad]:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(vehicle_frame, textvariable=self.wheel_pos_var).grid(row=row, column=1, sticky="w", padx=5, pady=2)

        row += 1
        ttk.Label(vehicle_frame, text="Engine RPM:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(vehicle_frame, textvariable=self.engine_rpm_var).grid(row=row, column=1, sticky="w", padx=5, pady=2)

        row += 1
        ttk.Label(vehicle_frame, text="Brakes Vol:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(vehicle_frame, textvariable=self.brakes_vol_var).grid(row=row, column=1, sticky="w", padx=5, pady=2)

        # --- Log (optional für Debug) ---
        log_frame = ttk.LabelFrame(self, text="Roh-JSON von world/state")
        log_frame.pack(fill="both", expand=False, padx=10, pady=5)

        self.log_text = tk.Text(log_frame, height=8)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_text.configure(state="disabled", font=("Consolas", 9))

    # ===== Queue-Verarbeitung =====

    def process_queue(self):
        while True:
            try:
                topic, payload = message_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self.handle_message(topic, payload)

        self.after(100, self.process_queue)

    # ===== Verarbeitung einer Weltmodell-Nachricht =====

    def handle_message(self, topic, payload):
        if topic != TOPIC_WORLD_STATE:
            return

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            print("Fehler beim JSON-Parsing:", payload)
            return

        # Erwartete Struktur wie in deiner Bridge
        self.world["objects"] = data.get("objects", [])
        self.world["lanestate"] = data.get("lanestate", self.world["lanestate"])
        self.world["vehicle_state"] = data.get("vehicle_state", self.world.get("vehicle_state", {}))
        self.world["last_update_ts"] = data.get("last_update_ts", None)

        # UI updaten
        self.update_time_label()
        self.update_objects_view()
        self.update_lane_view()
        self.update_vehicle_view()
        self.update_log(payload)

    def update_time_label(self):
        ts = self.world.get("last_update_ts")
        if ts is None:
            self.time_label.config(text="Letztes Update: –")
            return

        t_local = time.localtime(ts)
        t_str = time.strftime("%H:%M:%S", t_local)
        self.time_label.config(text=f"Letztes Update: {t_str}")

    def update_objects_view(self):
        # Treeview leeren
        for item in self.objects_tree.get_children():
            self.objects_tree.delete(item)

        objects = self.world.get("objects", [])

        nearest_obj = None
        nearest_dist = None

        for obj in objects:
            obj_type = obj.get("type", "?")
            dist = obj.get("distance", None)
            speed = obj.get("speed", None)
            status = obj.get("status", None)

            if dist is not None:
                dist_str = f"{dist:.1f}"
            else:
                dist_str = "–"

            # entweder speed oder status anzeigen
            if speed is not None:
                s_str = f"{speed:.1f} m/s"
            elif status is not None:
                s_str = str(status)
            else:
                s_str = "–"

            self.objects_tree.insert(
                "",
                "end",
                values=(obj_type, dist_str, s_str)
            )

            # Nächstes Objekt
            if dist is not None:
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_obj = obj

        if nearest_obj is None:
            self.summary_label.config(text="Nähestes Objekt: –")
        else:
            self.summary_label.config(
                text=f"Nähestes Objekt: {nearest_obj.get('type','?')} bei {nearest_dist:.1f} m"
            )

    def update_lane_view(self):
        ls = self.world.get("lanestate", {})
        lane_center = ls.get("lane_center", 0.0)
        curvature = ls.get("curvature", 0.0)

        self.lane_info_label.config(
            text=f"lane_center: {lane_center:.3f} m, curvature: {curvature:.6f}"
        )

        # lane_center grafisch darstellen:
        max_offset_m = 1.0
        x_center = 150
        max_pixels = 100  # ±100 px max

        factor = max_pixels / max_offset_m
        offset_px = max(-max_pixels, min(max_pixels, lane_center * factor))

        new_x1 = x_center - 10 + offset_px
        new_x2 = x_center + 10 + offset_px

        # Rechteck verschieben
        self.lane_canvas.coords(self.car_rect, new_x1, 25, new_x2, 45)

    def update_vehicle_view(self):
        vs = self.world.get("vehicle_state", {}) or {}

        def fmt(value, fmt_str="{:.3f}", default="–"):
            try:
                return fmt_str.format(float(value))
            except Exception:
                return default

        # timestamp
        ts = vs.get("timestamp", None)
        if ts is None:
            self.timestamp_var.set("–")
        else:
            self.timestamp_var.set(fmt(ts, "{:.3f}"))

        # throttle / brakes in %
        self.throttle_var.set(fmt(vs.get("throttle", None), "{:.1f}"))
        self.brakes_var.set(fmt(vs.get("brakes", None), "{:.1f}"))

        # wheel position
        self.wheel_pos_var.set(fmt(vs.get("wheel_position", None), "{:.6f}"))

        # engine rpm
        self.engine_rpm_var.set(fmt(vs.get("engine_rpm", None), "{:.0f}"))

        # brakes_vol
        self.brakes_vol_var.set(fmt(vs.get("brakes_vol", None), "{:.3f}"))

    def update_log(self, payload):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        try:
            parsed = json.loads(payload)
            pretty = json.dumps(parsed, indent=2)
        except Exception:
            pretty = payload
        self.log_text.insert("end", pretty)
        self.log_text.configure(state="disabled")


# ========== main ==========

def main():
    client = start_mqtt()

    app = WorldMonitorUI()
    try:
        app.mainloop()
    finally:
        client.loop_stop()
        client.disconnect()
        print("UI beendet, MQTT getrennt.")


if __name__ == "__main__":
    main()
