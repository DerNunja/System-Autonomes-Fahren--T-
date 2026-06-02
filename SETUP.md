# Setup - Autonomes Fahren System

## Architektur

Zwei Rechner kommunizieren über NDI (Video) und MQTT (Wahrnehmungs- und Diagnosedaten):

```
[Simulator/Drive PC]                    [Perception PC]
┌─────────────────────────────┐          ┌─────────────────────────────┐
│ Fahrsimulator                │          │ sender_video.py (NDI)       │
│ mqtt_to_thrustmaster.py      │          │ receiver_video.py           │
│  ├─ drive_controller.py      │◄─ MQTT ─ │  ├─ Lane Detection (UFLD v2)│
│  ├─ Thrustmaster FFB         │ lanestate│  ├─ World Model             │
│  └─ steering_cmd diagnostics │─ MQTT ─► │  └─ MQTT Publisher          │
└─────────────────────────────┘          └─────────────────────────────┘
```

## Voraussetzungen

- Python >= 3.10
- CUDA-fähige GPU (für Lane Detection)
- Mosquitto MQTT Broker installiert und läuft
- NDI SDK auf beiden Rechnern installiert
- `visiongraph-ndi` Python Package

### Abhängigkeiten installieren

```bash
uv sync
```

### Mosquitto Broker starten

```bash
# Linux
sudo systemctl start mosquitto

# Oder manuell
mosquitto -p 1883
```

Der Broker läuft standardmäßig auf `localhost:1883`. Auf einem zweiten Rechner die Broker-IP in den Scripts anpassen (`BROKER = "IP_ADRESSE"`).

---

## Skripte auf dem Perception PC

### 1. NDI Video Sender (optional - nur wenn Simulator kein NDI direkt sendet)

Sendet einen Videostream über das Netzwerk. Nimmt entweder eine Videodatei oder eine Live-Kamera.

```bash
cd Behavioural_Cloning_Basic/ndi_tools/
uv run sender_video.py
```

- `VIDEO_PATH` in `sender_video.py` anpassen für eigene Videos
- `USE_LIVE_SOURCE = True` für Live-Kamera
- Sendet NDI-Stream mit Name **"Demo"**

### 2. Wahrnehmungsskript - Lane Detection

Empfängt NDI-Video, erkennt Fahrstreifen, bildet einen Ego-Spurzustand und publiziert diesen über MQTT.

```bash
cd Behavioural_Cloning_Basic/ndi_tools/
uv run receiver_video.py
```

**Was es macht:**
- Verbindet sich zum NDI-Stream "Demo"
- Lädt UFLD v2 Lane Detection Modell (ResNet34, ~826MB)
- Verarbeitet jeden Frame: Lane Detection → Weltmodell → Lane-State MQTT
- Publiziert erweiterten Lane-Status auf `sensor/lanestate`
- Zeigt zwei OpenCV-Fenster: Original-Video und annotiertes Video

**MQTT Topics die publiziert werden:**

| Topic | Inhalt |
|-------|--------|
| `sensor/lanestate` | `{t_ms, has_ego_lane, offset_m, heading_error_rad, curvature_preview, quality, lane_center, curvature}` |

---

## Skripte auf dem Simulator PC

### 3. MQTT → Thrustmaster Drive Controller

Empfängt Lane-State vom Perception PC, berechnet Lenkbefehle auf dem Steuerrechner und bewegt das Thrustmaster-Lenkrad per Force Feedback.

```bash
cd Behavioural_Cloning_Basic/drive/
uv run mqtt_to_thrustmaster.py
```

**Wichtig:** `BROKER` in `mqtt_to_thrustmaster.py` auf die IP des MQTT Brokers setzen (nicht leer lassen für Remote-Broker).

**Was es macht:**
- Subscribt auf `sensor/lanestate`
- Berechnet Lenkbefehle über `drive_controller.py`
- Liest die aktuelle physische Lenkradposition lokal aus
- Wendet an: Gain → Deadzone (0.03) → EMA-Smoothing (alpha=0.25)
- Bewegt das Thrustmaster-Lenkrad per Force Feedback
- Publiziert Diagnosewerte auf `control/steering_cmd`
- Safety-Timeout: bei 0.25s keine Lane-State-Nachrichten → Lenkrad zentrieren

**Parameter in `mqtt_to_thrustmaster.py`:**

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| `STEER_GAIN` | 1.0 | Lenkungsverstärkung |
| `STEER_DEADZONE` | 0.03 | Totzone gegen Zittern |
| `EMA_ALPHA` | 0.25 | Smoothing (niedriger = glatter) |
| `CMD_TIMEOUT_S` | 0.25 | Safety-Timeout in Sekunden |

---

## MQTT Topics Übersicht

```
control/steering_cmd      ← mqtt_to_thrustmaster.py publiziert Lenkdiagnostik
                            mqtt_to_xbox.py kann optional subscriben

sensor/lanestate          ← receiver_video.py publiziert Lane-Status
                            mqtt_to_thrustmaster.py subscribt
                            mqtt_broker.py aggregiert
                            ui_world.py subscribt (teilweise)

sensor/objects            ← Mock Perception (test/demo)
sensor/vehicle_state      ← CSV-Replay (Offline-Analyse)

world/state               ← mqtt_broker.py publiziert aggregierten Zustand
                            ui_world.py subscribt (konzeptionell)
```

---

## MQTT Bridge (optional - für Visualisierung)

Aggregiert alle Sensordaten und publiziert konsolidierten Weltzustand.

```bash
cd Behavioural_Cloning_Basic/mqtt/
uv run mqtt_broker.py
```

Publiziert alle 100ms (`world/state`) mit: `objects`, `lanestate`, `steering_cmd`, `vehicle_state`, `last_update_ts`

---

## UI Monitor (optional - für Debugging)

Tkinter-GUI zur Echtzeit-Überwachung.

```bash
cd Behavioural_Cloning_Basic/mqtt/
uv run ui_world.py
```

Zeigt: Lane-Offset, Krümmung, Fahrzeugzustand, erkannte Objekte, rohe JSON-Daten.

---

## Vollständiger Start-Ablauf

```bash
# 1. MQTT Broker starten (empfohlen: Perception PC)
mosquitto -p 1883

# 2. Perception PC: Video-Quelle starten
cd Behavioural_Cloning_Basic/ndi_tools/
uv run sender_video.py          # Falls Simulator kein NDI direkt sendet

# 3. Perception PC: Wahrnehmung starten
uv run receiver_video.py        # Lane Detection + sensor/lanestate Publishing

# 4. Simulator/Drive PC: MQTT → Thrustmaster starten
cd Behavioural_Cloning_Basic/drive/
uv run mqtt_to_thrustmaster.py  # Berechnet Lenkung + bewegt Lenkrad

# Optional: Bridge + UI auf einem beliebigen Rechner
cd Behavioural_Cloning_Basic/mqtt/
uv run mqtt_broker.py
uv run ui_world.py
```
