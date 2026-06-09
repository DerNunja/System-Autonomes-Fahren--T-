# Setup - Autonomes Fahren System

## Architektur

Zwei Rechner kommunizieren über NDI (Video) und MQTT (Wahrnehmungs- und Diagnosedaten):

```
[Simulator/Drive PC]                    [Perception PC]
┌─────────────────────────────┐          ┌─────────────────────────────┐
│ Fahrsimulator                │          │ video/send_video.py         │
│ mqtt_to_thrustmaster.py      │          │ perception/run_perception.py│
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

### 1. Video Sender (optional - nur wenn Simulator kein NDI direkt sendet)

Sendet einen Videostream über das Netzwerk. Nimmt entweder eine Videodatei oder eine Live-Kamera.

```bash
cd src/
uv run video/send_video.py
```

- `video_path` in `video/sender_app.py` anpassen für eigene Videos
- `use_live_source=True` in `VideoSenderConfig` für Live-Kamera
- Sendet NDI-Stream mit Name **"Demo"**

### 2. Wahrnehmungsskript - Lane Detection

Empfängt NDI-Video, erkennt Fahrstreifen, bildet einen Ego-Spurzustand und publiziert diesen über MQTT.

```bash
cd src/
uv run perception/run_perception.py
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
cd src/
uv run drive/mqtt_to_thrustmaster.py
```

**Wichtig:** `BROKER` in `mqtt_to_thrustmaster.py` auf die IP des MQTT Brokers setzen (nicht leer lassen für Remote-Broker).

**Was es macht:**
- Subscribt auf `sensor/lanestate`
- Berechnet Lenkbefehle über `drive_controller.py`
- Liest die aktuelle physische Lenkradposition lokal aus
- Wendet an: Gain → Deadzone (0.01) → EMA-Smoothing (alpha=0.35)
- Bewegt das Thrustmaster-Lenkrad per Force Feedback
- Lane Assist kann per Lenkrad-Button ein- und ausgeschaltet werden
- Publiziert Diagnosewerte auf `control/steering_cmd`
- Safety-Timeout: bei 0.25s keine Lane-State-Nachrichten → Lenkrad zentrieren

**Parameter in `mqtt_to_thrustmaster.py`:**

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| `STEER_GAIN` | 1.0 | Lenkungsverstärkung |
| `STEER_DEADZONE` | 0.01 | Totzone gegen Zittern, niedrig halten für frühe Korrekturen |
| `EMA_ALPHA` | 0.35 | Smoothing (niedriger = glatter, höher = direkter) |
| `CMD_TIMEOUT_S` | 0.25 | Safety-Timeout in Sekunden |
| `K_STANLEY` | 2.0 | Querfehler-Verstärkung im Lateralregler |
| `V_REF` | 8.0 | Virtuelle Geschwindigkeit; niedriger = stärkere Offset-Korrektur |
| `K_D_OFFSET` | 0.20 | Dämpfung/Frühreaktion über Offset-Änderungsrate |
| `LANE_ASSIST_TOGGLE_BUTTON` | 0 | SDL-Button-Index zum Ein-/Ausschalten; `-1` deaktiviert den Toggle |

---

## MQTT Topics Übersicht

```
control/steering_cmd      ← mqtt_to_thrustmaster.py publiziert Lenkdiagnostik
                            mqtt_to_xbox.py kann optional subscriben

sensor/lanestate          ← perception/run_perception.py publiziert Lane-Status
                            mqtt_to_thrustmaster.py subscribt
                            mqtt_broker.py aggregiert

sensor/objects            ← Mock Perception (test/demo)
sensor/vehicle_state      ← CSV-Replay (Offline-Analyse)

world/state               ← mqtt_broker.py publiziert aggregierten Zustand
```

---

## MQTT Bridge (optional - für Visualisierung)

Aggregiert alle Sensordaten und publiziert konsolidierten Weltzustand.

```bash
cd src/
uv run mqtt/mqtt_broker.py
```

Publiziert alle 100ms (`world/state`) mit: `objects`, `lanestate`, `steering_cmd`, `vehicle_state`, `last_update_ts`

---

## UI Monitor (optional - für Debugging)

Ein UI-Monitor ist konzeptionell vorgesehen, liegt in diesem Checkout aber nicht als startbares Skript vor. Wenn er wieder ergänzt wird, sollte er `world/state` oder direkt `sensor/lanestate` abonnieren.

---

## Vollständiger Start-Ablauf

```bash
# 1. MQTT Broker starten (empfohlen: Perception PC)
mosquitto -p 1883

# 2. Perception PC: Video-Quelle starten
cd src/
uv run video/send_video.py      # Falls Simulator kein NDI direkt sendet

# 3. Perception PC: Wahrnehmung starten
uv run perception/run_perception.py  # Lane Detection + sensor/lanestate Publishing

# 4. Simulator/Drive PC: MQTT → Thrustmaster starten
cd src/
uv run drive/mqtt_to_thrustmaster.py  # Berechnet Lenkung + bewegt Lenkrad

# Optional: Bridge auf einem beliebigen Rechner
cd src/
uv run mqtt/mqtt_broker.py
```
