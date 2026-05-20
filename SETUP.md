# Setup - Autonomes Fahren System

## Architektur

Zwei Rechner kommunizieren über NDI (Video) und MQTT (Steuerung):

```
[Simulator PC]                          [Perception PC]
┌──────────────────┐                    ┌─────────────────────────────┐
│ Fahrsimulator     │                    │ sender_video.py (NDI)       │
│                   │  NDI Video ──────► │ receiver_video.py           │
│ mqtt_to_xbox.py   │◄── MQTT ───────── │  ├─ Lane Detection (UFLD v2)│
│ (Xbox-Emulation)  │   steering_cmd     │  ├─ World Model             │
└──────────────────┘                    │  ├─ Stanley Controller        │
                                        │  └─ MQTT Publisher            │
                                        └─────────────────────────────┘
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

### 2. Hauptskript - Lane Detection + Steuerung

Empfängt NDI-Video, erkennt Fahrstreifen, berechnet Lenkbefehle, publiziert über MQTT.

```bash
cd Behavioural_Cloning_Basic/ndi_tools/
uv run receiver_video.py
```

**Was es macht:**
- Verbindet sich zum NDI-Stream "Demo"
- Lädt UFLD v2 Lane Detection Modell (ResNet34, ~826MB)
- Verarbeitet jeden Frame: Lane Detection → Weltmodell → Stanley Controller
- Publiziert Lenkbefehle auf `control/steering_cmd`
- Publiziert Lane-Status auf `sensor/lanestate`
- Zeigt zwei OpenCV-Fenster: Original-Video und annotiertes Video

**MQTT Topics die publiziert werden:**

| Topic | Inhalt |
|-------|--------|
| `control/steering_cmd` | `{t_ms, steer_rad, steer_norm, ff_term, offset_m, heading_err_rad, curvature}` |
| `sensor/lanestate` | `{lane_center, curvature}` |

---

## Skripte auf dem Simulator PC

### 3. MQTT → Xbox Controller Bridge

Empfängt Lenkbefehle vom Perception PC und simuliert ein Xbox-Gamepad.

```bash
cd Behavioural_Cloning_Basic/drive/
uv run mqtt_to_xbox.py
```

**Wichtig:** `BROKER` in `mqtt_to_xbox.py` auf die IP des MQTT Brokers setzen (nicht leer lassen für Remote-Broker).

**Was es macht:**
- Subscribt auf `control/steering_cmd`
- Wendet an: Gain → Deadzone (0.03) → EMA-Smoothing (alpha=0.25)
- Sendet an virtuellen Xbox 360 Controller (Left Stick X)
- Safety-Timeout: bei 0.25s keine Befehle → Lenkrad zentrieren

**Parameter in `mqtt_to_xbox.py`:**

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| `STEER_GAIN` | 1.0 | Lenkungsverstärkung |
| `STEER_DEADZONE` | 0.03 | Totzone gegen Zittern |
| `EMA_ALPHA` | 0.25 | Smoothing (niedriger = glatter) |
| `CMD_TIMEOUT_S` | 0.25 | Safety-Timeout in Sekunden |

---

## MQTT Topics Übersicht

```
control/steering_cmd      ← receiver_video.py publiziert Lenkbefehle
                            mqtt_to_xbox.py subscribt

sensor/lanestate          ← receiver_video.py publiziert Lane-Status
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

Publiziert alle 100ms (`world/state`) mit: `objects`, `lanestate`, `vehicle_state`, `last_update_ts`

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
# 1. MQTT Broker starten (auf beiden Rechnern oder zentral)
mosquitto -p 1883

# 2. Perception PC: Video-Quelle starten
cd Behavioural_Cloning_Basic/ndi_tools/
uv run sender_video.py          # Falls Simulator kein NDI direkt sendet

# 3. Perception PC: Hauptsteuerung starten
uv run receiver_video.py        # Lane Detection + MQTT Publishing

# 4. Simulator PC: MQTT → Xbox Bridge starten
cd Behavioural_Cloning_Basic/drive/
uv run mqtt_to_xbox.py          # Konvertiert MQTT → Gamepad-Input

# Optional: Bridge + UI auf einem beliebigen Rechner
cd Behavioural_Cloning_Basic/mqtt/
uv run mqtt_broker.py
uv run ui_world.py
```
