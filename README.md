# System Autonomes Fahren (T) – S.A.F.T

Projekt zur Entwicklung eines autonomen Fahrsystems mithilfe des Fahrsimulators der Hochschule Harz.
*(Das „T“ steht aktuell noch offen – Vorschläge sind willkommen!)*

---

## Projektüberblick

Dieses Projekt zielt darauf ab, ein System für **autonomes Fahren im Simulator** zu entwickeln.
Dazu werden Sensordaten (z. B. Kameraaufnahmen und Steuerbefehle) aus dem **Fahrsimulator der Hochschule Harz** verarbeitet und für Trainingszwecke eines neuronalen Netzes aufbereitet.

---

## Tools zur Datenvorverarbeitung (`little_helpers`)

---

## Setup & Nutzung

```bash
# Repository klonen
git clone https://github.com/DerNunja/System-Autonomes-Fahren--T-.git
cd System-Autonomes-Fahren--T-

# Virtuelle Umgebung anlegen
python3 -m venv .venv
source .venv/bin/activate

# Abhängigkeiten installieren (sofern requirements.txt existiert)
pip install -r requirements.txt
```

---

## Geplante Erweiterungen

* Integration eines neuronalen Fahrmodells (Behavioural Cloning)
* Online-Synchronisierung mit dem Weltmodell
* Segmentierungsnetzwerke (UFLD / SegFormer)
* MQTT-basierte Kommunikation zwischen Sensorik und Entscheidungslogik
* Web-Interface zur Visualisierung

---

## Mitwirken

Ideen, Verbesserungen oder Namensvorschläge für das **„T“** sind ausdrücklich willkommen!
Erstelle einfach ein Issue oder öffne einen Pull Request.

---

## Resources 

Ultra-Fast-Lane-Detection-v2 pyTorch Implementation (verwendet in ../LaneDetection)
https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2?tab=readme-ov-file
