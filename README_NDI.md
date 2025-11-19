# NDI Receiver – Lokaler Video Stream für Perception Modelle

Dieses Modul (`ndi_receiver.py`) dient dazu, lokale Videodaten über **NDI** zu streamen und anschließend an die Perception-Modelle (Detection, Segmentierung usw.) weiterzugeben.

##  Voraussetzungen

### 1. NDI Tools installieren
Download über die offizielle NDI-Seite.  
In unserem Projekt wird **NDI Screen Capture** benutzt.

### 2. Virtuelle Umgebung (venv) benutzen



---

## ▶ Nutzung des Receivers

### 1. Video starten
Ein beliebiges Video (mp4, avi, etc.) mit VLC oder einem anderen Player öffnen.

### 2. NDI Screen Capture öffnen
- Das Video-Fenster oder den Monitor auswählen
- NDI sendet nun automatisch einen Stream im lokalen System

### 3. Receiver Code starten


### 4. Ausgabe
Zwei Windows erscheinen:

- **NDI Original** → Original-Video über NDI
- **Perception Result** → Ausgabe unseres Dummy-Models (aktuell Graustufen)


