# NDI Video Streaming – Sender & Receiver (Lokale Simulation)

Dieses Modul ermöglicht es, ein lokales Video über **NDI (Network Device Interface)** zu streamen und anschließend wieder zu empfangen.  
Der Stream wird direkt an das Perception-Modell weitergeleitet und dort verarbeitet.  
MQTT wird hier **nicht** für den Videostream genutzt, da MQTT keine Echtzeitbilddaten effizient übertragen kann.

---

# Voraussetzungen

### Benötigte Software
- Python **3.11**  
- **NDI Tools** (für die lokale NDI-Engine)  
- Python-Pakete:  
  - `visiongraph-ndi`  
  - `opencv-python` (cv2)

---

### Hinweis zur Ausführung
1. **Zuerst** den NDI-Sender starten  
2. **Dann** den NDI-Empfänger starten  

Das gesamte System läuft **lokal auf einem einzigen Rechner**.  
Ein Netzwerk ist nicht notwendig.

---
