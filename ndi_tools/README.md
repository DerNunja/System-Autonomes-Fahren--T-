# 🖥️ NDI Videoübertragung (Sender & Empfänger)

Dieses Tool sendet und empfängt Videostreams über das NDI-Protokoll (Network Device Interface).

**Hinweis:**
- Die Simulation erzeugt nur Videodaten, keine Sensordaten.  
- MQTT kann die Videoframes nicht an Perception-Module (z. B. YOLO oder Segmentierung) übertragen.

---

## ⚙️ Voraussetzungen

1. **NDI SDK / Runtime installieren**  
   👉 [https://ndi.video/ndi-sdk/](https://ndi.video/ndi-sdk/)  
   Stelle sicher, dass die Datei  
   `C:\Program Files\NDI\NDI 6 SDK\Bin\x64\Processing.NDI.Lib.x64.dll`  
   existiert.

2. **Python-Abhängigkeiten installieren**
   ```bash
   pip install -r requirements_ndi.txt
