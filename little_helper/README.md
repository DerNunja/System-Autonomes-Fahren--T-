## Tools zur Datenvorverarbeitung (`little_helpers`)

### `create_train_data.py`

* Liest **Videodateien** und die zugehörigen **TabData-CSV-Dateien** aus dem Simulator.
* Synchronisiert beide Quellen und erzeugt:

  * Einzelbilder (`frames/`)
  * Labeldatei (`labels_to_frames.csv`)
* Parameter wie `trim_video_start_sec` oder `sample_stride` können angepasst werden, um Startzeit und Samplingrate zu steuern.

### `crop_finder.py`

* Hilft beim **Bestimmen des passenden Bildausschnitts (Cropping)** für die Trainingsdaten.
* Die Werte des gefunden Crops können ausgeben werden und in `create_train_data.py` verwendet werden
* Praktisch, um irrelevante Teile (z. B. Cockpit oder Himmel) zu entfernen.
![Verwendung von crop_finder.py](Pictures/image.png)

### `test_labels.py`

* Visualisiert **Lenkwinkel**, **Throttle** und **Brake** über dem Originalvideo.
* Nutzt das von `create_train_data.py` erzeugte `.csv`.
* Wichtig:

  * `trim_video_start_sec = 0.0`
  * `sample_stride = 1`
    Damit die Label-Zeiten exakt mit den Videoframes übereinstimmen.
![alt text](Pictures/image-1.png)
![alt text](Pictures/image-2.png)
---

### `cuda_test.py`
* Testet ob die GPU verfügbar ist und ob die Berechnungen somit auf dieser möglich sind
* Die GPU ist essentiell für die Echtzeit Anwendung des autonomen Fahrsystems
Erwarteter Output: 
```bash
torch: 2.4.0+cu121
torchvision: 0.19.0+cu121
CUDA available: True
Device: NVIDIA GeForce RTX 4080 SUPER
```