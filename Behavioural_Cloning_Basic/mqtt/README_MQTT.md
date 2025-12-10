#  Projekt: MQTT Bridge, Perception Client und UI

Dieses Projekt demonstriert einen **kompletten Datenfluss** von einem simulierten **Perception-Modul** bis hin zu einer **grafischen Benutzeroberfläche (UI)** mithilfe des **MQTT-Protokolls**. Es dient als Beispiel für eine IoT- oder Robotik-Anwendung, bei der Sensordaten (simuliert als "Perception") über einen Broker an einen Client (die UI) verteilt werden.

---

##  Systemarchitektur

Das System besteht aus vier Hauptkomponenten:

1.  **Mosquitto Broker:** Der zentrale Nachrichtenverteiler.
2.  **MQTT-Bridge (Python Broker File):** Ein Python-Skript, das als Brücke oder **zentraler Client** fungiert, um möglicherweise Daten zu verarbeiten oder weiterzuleiten.
3.  **Perception Client:** Simuliert ein Wahrnehmungsmodul, das regelmäßig Daten über MQTT veröffentlicht.
4.  **UI Client:** Eine **Tkinter-basierte UI**, die die über MQTT empfangenen Daten darstellt.

---

##  Voraussetzungen

### Software-Abhängigkeiten

* **Mosquitto MQTT Broker:** Muss installiert und ausgeführt werden (z.B. über das Betriebssystem oder Docker).

### Python-Abhängigkeiten

Installieren Sie die notwendigen Python-Pakete:

```bash
pip install paho-mqtt
pip install tk

