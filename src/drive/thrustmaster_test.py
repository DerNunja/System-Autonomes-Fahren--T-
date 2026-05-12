"""
============================================================
  THRUSTMASTER LENKRAD – TEST SKRIPT
  Erkennung + Kurze Links/Rechts Bewegung via Force Feedback
============================================================

──────────────────────────────────────────────────────────
  VORAUSSETZUNGEN (vor dem ersten Start unbedingt erledigen)
──────────────────────────────────────────────────────────

  1. THRUSTMASTER TREIBER installieren
     → https://www.thrustmaster.com/de/support/
     → Passendes Modell wählen (z.B. T300RS, TX, T150 ...)
     → "Driver" herunterladen und installieren
     → PC neu starten!
     → NICHT nur Plug & Play verwenden – der offizielle
       Treiber ist für Force Feedback zwingend erforderlich.

  2. LENKRAD KALIBRIEREN (Windows)
     → Windows-Taste → "Gamecontroller" oder
       "joy.cpl" in die Ausführen-Box (Win+R) eingeben
     → Thrustmaster auswählen → Eigenschaften → Testen
     → Prüfen ob Achsen und Buttons erkannt werden
     → Falls nicht: Treiber neu installieren

  3. AUTOCENTER DEAKTIVIEREN (sehr wichtig!)
     → Thrustmaster Control Panel öffnen
       (wird mit dem Treiber installiert)
     → "Centering Spring" / "Rückstellfeder" auf 0% setzen
     → Sonst kämpft das Lenkrad gegen unsere Befehle!

  4. PYTHON PAKETE installieren
     → Kommandozeile (cmd) als Administrator öffnen:

       pip install pysdl2 pysdl2-dll

     → pysdl2-dll liefert die SDL2.dll automatisch mit,
       kein manueller DLL-Download nötig.

  5. SICHERHEITSHINWEIS
     → Das Lenkrad wird sich physisch bewegen!
     → Hände beim ersten Test NICHT am Lenkrad lassen
     → MAX_TORQUE ist absichtlich niedrig gesetzt (30%)
     → Strg+C stoppt das Skript jederzeit sofort

──────────────────────────────────────────────────────────
"""

import sys
import time
import ctypes

# ── Imports mit verständlicher Fehlermeldung ───────────────────────────────
try:
    import sdl2
    import sdl2.ext
except ImportError:
    print("FEHLER: pysdl2 nicht gefunden!")
    print("Bitte ausführen: pip install pysdl2 pysdl2-dll")
    sys.exit(1)


# ── Konfiguration ──────────────────────────────────────────────────────────

JOYSTICK_INDEX = 0      # Index des Lenkrads (0 = erstes Gerät)
                        # Falls falsch: Skript zeigt alle Geräte an

MAX_TORQUE     = 0.30   # Maximales Drehmoment (0.0 - 1.0)
                        # 0.30 = 30% – für den ersten Test bewusst niedrig!

MOVE_TORQUE    = 0.25   # Kraft für die Testbewegung (≤ MAX_TORQUE)
MOVE_DURATION  = 1.5    # Sekunden pro Richtung (links / rechts)
PAUSE_DURATION = 0.5    # Pause zwischen den Bewegungen

EFFECT_REFRESH_MS = 30  # Effekt alle 30ms neu senden (~33 Hz)
                        # Niedrigerer Wert = flüssiger, aber mehr CPU


# ── Hilfsfunktionen ────────────────────────────────────────────────────────

def find_wheel():
    """
    Sucht nach dem ersten Joystick/Lenkrad mit Force-Feedback-Unterstützung.
    Gibt (joystick, haptic, name) zurück oder beendet das Programm.
    """
    sdl2.SDL_Init(sdl2.SDL_INIT_JOYSTICK | sdl2.SDL_INIT_HAPTIC)

    num_joysticks = sdl2.SDL_NumJoysticks()
    print(f"\n{'='*55}")
    print(f"  Erkannte Eingabegeräte: {num_joysticks}")
    print(f"{'='*55}")

    if num_joysticks == 0:
        print("\nFEHLER: Kein Joystick / Lenkrad gefunden!")
        print("  → Lenkrad angeschlossen und Treiber installiert?")
        print("  → Prüfen mit: Win+R → joy.cpl")
        sdl2.SDL_Quit()
        sys.exit(1)

    # Alle Geräte auflisten
    for i in range(num_joysticks):
        name_bytes = sdl2.SDL_JoystickNameForIndex(i)
        name = name_bytes.decode("utf-8", errors="replace") if name_bytes else "Unbekannt"
        print(f"  [{i}] {name}")

    print(f"{'='*55}")

    # Zielgerät öffnen
    joystick = sdl2.SDL_JoystickOpen(JOYSTICK_INDEX)
    if not joystick:
        print(f"\nFEHLER: Gerät [{JOYSTICK_INDEX}] konnte nicht geöffnet werden.")
        print(f"  → JOYSTICK_INDEX am Anfang des Skripts anpassen (0 bis {num_joysticks - 1})")
        sdl2.SDL_Quit()
        sys.exit(1)

    name_bytes = sdl2.SDL_JoystickName(joystick)
    name = name_bytes.decode("utf-8", errors="replace") if name_bytes else "Unbekannt"
    axes = sdl2.SDL_JoystickNumAxes(joystick)
    buttons = sdl2.SDL_JoystickNumButtons(joystick)

    print(f"\n  Verwende Gerät [{JOYSTICK_INDEX}]: {name}")
    print(f"  Achsen: {axes}  |  Buttons: {buttons}")

    # Force Feedback prüfen
    haptic = sdl2.SDL_HapticOpenFromJoystick(joystick)
    if not haptic:
        print(f"\nFEHLER: Gerät '{name}' unterstützt kein Force Feedback!")
        print("  → Thrustmaster-Treiber korrekt installiert?")
        print("  → Anderes Gerät über JOYSTICK_INDEX versuchen")
        sdl2.SDL_JoystickClose(joystick)
        sdl2.SDL_Quit()
        sys.exit(1)

    # Welche FF-Effekte werden unterstützt?
    caps = sdl2.SDL_HapticQuery(haptic)
    print(f"\n  Force Feedback Fähigkeiten:")
    print(f"    Constant Force : {'✓' if caps & sdl2.SDL_HAPTIC_CONSTANT  else '✗'}")
    print(f"    Spring         : {'✓' if caps & sdl2.SDL_HAPTIC_SPRING    else '✗'}")
    print(f"    Damper         : {'✓' if caps & sdl2.SDL_HAPTIC_DAMPER    else '✗'}")
    print(f"    Sine           : {'✓' if caps & sdl2.SDL_HAPTIC_SINE      else '✗'}")

    if not (caps & sdl2.SDL_HAPTIC_CONSTANT):
        print("\nWARNUNG: Constant Force wird nicht unterstützt!")
        print("  → Lenkrad lässt sich möglicherweise nicht direkt ansteuern.")

    # Autocenter über SDL deaktivieren
    result = sdl2.SDL_HapticSetAutocenter(haptic, 0)
    if result == 0:
        print(f"\n  Autocenter via SDL deaktiviert ✓")
    else:
        print(f"\n  HINWEIS: Autocenter konnte nicht via SDL deaktiviert werden.")
        print(f"  → Bitte im Thrustmaster Control Panel manuell auf 0% setzen!")

    print(f"{'='*55}\n")
    return joystick, haptic, name


def set_torque(haptic, torque: float, duration_ms: int):
    """
    Sendet einen Constant-Force-Effekt ans Lenkrad.

    torque      : -1.0 (volle Kraft links) bis +1.0 (volle Kraft rechts)
    duration_ms : Wie lange der Effekt aktiv bleibt (in Millisekunden)

    Gibt die effect_id zurück (zum späteren Löschen).
    """
    # Sicherheitsclip
    torque = max(-MAX_TORQUE, min(MAX_TORQUE, torque))

    level = int(torque * 32767)  # SDL erwartet -32767 bis 32767

    effect = sdl2.SDL_HapticEffect()
    effect.type = sdl2.SDL_HAPTIC_CONSTANT

    effect.constant.direction.type    = sdl2.SDL_HAPTIC_CARTESIAN
    effect.constant.direction.dir[0]  = 1   # X-Achse (Lenkachse)
    effect.constant.direction.dir[1]  = 0
    effect.constant.direction.dir[2]  = 0

    effect.constant.length       = duration_ms
    effect.constant.delay        = 0
    effect.constant.button       = 0
    effect.constant.interval     = 0
    effect.constant.level        = level
    effect.constant.attack_length = 0
    effect.constant.attack_level  = abs(level)
    effect.constant.fade_length   = 0
    effect.constant.fade_level    = abs(level)

    effect_id = sdl2.SDL_HapticNewEffect(haptic, ctypes.byref(effect))
    if effect_id < 0:
        err = sdl2.SDL_GetError()
        print(f"  WARNUNG: Effekt konnte nicht erstellt werden: {err}")
        return -1

    sdl2.SDL_HapticRunEffect(haptic, effect_id, 1)
    return effect_id


def stop_force(haptic, effect_id: int):
    """Stoppt und löscht einen aktiven FF-Effekt."""
    if effect_id >= 0:
        sdl2.SDL_HapticStopEffect(haptic, effect_id)
        sdl2.SDL_HapticDestroyEffect(haptic, effect_id)


def get_steering_angle(joystick) -> float:
    """Liest den aktuellen Lenkwinkel: -1.0 (links) bis +1.0 (rechts)"""
    sdl2.SDL_JoystickUpdate()
    raw = sdl2.SDL_JoystickGetAxis(joystick, 0)  # Achse 0 = Lenkachse
    return raw / 32767.0


# ── Testbewegung ───────────────────────────────────────────────────────────

def run_test(joystick, haptic):
    """
    Führt eine langsame Links-Rechts Testbewegung durch.
    Jede Richtung hält MOVE_DURATION Sekunden an.
    """
    effect_id = -1

    def move(direction_label: str, torque: float, duration: float):
        nonlocal effect_id
        print(f"  → {direction_label} (Kraft: {torque:+.0%}, Dauer: {duration:.1f}s)")

        start = time.time()
        while time.time() - start < duration:
            stop_force(haptic, effect_id)
            effect_id = set_torque(haptic, torque, EFFECT_REFRESH_MS * 2)

            angle = get_steering_angle(joystick)
            elapsed = time.time() - start
            bar_pos = int((angle + 1) / 2 * 40)
            bar = "─" * bar_pos + "●" + "─" * (40 - bar_pos)
            print(f"\r     [{bar}]  {angle:+.3f}  ({elapsed:.1f}s) ", end="", flush=True)

            time.sleep(EFFECT_REFRESH_MS / 1000.0)

        print()  # Zeilenumbruch nach der Fortschrittsanzeige
        stop_force(haptic, effect_id)
        effect_id = -1

    print("  Testbewegung startet in 3 Sekunden ...")
    print("  HÄNDE VOM LENKRAD! Strg+C zum Abbrechen.\n")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    try:
        print("\n  ── Phase 1: Mitte → Links ──")
        move("LINKS ", -MOVE_TORQUE, MOVE_DURATION)

        print(f"\n  ── Pause {PAUSE_DURATION:.1f}s ──")
        time.sleep(PAUSE_DURATION)

        print("\n  ── Phase 2: Links → Rechts ──")
        move("RECHTS", +MOVE_TORQUE, MOVE_DURATION)

        print(f"\n  ── Pause {PAUSE_DURATION:.1f}s ──")
        time.sleep(PAUSE_DURATION)

        print("\n  ── Phase 3: Zurück zur Mitte ──")
        move("MITTE ", 0.0, 0.5)  # Kurz Nullkraft → Rad bleibt wo es ist

        print("\n  ✓ Testbewegung abgeschlossen!")

    except KeyboardInterrupt:
        print("\n\n  ⚠ Abgebrochen! Stoppe alle Effekte ...")
        stop_force(haptic, effect_id)

    finally:
        # Sicherheitshalber alle Effekte stoppen
        sdl2.SDL_HapticStopAll(haptic)


# ── Einstiegspunkt ─────────────────────────────────────────────────────────

def main():
    print(__doc__)
    print("Starte Gerätesuche ...\n")

    joystick, haptic, name = find_wheel()

    print(f"Lenkrad bereit: {name}")
    print(f"Einstellungen : Kraft={MOVE_TORQUE:.0%}, Dauer={MOVE_DURATION:.1f}s\n")

    run_test(joystick, haptic)

    # Aufräumen
    sdl2.SDL_HapticClose(haptic)
    sdl2.SDL_JoystickClose(joystick)
    sdl2.SDL_Quit()
    print("\nProgramm beendet.")


if __name__ == "__main__":
    main()