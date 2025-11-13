import math
import time
import uuid

class Weltmodell:
    def __init__(self):
        # Ego vehicle state
        self.ego = {
            "position": (0.0, 0.0),
            "geschwindigkeit": 0.0,
            "richtung": 0.0,  # heading in degrees
            "lenkwinkel": 0.0
        }

        # Detected objects
        self.statische_objekte = {}   # id -> object
        self.dynamische_objekte = {}  # id -> object

        # Time tracking
        self.last_update_time = time.time()

    # ------------------------------------------------------------------
    # Update Ego Vehicle
    # ------------------------------------------------------------------
    def update_ego(self, position, geschwindigkeit, richtung, lenkwinkel):
        self.ego.update({
            "position": position,
            "geschwindigkeit": geschwindigkeit,
            "richtung": richtung,
            "lenkwinkel": lenkwinkel
        })

    # ------------------------------------------------------------------
    # Update from Perception (camera / sensors)
    # ------------------------------------------------------------------
    def update_from_detection(self, detected_objects):
        """
        Updates the world model with detected objects.
        Keeps previous IDs for tracking when possible.
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        new_dynamische = {}
        new_statische = {}

        for obj in detected_objects:
            obj_type = obj.get("typ", "Unbekannt")
            pos = obj.get("position", (0, 0))
            speed = obj.get("geschwindigkeit", 0.0)
            angle = obj.get("richtung", 0.0)

            # Compute distance from ego
            dist = self._calc_distance(self.ego["position"], pos)
            obj["entfernung_zum_ego"] = dist

            # Assign or update ID (simple rule: same type + close position)
            existing_id = self._find_existing_id(obj_type, pos)
            if existing_id:
                obj_id = existing_id
            else:
                obj_id = str(uuid.uuid4())  # create new unique ID

            # Store in correct list
            if obj_type in ["Ampel", "Schild", "Spur"]:
                new_statische[obj_id] = obj
            else:
                # update velocity and predict new position (optional)
                old_obj = self.dynamische_objekte.get(obj_id)
                if old_obj:
                    obj["vorherige_position"] = old_obj["position"]
                    obj["richtung"] = self._calc_direction(old_obj["position"], pos)
                    obj["geschwindigkeit"] = self._calc_speed(old_obj["position"], pos, dt)
                new_dynamische[obj_id] = obj

        self.statische_objekte = new_statische
        self.dynamische_objekte = new_dynamische

    # ------------------------------------------------------------------
    # Internal calculations
    # ------------------------------------------------------------------
    def _calc_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _calc_speed(self, p1, p2, dt):
        if dt <= 0:
            return 0.0
        return self._calc_distance(p1, p2) / dt

    def _calc_direction(self, p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360

    def _find_existing_id(self, obj_type, pos, max_distance=5.0):
        """
        Tries to match a new object with an old one based on proximity and type.
        """
        for obj_id, obj in self.dynamische_objekte.items():
            if obj["typ"] == obj_type:
                if self._calc_distance(obj["position"], pos) < max_distance:
                    return obj_id
        return None

    # ------------------------------------------------------------------
    # Export & Visualization
    # ------------------------------------------------------------------
    def get_summary(self):
        summary = {
            "ego": self.ego,
            "statische_objekte": list(self.statische_objekte.values()),
            "dynamische_objekte": list(self.dynamische_objekte.values())
        }
        return summary

    def print_state(self):
        print("\n=== 🌍 Weltmodell-Status ===")
        print(f"Ego-Fahrzeug: {self.ego}")
        print(f"Statische Objekte: {len(self.statische_objekte)}")
        for o in self.statische_objekte.values():
            print(f"  - {o['typ']} @ {o['position']}")
        print(f"Dynamische Objekte: {len(self.dynamische_objekte)}")
        for o in self.dynamische_objekte.values():
            print(f"  - {o['typ']} @ {o['position']} (Entfernung: {o['entfernung_zum_ego']:.1f} m)")
        print("=============================")
