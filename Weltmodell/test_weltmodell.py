# test_weltmodell.py
from Weltmodell import Weltmodell

def test_weltmodell():
    wm = Weltmodell()

    # Mock ego state
    wm.ego['position'] = (0, 0)
    wm.ego['geschwindigkeit'] = 50
    wm.ego['richtung'] = 90

    # Mock lane info
    wm.lane_info['curvature_m'] = 1000
    wm.lane_info['lateral_offset_m'] = 0.1
    wm.lane_info['confidence'] = 0.9

    # Mock detections
    mock_detections = [
        {'box':[100, 150, 140, 190], 'score':0.95, 'class_name':'speed_sign'},
        {'box':[200, 100, 220, 130], 'score':0.85, 'class_name':'traffic_light_red'},
        {'box':[250, 120, 270, 150], 'score':0.9, 'class_name':'traffic_light_green'}
    ]

    wm.update_from_vision(mock_detections)
    wm.print_state()

if __name__ == "__main__":
    test_weltmodell()
