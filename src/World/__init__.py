class WorldModel:
    """
    Einfaches Weltmodell, das nur LaneDetection in 2D (Bildpixel) verwaltet.
    """

    def __init__(self, img_width: int, img_height: int, lane_width_m: float = 3.7):
        self.img_width = img_width
        self.img_height = img_height
        self.lane_width_m = lane_width_m      # <<< HIER: wichtig für px->m
        self.state: Optional[WorldModelState] = None