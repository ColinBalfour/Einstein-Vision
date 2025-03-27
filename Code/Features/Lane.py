

class Lane:
    
    LANE_TYPES = [
        "divider-line",
        "dotted-line",
        "double-line",
        "random-line",
        "road-sign-line",
        "solid-line"
    ]
    
    def __init__(self, keypoints, lane_type, world_coords=None):
        
        self.keypoints = keypoints # pixel coordinates
        self.world_coords = world_coords # 3d world coordinates
        
        if lane_type not in Lane.LANE_TYPES:
            raise ValueError(f"Invalid lane type: {lane_type}. Must be one of {Lane.LANE_TYPES}")
        
        self.lane_type = lane_type

    def __repr__(self):
        return f"Lane(type={self.lane_type}, keypoints={self.keypoints}, world_coords={self.world_coords})"
    
    def __str__(self):
        return self.__repr__()
    
    def get_keypoints(self):
        return self.keypoints
    
    def get_lane_type(self):
        return self.lane_type
    
    def get_lane_color(self):
        return self.lane_color
    
    def get_world_coords(self):
        if self.world_coords is None:
            print("Warning: world coordinates are not set.")
        return self.world_coords
    
    