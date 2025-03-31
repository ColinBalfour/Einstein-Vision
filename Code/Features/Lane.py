import numpy as np

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
    
    def get_world_coords(self):
        if self.world_coords is None:
            print("Warning: world coordinates are not set.")
        return self.world_coords
    
    def set_world_coords(self, world_coords):
        """
        Set the world coordinates for the lane.
        
        Args:
            world_coords (np.ndarray): The 3D world coordinates for the lane.
        """
        if not isinstance(world_coords, (list, np.ndarray)):
            raise ValueError("world_coords must be a list, or numpy array.")
        
        self.world_coords = world_coords

    def to_json(self, image_path=None):
        """
        Convert the Lane object to a JSON-compatible dictionary format.
        This can be used to serialize the object for saving to a JSON file.
        
        Returns:
            dict: A dictionary representation of the Lane object.
        """
        return {
            'name': 'Lane',
            'image_path': image_path,  # Optional: path to the image if needed for reference
            'lane_type': self.lane_type,
            'keypoints': self.keypoints.tolist() if isinstance(self.keypoints, np.ndarray) else self.keypoints,
            'world_coords': self.world_coords.tolist() if isinstance(self.world_coords, np.ndarray) else self.world_coords
        }
    
    