

class Object:
    
    """
    This class represents a generic object from a YOLO output
    """

    def __init__(self, bbox, center, confidence, pose=None):
        
        self.bbox = bbox # Bounding box in pixel coordinates [x1, y1, x2, y2]
        self.center = center  # Center of the bounding box in pixel coordinates
        self.confidence = confidence # Confidence score of the detection
        self.pose = pose # 3D world coordinates (if available). This cis usually None at init and set later
    
    def __repr__(self):
        return f"Object(bbox={self.bbox}, center={self.center}, confidence={self.confidence}, pose={self.pose})"
        
    def __str__(self):
        return self.__repr__()
        
    def get_bbox_keypoints(self):
        return self.bbox
    
    def get_center(self):
        # Returns the center of the bounding box in pixel coordinates
        return self.center
    
    def get_confidence(self):
        # Returns the confidence score of the detection
        # This is a float value indicating how confident the model is about this detection
        return self.confidence
    
    def get_pose(self):
        if self.pose is None:
            print("Warning: pose is not set.")
        return self.pose