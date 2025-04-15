import numpy as np

class Object:
    
    """
    This class represents a generic object from a YOLO output
    """

    def __init__(self, bbox, center, confidence, mask=None, pose=None):
        
        self.bbox = bbox # Bounding box in pixel coordinates [x1, y1, x2, y2]
        self.center = center  # Center of the bounding box in pixel coordinates
        self.mask = mask # Optional: segmentation mask (if available)
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
    
    def set_pose(self, pose):
        """
        Set the 3D world coordinates for the object
        This can be used to update the pose after the initial detection
        :param pose: 3D world coordinates (e.g., [x, y, z])
        """
        self.pose = pose
    
    def to_json(self):
        if self.pose is None:
            # If pose is None, we can still serialize the object without it
            print("Warning: pose is None, will not be included in JSON output. BLENDER RENDER WILL NOT WORK.")
            
        """
        Convert the object to a JSON-serializable dictionary
        This can be used to save the object data in JSON format
        """
        return {
            'bbox': self.bbox,
            'center': self.center,
            'confidence': self.confidence,
            'pose': self.pose
        }