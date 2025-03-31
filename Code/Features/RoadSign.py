from Features.Object import Object


class RoadSign(Object):
    
    SIGN_TYPES = {
        "STOP": [],
        "SPEED_LIMIT": ["speed_limit"],
        "YIELD": [],
        "SCHOOL_ZONE": ["speed_limit"],
        "CROSSWALK": [],
    }
    
    def __init__(self, bbox, center, confidence, pose=None, sign_type='STOP', attr=None):
        super().__init__(bbox, center, confidence, pose)
        
        self.type = sign_type        
        if self.type not in RoadSign.SIGN_TYPES:
            raise ValueError(f"Invalid sign type: {self.type}. Must be one of {RoadSign.SIGN_TYPES}")
        
        self.attr = dict.fromkeys(RoadSign.SIGN_TYPES[self.type], None)
        if isinstance(attr, dict) and all([a in attr for a in attr.keys()]):
            self.attr.update(attr)
        
    def __str__(self):
        return f"RoadSign(type={self.type}, pose={self.pose}, attr={self.attr})"
    
    def __repr__(self):
        return self.__str__()
    
    def get_attr(self, attr_name):
        return self.attr.get(attr_name, None)
    
    def set_attr(self, attr_name, value):
        if attr_name in self.attr:
            self.attr[attr_name] = value
        else:
            raise ValueError(f"Invalid attribute name: {attr_name}. Must be one of {self.attr.keys()}")
        
    def get_type(self):
        return self.type
        
    def get_attr_names(self):
        return list(self.attr.keys())
    
    def get_attr_values(self):
        return list(self.attr.values())
    
    def get_attr_dict(self):
        return self.attr.copy()
    
    def to_json(self, image_path=None):
        """
        Convert the RoadSign object to a JSON-compatible dictionary format.
        This can be used to serialize the object for saving to a JSON file.
        
        Returns:
            dict: A dictionary representation of the RoadSign object.
        """
        return {
            'name': 'RoadSign',  # Name of the object type
            'image_path': image_path,  # Optional: path to the image if needed for reference
            'type': self.type,  # Type of the road sign
            'attr': self.attr,  # Attribute of the sign type
            'object_data': super().to_json()  # Call the parent method to get bbox, center, confidence, and pose
        }

        