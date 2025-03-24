


class RoadSign:
    
    SIGN_TYPES = {
        "STOP": [],
        "SPEED_LIMIT": ["speed_limit"],
        "YIELD": [],
        "SCHOOL_ZONE": ["speed_limit"],
        "CROSSWALK": [],
    }
    
    def __init__(self, sign_type, idx=None, pose=None, attr=None):
        
        self.type = sign_type
        self.id = idx
        self.pose = pose
        
        if self.type not in RoadSign.SIGN_TYPES:
            raise ValueError(f"Invalid sign type: {self.type}. Must be one of {RoadSign.SIGN_TYPES}")
        
        self.attr = dict.fromkeys(RoadSign.SIGN_TYPES[self.type], None)
        if isinstance(attr, dict) and all([a in attr for a in attr.keys()]):
            self.attr.update(attr)
        
    def __str__(self):
        return f"RoadSign(type={self.type}, id={self.id}, pose={self.pose}, attr={self.attr})"
    
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
    
    def get_id(self):
        return self.id
    
    def get_pose(self):
        return self.pose
    
    def set_pose(self, pose):
        self.pose = pose
        
    def get_attr_names(self):
        return list(self.attr.keys())
    
    def get_attr_values(self):
        return list(self.attr.values())
    
    def get_attr_dict(self):
        return self.attr.copy()

        