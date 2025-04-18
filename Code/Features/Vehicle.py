import torch
import numpy as np
import cv2
import os
import json
import sys
import depth_pro
from ultralytics import YOLO

from Features.Object import Object


class Vehicle(Object):
    
    VEHICLE_TYPES = [
        "car",
        "SUV",
        "truck",
        "bicycle",
        "pickup",
        "motorcycle",
    ]
    
    def __init__(self, bbox, center, confidence, mask=None, pose=None, vehicle_type='car', left_taillight=None, right_taillight=None, moving=False):
        super().__init__(bbox, center, confidence, mask, pose) # Call the parent constructor to initialize bbox, center, mask, confidence, and pose
        
        if vehicle_type not in Vehicle.VEHICLE_TYPES:
            raise ValueError(f"Invalid vehicle type: {vehicle_type}. Must be one of {Vehicle.VEHICLE_TYPES}")
        
        self.vehicle_type = vehicle_type
        self.left_taillight = left_taillight  # Optional: left taillight object
        self.right_taillight = right_taillight  # Optional: right taillight object
        self.moving = moving  # Optional: flag to indicate if the vehicle is moving
        
    def __repr__(self):
        # Custom string representation of the Vehicle object
        return f"Vehicle(type={self.vehicle_type}, bbox={self.bbox}, center={self.center}, confidence={self.confidence}, pose={self.pose})"
        
    def __str__(self):
        return self.__repr__()
    
    def get_vehicle_type(self):
        return self.vehicle_type
    
    def to_json(self, image_path=None):
        """
        Convert the Vehicle object to a JSON-compatible dictionary format.
        This can be used to serialize the object for saving to a JSON file.
        """
        return {
            'name': 'Vehicle',  # Name of the object type
            'image_path': image_path,  # Optional: path to the image if needed for reference
            'vehicle_type': self.vehicle_type,  # Include the vehicle type
            'left_taillight': self.left_taillight.to_json() if self.left_taillight else None,
            'right_taillight': self.right_taillight.to_json() if self.right_taillight else None,
            'moving': self.moving,  # Include the moving status
            'object_data': super().to_json()
        }


def main():
    # Base directories
    base_data_dir = 'Code/P3Data/ExtractedFrames/Undist'
    base_output_dir = 'Code/outputs'

    # Define scenes to process - now 13 scenes
    # scenes = ['scene_1', 'scene_2', 'scene_3', 'scene_4', 'scene_5',
    #           'scene_6', 'scene_7', 'scene_8', 'scene_9', 'scene_10',
    #           'scene_11', 'scene_12', 'scene_13']

    # Set confidence threshold
    conf_threshold = 0.65
    scene = 'scene_1'

    # Initialize models
    # model, depth_model = initialize_models()

    # Process each scene
    # for scene in scenes:
    data_dir = os.path.join(base_data_dir, scene)
    output_dir = os.path.join(base_output_dir, scene)
    os.makedirs(output_dir, exist_ok=True)

    # process_scene(scene, data_dir, output_dir, model, depth_model, conf_threshold)




if __name__ == "__main__":
    main()
