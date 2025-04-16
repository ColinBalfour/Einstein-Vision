#!/usr/bin/env python3
# Taillight.py - Class for taillight and brakelight detection using DETIC

import os
import json
import cv2
import numpy as np
import sys

from Features.Object import Object

class Taillight(Object):
    """
    Class to represent a taillight or brakelight in a vehicle
    """

    def __init__(self, bbox, center, confidence, pose=None, direction=None, light_type=None):
        """
        Initialize a taillight instance

        Args:
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            center (tuple): Center coordinates (x, y)
            confidence (float): Confidence score of the detection
            pose (list, optional): 3D world coordinates [x, y, z]
            is_brake (bool): True if the taillight is a brakelight
        """
        super().__init__(bbox, center, confidence, pose)

        if direction not in ['left', 'right', None]:
            raise ValueError("Direction must be 'left' or 'right'")
        self.direction = direction
        
        if light_type not in ['brake_light', 'turn_signal', None]:
            raise ValueError("Type must be 'brake' or 'turn_signal'")
        
        self.light_type = light_type
        
    def is_brake(self):
        """
        Check if the taillight is a brakelight
        """
        return self.light_type == 'brake_light'

    def is_blink(self):
        """
        Check if the taillight is a blinker
        """
        return self.light_type == 'turn_signal'
    
    def get_direction(self):
        """
        Get the direction of the taillight
        """
        return self.direction

    def to_json(self, image_path=None):
        """
        Convert the TrafficLight object to a JSON-compatible dictionary format.
        This can be used to serialize the object for saving to a JSON file.
        """
        return {
            'name': 'Taillight',  # Name of the object type
            'image_path': image_path,  # Optional: path to the image if needed for reference
            'direction': self.direction,  # Direction of the taillight (left or right side of car)
            'type': self.light_type,  # Type of the taillight (brake or turn)
            'object_data': super().to_json()
        }



if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Taillight Detection")
    parser.add_argument("--input", required=True, help="Path to input image or directory")
    parser.add_argument("--output", default="taillight_detections.json", help="Path to output JSON file")
    parser.add_argument("--visualize", action="store_true", help="Visualize detections")
    parser.add_argument("--output-dir", default="output", help="Directory to save visualization results")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    # Initialize detector
    detector = TaillightDetector(confidence_threshold=args.confidence, use_gpu=not args.cpu)

    # Create output directory if it doesn't exist
    if args.visualize and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Process input
    if os.path.isfile(args.input):
        # Process single image
        image_path = args.input
        if args.visualize:
            output_path = os.path.join(args.output_dir, os.path.basename(image_path))
            taillights, _ = detector.detect_and_visualize(image_path, output_path)
        else:
            taillights = detector.detect(image_path)

        # Export to JSON
        detector.export_to_json(taillights, args.output, image_path)
    elif os.path.isdir(args.input):
        # Process directory of images
        all_taillights = {}
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(args.input, filename)
                print(f"Processing {image_path}")

                if args.visualize:
                    output_path = os.path.join(args.output_dir, filename)
                    taillights, _ = detector.detect_and_visualize(image_path, output_path)
                else:
                    taillights = detector.detect(image_path)

                all_taillights[image_path] = [t.to_dict() for t in taillights]

        # Export all results to JSON
        with open(args.output, 'w') as f:
            json.dump(all_taillights, f, indent=4)

        print(f"All results exported to {args.output}")
    else:
        print(f"Input path {args.input} does not exist")
