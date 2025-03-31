import torch
import numpy as np
import cv2
import os
import json
import sys
import depth_pro
from ultralytics import YOLO

from Features.Object import Object

class TrafficLight(Object):

    def __init__(self, bbox, center, confidence, pose=None, color=None):
        super().__init__(bbox, center, confidence, pose)  # Call the parent constructor to initialize bbox, center, confidence, and pose
        
        self.color = color  # Color of the traffic light (e.g., 'red', 'green', 'yellow'). This can be set later if needed.

    def __repr__(self):
        return f"TrafficLight(bbox={self.bbox}, center={self.center}, confidence={self.confidence})"
    
    def __str__(self):
        return self.__repr__()
    
    def get_color(self):
        """
        Get the color of the traffic light.
        
        Returns:
            str: The color of the traffic light (e.g., 'red', 'green', 'yellow'). 
                 Returns None if not set.
        """
        return self.color
    
    def to_json(self, image_path=None):
        """
        Convert the TrafficLight object to a JSON-compatible dictionary format.
        This can be used to serialize the object for saving to a JSON file.
        """
        return {
            'name': 'TrafficLight',  # Name of the object type
            'image_path': image_path,  # Optional: path to the image if needed for reference
            'color': self.color,  # Include the color of the traffic light if set
            'object_data': super().to_json()
        }



# NOTE: nothing below is used (moved to main.py), but left for reference. Should delete later.

def process_scene(scene, data_dir, output_dir, model, depth_model, detector, conf_threshold):
    print(f"\nProcessing scene: {scene}")

    if os.path.exists(data_dir):
        # Create output directories
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Create a subdirectory for JSON files
        json_output_dir = os.path.join(output_dir, "json")
        os.makedirs(json_output_dir, exist_ok=True)

        # Get list of frames to process
        frame_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Found {len(frame_files)} frames to process in {scene}")

        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(data_dir, frame_file)

            # Load the image
            image = cv2.imread(frame_path)
            if image is None:
                print(f"Error: Could not load image {frame_path}")
                continue

            # Run object detection
            results = model(frame_path, conf=conf_threshold)

            # Run traffic sign detection
            sign_results = detector(frame_path, conf=conf_threshold)

            # Run depth estimation
            depth_map, _ = depth_model.infer(frame_path)

            # Save visualization
            output_path = os.path.join(output_dir, frame_file)

            # Prepare JSON data for Blender
            json_path = os.path.join(json_output_dir, os.path.splitext(frame_file)[0] + '.json')
            detections = []
            traffic_signs = []
            stop_signs = []

            # Process general detection results
            for r in results:
                # Save visualization image (we'll overwrite this later with combined results)
                r.save(filename=output_path)

                # Extract object detections
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]

                    # Calculate center point
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # Extract depth for this object
                    if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                        object_depth = float(depth_map[center_y, center_x])
                    else:
                        object_depth = 0.0

                    # Add to detections list with 3D information
                    detections.append({
                        'class': class_name,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'position': {
                            'x': center_x,
                            'y': center_y,
                            'depth': object_depth
                        },
                        'dimensions': {
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'depth': (x2 - x1) / 2  # Approximation
                        }
                    })

            # Process traffic sign detection results
            for r in sign_results:
                # Get the modified image with traffic sign detections
                annotated_img = r.plot()

                # Extract traffic sign detections
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = detector.names[cls]

                    # Calculate center point
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # Extract depth for this sign
                    if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                        sign_depth = float(depth_map[center_y, center_x])
                    else:
                        sign_depth = 0.0

                    # Add to traffic_signs list
                    sign_info = {
                        'class': class_name,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'position': {
                            'x': center_x,
                            'y': center_y,
                            'depth': sign_depth
                        },
                        'dimensions': {
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'depth': (x2 - x1) / 2  # Approximation
                        }
                    }

                    traffic_signs.append(sign_info)

                    # Check if this is a stop sign
                    if "stop" in class_name.lower():
                        stop_signs.append(sign_info)
                        # Draw a more prominent red box for stop signs
                        cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                        cv2.putText(annotated_img, "STOP", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Save the annotated image with both general and traffic sign detections
                cv2.imwrite(output_path, annotated_img)

            # Create combined data structure
            combined_data = {
                'frame_id': os.path.splitext(frame_file)[0],
                'objects': detections,
                'traffic_signs': traffic_signs,
                'stop_signs': stop_signs,  # Specifically highlight stop signs
                'lanes': {
                    'left_lane': [],  # Fill with actual lane data if available
                    'right_lane': []
                }
            }

            try:
                # Save to JSON
                with open(json_path, 'w') as f:
                    json.dump(combined_data, f, indent=4)

                # Print a special message if stop signs were detected
                if stop_signs:
                    print(f"⚠️ STOP SIGN DETECTED in {frame_file} at positions: " +
                          ", ".join([f"({s['position']['x']}, {s['position']['y']})" for s in stop_signs]))

                print(f"Processed {frame_file} ({i + 1}/{len(frame_files)})")
            except Exception as e:
                print(f"Error saving JSON for {frame_file}: {str(e)}")
                print(f"Attempted path: {json_path}")

def main():
    # Base directories
    base_data_dir = 'Code/P3Data/ExtractedFrames/Undist'
    base_output_dir = 'Code/outputs'

    # Set confidence threshold
    conf_threshold = 0.65
    scene = 'scene_10'

    # Initialize models
    # model, depth_model, detector = initialize_models()

    # Process scene
    data_dir = os.path.join(base_data_dir, scene)
    output_dir = os.path.join(base_output_dir, scene)
    os.makedirs(output_dir, exist_ok=True)

    # process_scene(scene, data_dir, output_dir, model, depth_model, detector, conf_threshold)

if __name__ == "__main__":
    main()
