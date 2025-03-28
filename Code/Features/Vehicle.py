import torch
import numpy as np
import cv2
import os
import json
import sys
import depth_pro
from ultralytics import YOLO


def initialize_models():
    # Check GPU availability and set CUDA settings
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU instead")

    # Load YOLO model with GPU support
    model = YOLO('yolo11x.pt')
    if torch.cuda.is_available():
        model.to('cuda')  # Move model to GPU

    # Initialize depth model
    depth_model = MetricDepthModel()

    return model, depth_model


# depth model
class MetricDepthModel:
    def __init__(self):
        # Load model and preprocessing transform
        self.model, self.transform = depth_pro.create_model_and_transforms()

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.eval()

    def infer(self, image_path, focal_length=None):
        # Load and preprocess an image
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = self.transform(image)

        # Move image to GPU if available
        if torch.cuda.is_available():
            image = image.cuda()

        if focal_length:
            f_px = focal_length

        # Run inference
        prediction = self.model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth

        # Move depth map back to CPU for numpy processing
        if torch.cuda.is_available():
            depth = depth.cpu().numpy()
        else:
            depth = depth.numpy()

        focallength_px = prediction["focallength_px"]  # Focal length in pixels

        return depth, focallength_px


def process_scene(scene, data_dir, output_dir, model, depth_model, conf_threshold):
    print(f"\nProcessing scene: {scene}")

    if os.path.exists(data_dir):
       
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

            # Run depth estimation
            try:
                depth_map, _ = depth_model.infer(frame_path)

                # Save visualization
                output_path = os.path.join(output_dir, frame_file)

                # Save combined data for Blender
                json_path = os.path.join(output_dir, os.path.splitext(frame_file)[0] + '.json')
                detections = []

                for r in results:
                    r.save(filename=output_path)

                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = model.names[cls]

                        # Get depth for this object
                        # Center point of the bounding box
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                        # Extract depth at this point (if within bounds)
                        if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                            object_depth = float(depth_map[center_y, center_x])
                        else:
                            object_depth = 0.0

                        # Add 3D information to detection
                        detections.append({
                            'class': class_name,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'position': {
                                'x': center_x,
                                'y': center_y,
                                'depth': object_depth
                            }
                        })

                # Combine object and lane data
                combined_data = {
                    'frame_id': os.path.splitext(frame_file)[0],
                    'objects': detections
                }

                with open(json_path, 'w') as f:
                    json.dump(combined_data, f, indent=4)

                # Print progress
                if (i + 1) % 10 == 0 or i == 0 or i == len(frame_files) - 1:
                    print(f"Processed {i + 1}/{len(frame_files)} frames in {scene}")

            except Exception as e:
                print(f"Error processing {frame_file} in {scene}: {str(e)}")
                continue
    else:
        print(f"ERROR: Directory does not exist: {data_dir}")


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
    model, depth_model = initialize_models()

    # Process each scene
    # for scene in scenes:
    data_dir = os.path.join(base_data_dir, scene)
    output_dir = os.path.join(base_output_dir, scene)
    os.makedirs(output_dir, exist_ok=True)

    process_scene(scene, data_dir, output_dir, model, depth_model, conf_threshold)




if __name__ == "__main__":
    main()
