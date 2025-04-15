
import cv2
import numpy as np
import matplotlib.pyplot as plt

import json
import os

from ImageModels.LaneSegmentation import *
from ImageModels.MetricDepth import *
from ImageModels.ObjectDetection import *  # Import ObjectDetection for object detection


def main():
    
    image_path = "P3Data/ExtractedFrames/Undist/scene_9/frame_000298.png"
    img = cv2.imread(image_path)
    
    output_pth = "json_outputs"
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
    else:
        # Clear the existing JSON files in the output directory
        for f in os.listdir(output_pth):
            if f.endswith('.json'):
                os.remove(os.path.join(output_pth, f))
        print(f"Cleared existing JSON files in {output_pth}")
    
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    camera_mtx = np.load("P3Data/Calib/calib_mat_front.npy")
    
    depth = MetricDepthModel.get_depth_image_from_path(image_path)
    
    lane_model = LaneSegmentationModel(checkpoint_path="Code/ImageModels/lane_segmentation.pth")
    
    masks, boxes, labels = lane_model.infer(image_path=image_path, show=True, draw_boxes=True, OUT_DIR="outputs")
    lanes = lane_model.get_lanes_from_detection(masks, boxes, labels)
    
    
    object_model = ObjectDetectionModel(
        model_path='yolo12x.pt',  # Path to the YOLO model
        classes=['car', 'traffic light', 'person', 'stop sign'],
        conf_threshold=0.4  # Confidence threshold for detection
    )
    
    
    # Run object detection on the same image
    results = object_model.infer(image_path=image_path) # dictionary of Objects
    
    # print(results)
    
    # Process the results
    debug_img = object_model.visualize(img, results)
    
    # Show the debug image with detections
    plt.imshow(debug_img)
    plt.show()
    
    
    for i, lane in enumerate(lanes):
        world_pts = depth_model.get_world_coords_from_keypoints(lane.keypoints, depth)
        lane.set_world_coords(world_pts)  # Set the world coordinates for the lane object
        json_object = lane.to_json(image_path)
        json_filename = os.path.join(output_pth, f"lane_{i}.json")
        with open(json_filename, 'w') as json_file:
            # Save the JSON output for each lane
            json.dump(json_object, json_file, indent=4)
        print(f"Saved JSON output for lane {i} to {json_filename}")
        
        
    i = 0
    for name, detected_objects in results.items():
        for obj in detected_objects:
            center = obj.get_center()
            t = depth_model.get_translation_at_point(center[0], center[1], depth)
            
            print(f"Detected {name} at pixel coordinates {center} with depth translation: {t}")
            
            pose = [*t, 0, 0, 0] # 0 for roll, pitch, yaw (assuming no rotation for simplicity)
            obj.set_pose(pose)
            
            # Save to JSON format
            json_output = obj.to_json(image_path)
            json_filename = os.path.join(output_pth, f"{name}_{i}.json")
            with open(json_filename, 'w') as json_file:
                json.dump(json_output, json_file, indent=4)
            print(f"Saved JSON output for {name} to {json_filename}")
            i += 1
    
    
if __name__ == "__main__":
    main()