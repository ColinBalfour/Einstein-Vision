
import cv2
import numpy as np
import matplotlib.pyplot as plt

import json
import os

from ImageModels.LaneSegmentation import *
from ImageModels.MetricDepth import *
from ImageModels.ObjectDetection import *  # Import ObjectDetection for object detection
from ImageModels.PoseEstimationModel import *
from ImageModels.SpeedOCR import OCRModel


def main():
    
    image_path = "P3Data/ExtractedFrames/Undist/scene_8/frame_000431.png"
    show = False
    
    
    img = cv2.imread(image_path)
    cv2.imwrite("outputs/original_image.png", img)
    
    output_pth = "json_outputs"
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
    else:
        # Clear the existing JSON files in the output directory
        for f in os.listdir(output_pth):
            if f.endswith('.json'):
                os.remove(os.path.join(output_pth, f))
        print(f"Cleared existing JSON files in {output_pth}")
    
    if show:
        cv2.imshow("Original Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    camera_mtx = np.load("P3Data/Calib/calib_mat_front.npy")
    
    load_model = False
    depth_model = MetricDepthModel(camera_mtx, load_model=load_model)
    if load_model:
        depth, normalized_depth, focal = depth_model.infer(image_path=image_path, focal_length=None, save_path="outputs/depth")
    else:
        depth = MetricDepthModel.get_depth_image_from_path(image_path)
        
        
    
    lane_model = LaneSegmentationModel(checkpoint_path="Code/ImageModels/lane_segmentation.pth")
    
    masks, boxes, labels = lane_model.infer(image_path=image_path, show=show, draw_boxes=True, OUT_DIR="outputs")
    lanes = lane_model.get_lanes_from_detection(masks, boxes, labels)
    
    
    object_model = YOLODetector(
        model_path='yolo11x-seg.pt',  # Path to the YOLO model
        classes=ObjectDetectionModel.YOLO_DEFAULT_CLASS_DETECTIONS,
        conf_threshold=0.4  # Confidence threshold for detection
    )
    
    # Run object detection on the same image
    results = object_model.get_outputs(image_path=image_path) # dictionary of Objects
    
    
    vehicle_model = DeticDectector(
        vocabulary={
            'car': ['car', 'mid_size_car', 'small_car'],
            'SUV': ['SUV_car', 'crossover_car', 'compact_SUV', 'large_SUV'],
            'hatchback': ['hatchback', 'compact'],
            'pickup': ['pickup', 'pickup_truck'],
            'truck': ['truck', 'bus', 'large_vehicle'],
            'bicycle': ['bicycle'],
            'motorcycle': ['motorcycle'],
        }
    )
    vehicle_results = vehicle_model.get_outputs(image_path=image_path)
    results.update(vehicle_results)  # Merge vehicle detection results with existing results
    
    
    # speed_limit_model = DeticDectector(
    #     vocabulary={
    #         'speed_limit': ['speed_limit'],
    #     }
    # )
    # ocr = OCRModel()
    # speed_limit_results = speed_limit_model.get_outputs(image_path=image_path)
    # for name, detected_objects in speed_limit_results.items():
    #     if name == 'speed_limit':
    #         for obj in detected_objects:
    #             # Assuming the OCR model can read the text from the detected object
    #             text = ocr.get_number(image_path, obj.bbox)
    #             obj.attr['speed_limit'] = text
    
    
    # Process the results
    debug_img = object_model.visualize(img, results)
    
    
    # Show the debug image with detections
    plt.imsave("outputs/debug_image.png", debug_img)
    if show:
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