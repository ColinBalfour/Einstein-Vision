
import cv2
import numpy as np
import matplotlib.pyplot as plt

import json
import os

from ImageModels.LaneSegmentation import *
from ImageModels.MetricDepth import *
from ImageModels.ObjectDetection import *  # Import ObjectDetection for object detection
from ImageModels.PoseEstimationModel import *
from ImageModels.ClassicalModels import *
from ImageModels.OpticalFlow import RAFTModel


def main():
    
    key_images = {
        "cars_with_bike": "P3Data/ExtractedFrames/Undist/scene_8/frame_000431.png",
        "truck_cars_highway": "P3Data/ExtractedFrames/Undist/scene_1/frame_000256.png",
        "traffic_light": "P3Data/ExtractedFrames/Undist/scene_4/frame_000045.png",
        "traffic_cones_brake_light": "P3Data/ExtractedFrames/Undist/scene_6/frame_000177.png",
        "traffic_light2": "P3Data/ExtractedFrames/Undist/scene_7/frame_000200.png",
        "pedestrian_stop_sign": "P3Data/ExtractedFrames/Undist/scene_8/frame_000200.png",
        "speed_limit": "P3Data/ExtractedFrames/Undist/scene_9/frame_000381.png",
        "motorcycle": "P3Data/ExtractedFrames/Undist/scene_13/frame_000244.png",
    }
    
    image_path = key_images["traffic_cones_brake_light"]
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
    
    # Get Depth Image
    camera_mtx = np.load("P3Data/Calib/calib_mat_front.npy")
    
    load_model = False
    depth_model = MetricDepthModel(camera_mtx, load_model=load_model)
    if load_model:
        depth, normalized_depth, focal = depth_model.infer(image_path=image_path, focal_length=None, save_path="outputs/depth")
    else:
        depth = MetricDepthModel.get_depth_image_from_path(image_path)
        
    
    # Get Optical Flow
    load_model = False
    optical_flow_model = RAFTModel()
    if load_model:
        _, flow, flow_im = optical_flow_model.infer(image_path=image_path, save_path="outputs/optical_flow")
        sampson = optical_flow_model.compute_sampson_distance(flow)
    else:
        # Load the precomputed optical flow
        flow, sampson = RAFTModel.load_flow_from_path(image_path) 
    
    
    
    lane_model = LaneSegmentationModel(checkpoint_path="Code/ImageModels/lane_segmentation.pth")
    
    masks, boxes, labels = lane_model.infer(image_path=image_path, show=show, draw_boxes=True, OUT_DIR="outputs")
    lanes = lane_model.get_lanes_from_detection(masks, boxes, labels)
    
    
    object_model = YOLODetector(
        model_path='yolo11x-seg.pt',  # Path to the YOLO model
        classes=ObjectDetectionModel.YOLO_DEFAULT_CLASS_DETECTIONS,
        conf_threshold=0.4  # Confidence threshold for detection
    )
    
    # Run object detection on the same image
    results = object_model.get_outputs(img=image_path) # dictionary of Objects
    
    for name, detected_objects in results.items():
        if name == 'traffic light':
            for obj in detected_objects:
                # Check if the object is a traffic light
                if isinstance(obj, TrafficLight):
                    # Set the arrow direction based on the detected object
                    color, arrow_direction = detect_traffic_light_arrows(img.copy(), obj, 'rgb')
                    obj.color = color
                    obj.arrow_direction = arrow_direction
                    print(f"Detected traffic light with color: {color} and arrow direction: {arrow_direction}")
    
    
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
    tailight_model = DeticDectector(
        vocabulary={
            # 'taillight': ['taillight_off', 'car_taillight_off', 'car_rear_light_off', 'rear_light_off', 'brake_light_off', 'turn_signal_off'],
            # 'brake_light': ['brake_light_on', 'car_brake_light_on', 'car_stopping_light_on', 'rear_light_on', 'red_light_on', 'braking_light'],
            # 'turn_signal': ['turn_signal_on'],
            'taillight': ['taillight', 'car_taillight', 'car_rear_light', 'rear_light', 'brake_light', 'turn_signal', 'car_light'],
        }
    )
    vehicle_results = vehicle_model.get_outputs(img=image_path)
    
    # Detect taillights on vehicles
    for name, detected_objects in vehicle_results.items():
        for i, obj in enumerate(detected_objects):
            
            # Get an array of detected vehicle coorindinates (bbox or mask, if available)
            pixel_coords = []
            if obj.mask is not None:
                # Get the bounding box from the mask
                mask = obj.mask.astype(np.uint8)
                for y in range(mask.shape[0]):
                    for x in range(mask.shape[1]):
                        if mask[y, x] > 0:
                            pixel_coords.append((x, y))
            elif obj.bbox is not None:
                # Use the bounding box directly
                for y in range(obj.bbox[1], obj.bbox[3]):
                    for x in range(obj.bbox[0], obj.bbox[2]):
                        pixel_coords.append((x, y))

            pixel_coords = np.array(pixel_coords)
            
            rows = pixel_coords[:, 1]
            cols = pixel_coords[:, 0]
            
            mean_sampson = np.mean(sampson[rows, cols])
            obj.moving = mean_sampson > 0.1
                
            
                
            
            x1, y1, x2, y2 = obj.bbox
            
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img.shape[1], x2 + padding)
            y2 = min(img.shape[0], y2 + padding)
            
            # print(x1, y1, x2, y2)
            # Crop the image to the detected vehicle bounding box
            cropped_img = img.copy()[y1:y2, x1:x2]
            cv2.imwrite(f"outputs/vehicles_debug/cropped_vehicle_{i}.png", cropped_img)
            
            # Check for taillights and turn signals
            taillights = tailight_model.get_outputs(img=cropped_img)
            
            leftmost = None
            rightmost = None
            for light_name, light_objects in taillights.items():
                for light_obj in light_objects:
                    if leftmost is None or light_obj.bbox[0] < leftmost.bbox[0]:
                        leftmost = light_obj
                        
                    if rightmost is None or light_obj.bbox[0] > rightmost.bbox[0]:
                        rightmost = light_obj
            
            if rightmost == leftmost and rightmost is not None:
                # set side closest to edge of bbox as the direction
                if rightmost.bbox[0] < (obj.bbox[0] + obj.bbox[2]) / 2:
                    rightmost = None
                else:
                    leftmost = None
            
            if leftmost:
                # Set the direction based on the leftmost taillight
                leftmost.direction = 'left'
                # Update bbox coordinates to the original image
                print('conf', leftmost.confidence)
                print('before bbox:', leftmost.bbox)
                print('original bbox:', obj.bbox)
                leftmost.bbox[0] += x1
                leftmost.bbox[1] += y1
                leftmost.bbox[2] += x1
                leftmost.bbox[3] += y1
                print('new bbox:', leftmost.bbox)
                
                leftmost.center = [(leftmost.bbox[0] + leftmost.bbox[2]) / 2, (leftmost.bbox[1] + leftmost.bbox[3]) / 2]
                
                light_type = detect_brake_and_indicator_lights(img.copy(), leftmost, 'rgb')
                leftmost.light_type = light_type
                
                obj.left_taillight = leftmost
                
            if rightmost:
                # Set the direction based on the rightmost taillight
                rightmost.direction = 'right'
                # Update bbox coordinates to the original image
                print('conf', rightmost.confidence)
                print('before bbox:', rightmost.bbox)
                print('original bbox:', obj.bbox)
                rightmost.bbox[0] += x1
                rightmost.bbox[1] += y1
                rightmost.bbox[2] += x1
                rightmost.bbox[3] += y1
                print('new bbox:', rightmost.bbox)
                
                rightmost.center = [(rightmost.bbox[0] + rightmost.bbox[2]) / 2, (rightmost.bbox[1] + rightmost.bbox[3]) / 2]
                
                light_type = detect_brake_and_indicator_lights(img.copy(), rightmost, 'rgb')
                rightmost.light_type = light_type
                
                obj.right_taillight = rightmost       
    
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