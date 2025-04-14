
import cv2
import numpy as np
import matplotlib.pyplot as plt

import json
import os

from ImageModels.LaneSegmentation import *
from ImageModels.MetricDepth import *
from ImageModels.ObjectDetection import *  # Import ObjectDetection for object detection
from ImageModels.PoseEstimationModel import *


def main():
    
    image_path = "P3Data/ExtractedFrames/Undist/scene_8/frame_000101.png"
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
    
    camera_mtx = np.load("D:/3. Computer vision/Homeworks/Einstein-Vision/P3Data/Calib/calib_mat_front.npy")
    
    depth_model = MetricDepthModel(camera_mtx)
    depth, normalized_depth, focal = depth_model.infer(image_path=image_path, focal_length=None, save_path="outputs/depth")
    
    # lane_model = LaneSegmentationModel(checkpoint_path="Code/ImageModels/lane_segmentation.pth")
    #
    # masks, boxes, labels = lane_model.infer(image_path=image_path, show=True, draw_boxes=True, OUT_DIR="outputs")
    # lanes = lane_model.get_lanes_from_detection(masks, boxes, labels)
    
    
    object_model = ObjectDetectionModel(
        model_path='D:/3. Computer vision/Homeworks/Einstein-Vision/Code/ImageModels/yolo12x.pt',  # Path to the YOLO model
        classes=['car', 'traffic light', 'person', 'stop sign'],
        conf_threshold=0.4  # Confidence threshold for detection
    )
    
    
    # Run object detection on the same image
    results = object_model.infer(image_path=image_path) # dictionary of Objects

    #Initialise pose estimation model
    print("running pose model")
    pose_model = PoseEstimationModel(
        model_path='D:/3. Computer vision/Homeworks/Einstein-Vision/Code/ImageModels/yolo11x-pose.pt',
        conf_threshold=0.4
    )

    # Get pose data if people were detected
    print("extracted each of the frames")
    if 'person' in results and results['person']:
        print(f"Number of pose data items: {len(results)}")
        print(f"Number of detected people: {len(results['person'])}")
        pose_data = pose_model.infer(image_path=image_path, show=True, save_path="outputs")
        print("pose_Data", pose_data)

        # Get 3D pose data using depth information
        poses_3d = pose_model.estimate_3d_poses(pose_data, depth_model, depth)
        print("pose estimation results:", poses_3d)

        # Match poses to detected pedestrians
        if pose_data:
            matches = pose_model.match_poses_to_pedestrians(pose_data, results['person'])

            # Set pose data for each matched pedestrian
            for person, keypoints in matches.items():
                # Find the corresponding 3D pose data
                for pose_3d in poses_3d:
                    if np.array_equal(pose_3d['keypoints_2d'], keypoints):
                        # Set 3D pose for the pedestrian
                        # Assuming your Object class can store keypoints in the pose attribute
                        person.pose = {
                            'keypoints_2d': keypoints,
                            'keypoints_3d': pose_3d['keypoints_3d']
                        }
                        break

    # print(results)
    
    # Process the results
    debug_img = object_model.visualize(img, results)
    
    # Show the debug image with detections
    plt.imshow(debug_img)
    plt.show()
    
    
    # display depth image
    fig = plt.figure()
    ax_rgb = fig.add_subplot(121)
    ax_disp = fig.add_subplot(122)
    
    ax_rgb.imshow(img)
    ax_disp.imshow(normalized_depth, cmap="turbo")
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()
    
    
    # for i, lane in enumerate(lanes):
    #     world_pts = depth_model.get_world_coords_from_keypoints(lane.keypoints, depth)
    #     lane.set_world_coords(world_pts)  # Set the world coordinates for the lane object
    #     json_object = lane.to_json(image_path)
    #     json_filename = os.path.join(output_pth, f"lane_{i}.json")
    #     with open(json_filename, 'w') as json_file:
    #         # Save the JSON output for each lane
    #         json.dump(json_object, json_file, indent=4)
    #     print(f"Saved JSON output for lane {i} to {json_filename}")
        
        
    i = 0
    for name, detected_objects in results.items():
        for obj in detected_objects:
            center = obj.get_center()
            t = depth_model.get_translation_at_point(center[0], center[1], depth)
            
            print(f"Detected {name} at pixel coordinates {center} with depth translation: {t}")

            # Set 3D position and orientation (assuming no rotation for simplicity)
            obj_pose = [*t, 0, 0, 0]  # x, y, z, roll, pitch, yaw

            # If the object already has pose data (like a pedestrian with keypoints),
            # update it to include position and orientation
            if hasattr(obj, 'pose') and obj.pose is not None:
                # If pose is a dict with keypoints, add position and orientation
                if isinstance(obj.pose, dict) and 'keypoints_2d' in obj.pose:
                    obj.pose['position'] = t
                    obj.pose['orientation'] = [0, 0, 0]  # Default orientation
                else:
                    # Otherwise just set the pose directly
                    obj.pose = obj_pose
            else:
                # If no pose data, set the basic pose
                obj.pose = obj_pose
            
            # Save to JSON format
            json_output = obj.to_json(image_path)
            # Add keypoints to the JSON after to_json is called
            if name == 'person' and hasattr(obj, 'pose') and isinstance(obj.pose, dict):
                if 'keypoints_2d' in obj.pose:
                    if 'pose' not in json_output:
                        json_output['pose'] = {}
                    json_output['pose']['keypoints_2d'] = obj.pose['keypoints_2d'].tolist() if isinstance(
                        obj.pose['keypoints_2d'], np.ndarray) else obj.pose['keypoints_2d']

                if 'keypoints_3d' in obj.pose:
                    if 'pose' not in json_output:
                        json_output['pose'] = {}
                    json_output['pose']['keypoints_3d'] = obj.pose['keypoints_3d'].tolist() if isinstance(
                        obj.pose['keypoints_3d'], np.ndarray) else obj.pose['keypoints_3d']

            json_filename = os.path.join(output_pth, f"{name}_{i}.json")
            with open(json_filename, 'w') as json_file:
                json.dump(json_output, json_file, indent=4)
            print(f"Saved JSON output for {name} to {json_filename}")
            i += 1


    
    
if __name__ == "__main__":
    main()



