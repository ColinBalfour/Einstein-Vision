
import cv2
import numpy as np
import matplotlib.pyplot as plt

import json
import os

from ImageModels.LaneSegmentation import *
from ImageModels.MetricDepth import *
from ImageModels.ObjectDetection import *  # Import ObjectDetection for object detection


def main():
    
    image_path = "P3Data/ExtractedFrames/Undist/scene_10/frame_000000.png"
    img = cv2.imread(image_path)
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    camera_mtx = np.load("P3Data/Calib/calib_mat_front.npy")
    
    depth_model = MetricDepthModel(camera_mtx)
    depth, normalized_depth, focal = depth_model.infer(image_path=image_path, focal_length=None, save_path="outputs/depth")
    
    # lane_model = LaneSegmentationModel(checkpoint_path="Code/ImageModels/lane_segmentation.pth")
    
    # masks, boxes, labels = lane_model.infer(image_path=image_path, show=True, draw_boxes=True, OUT_DIR="outputs")
    # lanes = lane_model.get_lanes_from_detection(masks, boxes, labels)
    # print(lanes[0])
    
    # pointcloud = np.zeros((0, 3))
    # for lane in lanes:
    #     points = depth_model.get_world_coords_from_keypoints(lane.keypoints, depth)
    #     pointcloud = np.vstack((pointcloud, points))
    
    # # vizualize pointcloud in matplotlib
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('3D Point Cloud')
    # plt.show()
    
    
    
    # display depth image
    # plt.ion()
    # fig = plt.figure()
    # ax_rgb = fig.add_subplot(121)
    # ax_disp = fig.add_subplot(122)
    
    # ax_rgb.imshow(img)
    # ax_disp.imshow(normalized_depth, cmap="turbo")
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    # plt.show()
    
    
    vehicle_model = ObjectDetectionModel(
        model_path='yolo12x.pt',  # Path to the YOLO model
        classes=['car'],
        conf_threshold=0.65  # Confidence threshold for detection
    )
    
    
    # Run object detection on the same image
    results = vehicle_model.infer(image_path=image_path) # dictionary of Objects
    
    print(results)
    
    # Process the results
    debug_img = vehicle_model.visualize(img, results)
    
    # Show the debug image with detections
    plt.imshow(debug_img)
    plt.show()
    
    
    output_pth = "json_outputs"
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
        
    i = 0
    for name, detected_objects in results.items():
        if name == 'car':
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



