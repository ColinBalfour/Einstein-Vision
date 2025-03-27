
import cv2
import numpy as np

from ImageModels.LaneSegmentation import *
from ImageModels.MetricDepth import *

import matplotlib.pyplot as plt



def main():
    
    image_path = "P3Data/ExtractedFrames/Undist/scene_5/frame_000000.png"
    img = cv2.imread(image_path)
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    camera_mtx = np.load("P3Data/Calib/calib_mat_front.npy")
    
    lane_model = LaneSegmentationModel(checkpoint_path="Code/ImageModels/lane_segmentation.pth")
    depth_model = MetricDepthModel(camera_mtx)
    
    masks, boxes, labels = lane_model.infer(image_path=image_path, show=True, draw_boxes=True, OUT_DIR="outputs")
    lanes = lane_model.get_lanes_from_detection(masks, boxes, labels)
    print(lanes[0])
    
    depth, focal = depth_model.infer(image_path=image_path, focal_length=None)
    
    # display depth image
    depth = np.squeeze(depth)
    plt.imshow(depth, cmap='plasma')
    plt.colorbar()
    plt.show()
    
    pointcloud = np.zeros((0, 3))
    for lane in lanes:
        points = depth_model.get_world_coords_from_keypoints(lane.keypoints, depth)
        pointcloud = np.vstack((pointcloud, points))
    
    # vizualize pointcloud in matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')
    plt.show()
    
    
    
    
    
    
if __name__ == "__main__":
    main()



