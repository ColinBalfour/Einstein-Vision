
import cv2
import numpy as np

from ImageModels.LaneSegmentation import *
from ImageModels.MetricDepth import *



def main():
    
    image_path = "P3Data/ExtractedFrames/Undist/scene_2/frame_000000.png"
    img = cv2.imread(image_path)
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    lane_model = LaneSegmentationModel(checkpoint_path="lane_segmentation.pth")
    depth_model = MetricDepthModel()
    
    masks, boxes, labels = lane_model.infer(image_path=image_path, show=True, draw_boxes=True, OUT_DIR="outputs")
    lanes = lane_model.get_lanes_from_detection(masks, boxes, labels)
    print(lanes)
    
    depth, focal = depth_model.infer(image_path=image_path, focal_length=None)
    
    
    
    
    
if __name__ == "__main__":
    main()



