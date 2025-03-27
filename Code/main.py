
import cv2
import numpy as np

from ImageModels.LaneSegmentation import *
from ImageModels.MetricDepth import *



def main():
    
    image_path = "input_image.png"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    lane_model = LaneSegmentationModel(checkpoint_path="lane_segmentation.pth")
    depth_model = MetricDepthModel()
    
    
    
if __name__ == "__main__":
    main()



