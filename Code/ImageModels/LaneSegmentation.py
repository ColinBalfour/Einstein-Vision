import cv2
import numpy as np
from PIL import Image
import os
from typing import List

import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import transforms as transforms

from Features.Lane import Lane
from .infer_utils import draw_segmentation_map, get_outputs

class LaneSegmentationModel:
    
    INSTANCE_CATEGORY_NAMES = [
        '__background__', 
        'divider-line',
        'dotted-line',
        'double-line',
        'random-line',
        'road-sign-line',
        'solid-line'
    ]
    
    def __init__(self, threshold=0.4, checkpoint_path=None):
        """
        Initializes the LaneSegmentationModel by loading the pretrained Mask R-CNN model for lane segmentation.
        
        Args:
            threshold (float, optional): The confidence threshold for filtering out detections. Defaults to 0.8.
            checkpoint_path (str, optional): Path to the model checkpoint. If None, the default path is used.
        """
        self.threshold = threshold
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(os.path.dirname(__file__), "lane_segmentation.pth")
        
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            pretrained=False, num_classes=91
        )
        
        model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=len(LaneSegmentationModel.INSTANCE_CATEGORY_NAMES), bias=True)
        model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=len(LaneSegmentationModel.INSTANCE_CATEGORY_NAMES)*4, bias=True)
        model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, len(LaneSegmentationModel.INSTANCE_CATEGORY_NAMES), kernel_size=(1, 1), stride=(1, 1))
        
        # initialize the model
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        # set the computation device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load the modle on to the computation device and set to eval mode
        model.to(device).eval()
        # print(model)
        print(device)
        
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def infer(self, image_path, show=False, draw_boxes=False, OUT_DIR=None):
        """
        Performs lane segmentation on a given image and optionally displays or saves the result.
        
        Args:
            image_path (str): Path to the image to be segmented.
            show (bool, optional): If True, displays the segmented image using OpenCV. Defaults to False.
            draw_boxes (bool, optional): If True, draws bounding boxes around detected lanes. Defaults to False.
            OUT_DIR (str, optional): Directory to save the output image. If None, the result is not saved. Defaults to None.
        
        Returns:
            tuple: A tuple containing the segmentation masks, bounding boxes, and labels for the detected lanes.
        """
        # print(image_path)
        image = Image.open(image_path)
        # keep a copy of the original image for OpenCV functions and applying masks
        orig_image = image.copy()
        
        # transform the image
        image = self.transform(image)
        # add a batch dimension
        image = image.unsqueeze(0).to(self.device)
        
        masks, boxes, labels = get_outputs(image, self.model, self.threshold, LaneSegmentationModel.INSTANCE_CATEGORY_NAMES)
        
        result = draw_segmentation_map(orig_image, masks, boxes, labels, draw_boxes, LaneSegmentationModel.INSTANCE_CATEGORY_NAMES)
        
        # visualize the image
        if show:
            cv2.imshow('Segmented image', np.array(result))
            cv2.waitKey(0)
        
        # set the save path
        if OUT_DIR:
            save_path = f"{OUT_DIR}/{image_path.split(os.path.sep)[-1].split('.')[0]}.jpg"
            cv2.imwrite(save_path, result)
            
        return masks, boxes, labels
    
    
    def get_lanes_from_image(self, image_path) -> List[Lane]:
        """
        Processes an image, performs lane segmentation, and extracts the detected lanes as Lane objects.
        
        Args:
            image_path (str): Path to the image to be processed.
        
        Returns:
            List[Lane]: A list of Lane objects representing the detected lanes in the image.
        """
        masks, boxes, labels = self.infer(image_path)
        return self.get_lanes_from_detection(masks, boxes, labels)
    
    def get_lanes_from_detection(self, masks, boxes, labels) -> List[Lane]:
        """
        Converts segmentation masks, bounding boxes, and labels from a detection model into Lane objects.
        
        Args:
            masks (list): A list of binary masks for detected lanes.
            boxes (list): A list of bounding boxes for detected lanes.
            labels (list): A list of labels corresponding to the detected lanes.
        
        Returns:
            List[Lane]: A list of Lane objects created from the detected masks, boxes, and labels.
        """
        lanes = []
        for mask, box, label in zip(masks, boxes, labels):
            print(mask.shape)
            all_points = np.argwhere(mask)
            # convert to x, y coordinates
            x_coords = all_points[:, 1]
            y_coords = all_points[:, 0]
            
            # fit a polynomial to the points
            coeffs = np.polyfit(y_coords, x_coords, 2)
            
            keypoints_y = np.linspace(min(y_coords), max(y_coords), num=10)
            keypoints_x = np.polyval(coeffs, keypoints_y)
            
            # filter out points outside the image
            keypoints_x = np.clip(keypoints_x, 0, mask.shape[1] - 1)
            
            keypoints = np.column_stack((keypoints_x, keypoints_y))
            # keypoints = np.column_stack((x_coords, y_coords))
            
            # create a Lane object
            lanes.append(Lane(
                keypoints=keypoints,
                lane_type=label,
                world_coords=None
            ))
        
        return lanes
