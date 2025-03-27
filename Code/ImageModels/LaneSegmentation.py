
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
from infer_utils import draw_segmentation_map, get_outputs

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
    
    def __init__(self, threshold=0.8, checkpoint_path=None):
        
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
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model'])
        # set the computation device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load the modle on to the computation device and set to eval mode
        model.to(device).eval()
        print(model)
        print(device)
        
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def infer(self, image_path, show=False, draw_boxes=False, OUT_DIR=None):
        
        print(image_path)
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
            cv2.waitKey(1)
        
        # set the save path
        if OUT_DIR:
            save_path = f"{OUT_DIR}/{image_path.split(os.path.sep)[-1].split('.')[0]}.jpg"
            cv2.imwrite(save_path, result)
            
        return masks, boxes, labels
    
    
    def get_lanes_from_image(self, image_path) -> List[Lane]:
        pass
    
    
    def get_lanes_from_detection(self, maxes, boxes, labels) -> List[Lane]:
        pass
        