import sys
import os

import cv2
import numpy as np
import torch
from PIL import Image
import glob
import tqdm
import matplotlib.pyplot as plt
import argparse


sys.path.append('RAFT/core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

class RAFTModel:
    
    
    def __init__(self, args, calibration_mtx, checkpoint_path="RAFT/models/raft-things.pth", load_model=True):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.args = args
        self.model = None
        self.checkpoint_path = checkpoint_path
        self.calibration_mtx = calibration_mtx
        
        if load_model:
            self._setup_model()
    
    def _setup_model(self):
        self.model = torch.nn.DataParallel(RAFT(self.args))
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()


    def load_image(self, imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self.device)
    
    
    def infer(self, img1_path, img2_path, save_path="outputs/optical_flow", show=False):
        if not isinstance(img1_path, str) or not isinstance(img2_path, str):
            raise ValueError("Both img1 and img2 should be file paths as strings.")
        
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            raise FileNotFoundError(f"One or both image files do not exist: {img1_path}, {img2_path}")
        
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please load the model before inference.")
        
        
        scene_img = img1_path.split('/')[-2:]
        scene_img = "/".join(scene_img).split(".")[0]
                    
        with torch.no_grad():
            image1 = self.load_image(img1_path)
            image2 = self.load_image(img2_path)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = self.model(image1, image2, iters=100, test_mode=True)
            flow_im = self.viz(image1, flow_up, show=show)
            flow = flow_up[0].permute(1,2,0).cpu().numpy()
            
            sampson = self.compute_sampson_distance(flow)
            sampson_norm = (sampson - sampson.min()) / (sampson.max() - sampson.min() + 1e-8)
            sampson_vis = (sampson_norm * 255).astype(np.uint8)
            
            # Save the results
            if save_path:
                cv2.imwrite(os.path.join(save_path, f"{scene_img}_flow.png"), flow_im)
                cv2.imwrite(os.path.join(save_path, f"{scene_img}_sampson.png"), sampson_vis)
                
                np.save(os.path.join(save_path, f"{scene_img}_flow.npy"), flow)
                np.save(os.path.join(save_path, f"{scene_img}_sampson.npy"), sampson)
            
        return flow_low, flow_up, flow_im
    
    def compute_sampson_distance(self, flow):
        h, w = flow.shape[:2]
        fx, fy = self.calibration_mtx[0, 0], self.calibration_mtx[1, 1]
        cx, cy = self.calibration_mtx[0, 2], self.calibration_mtx[1, 2]

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

        x1 = (grid_x - cx) / fx
        y1 = (grid_y - cy) / fy
        ones = np.ones_like(x1)
        p1 = np.stack([x1, y1, ones], axis=-1)

        x2 = (grid_x + flow[..., 0] - cx) / fx
        y2 = (grid_y + flow[..., 1] - cy) / fy
        p2 = np.stack([x2, y2, ones], axis=-1)

        norm_p1 = np.linalg.norm(p1, axis=-1, keepdims=True)
        norm_p2 = np.linalg.norm(p2, axis=-1, keepdims=True)
        p1_unit = p1 / (norm_p1 + 1e-8)
        p2_unit = p2 / (norm_p2 + 1e-8)

        dot = np.sum(p1_unit * p2_unit, axis=-1)
        dot = np.clip(dot, -1.0, 1.0)
        sampson_distance = 1.0 - dot

        return sampson_distance
            
    def viz(self, img, flo, show=False):
        img = img[0].permute(1,2,0).cpu().numpy()
        flo = flo[0].permute(1,2,0).cpu().numpy()
        
        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        img_flo = np.concatenate([img, flo], axis=0)

        # import matplotlib.pyplot as plt
        # plt.imshow(img_flo / 255.0)
        # plt.show()

        if show:
            cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
            cv2.waitKey()
            
        return img_flo
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--scene_num', type=int, default=-1, help='Scene number for the dataset')
    args = parser.parse_args()
    
    if args.scene_num == -1:
        raise ValueError("Please provide a scene number using --scene_num argument.")
    
    # Example usage
    camera_mtx = np.load("P3Data/Calib/calib_mat_front.npy")
    flow_model = RAFTModel(args, camera_mtx, checkpoint_path="RAFT/models/raft-things.pth", load_model=True)
    
    images_dir = f"P3Data/ExtractedFrames/Undist/scene_{args.scene_num}"
    images = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    
    output_path = "outputs/optical_flow"
    if not os.path.exists(os.path.join(output_path, f"scene_{args.scene_num}")):
        os.makedirs(os.path.join(output_path, f"scene_{args.scene_num}"))
    
    for i in tqdm.tqdm(range(len(images)-2)):
        img1 = images[i]
        img2 = images[i+1]
        flow_low, flow_up, flow_im = flow_model.infer(img1, img2, save_path="outputs/optical_flow")
    