
import depth_pro
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
import numpy as np
import open3d as o3d
import os
import torch

class MetricDepthModel:
    
    def __init__(self, calibration_mtx):
        # Load model and preprocessing transform
        depth_pro_config = DEFAULT_MONODEPTH_CONFIG_DICT
        depth_pro_config.checkpoint_uri = os.path.join("ml-depth-pro/", depth_pro_config.checkpoint_uri)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # print(depth_pro_config.checkpoint_uri)
        
        self.model, self.transform = depth_pro.create_model_and_transforms(config=depth_pro_config, device=device)
        self.model.eval()
        self.device = device
        
        # print(self.model)
        print(self.device)
        
        self.calibration_mtx = calibration_mtx
        
        
    def infer(self, image_path, focal_length=None):
        
        # Load and preprocess an image.
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = self.transform(image)
        
        if focal_length:
            f_px = focal_length

        # Run inference.
        prediction = self.model.infer(image, f_px=f_px)
        
        # Get the depth and focal length
        depth = prediction["depth"].cpu().numpy()  # Depth in [m].
        focallength_px = prediction["focallength_px"].cpu()  # Focal length in pixels.
        
        return depth, focallength_px
        
    def get_world_coords_from_keypoints(self, keypoints, depth_image):
        
        fx, fy, cx, cy = self.calibration_mtx[0, 0], self.calibration_mtx[1, 1], self.calibration_mtx[0, 2], self.calibration_mtx[1, 2]
        
        keypoints = keypoints.astype(int)
        r, c = keypoints[:, 1], keypoints[:, 0]

        z = depth_image[keypoints[:, 1], keypoints[:, 0]]
        x =  z * (c - cx) / fx
        y =  z * (r - cy) / fy
    
        z = np.ravel(z)
        x = np.ravel(x)
        y = np.ravel(y)

        pointsxyz = np.dstack((x, y, z))
        pointsxyz[~np.isfinite(pointsxyz)] = 0 
        
        pointsxyz = pointsxyz.reshape(-1,3)
        
        return pointsxyz
        
        
        