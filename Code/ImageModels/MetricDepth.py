
import depth_pro
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
import numpy as np
# import open3d as o3d
import os
import torch
import cv2
import matplotlib.pyplot as plt
import glob
import tqdm

class MetricDepthModel:
    
    def __init__(self, calibration_mtx, load_model=True):
        # Load model and preprocessing transform
        depth_pro_config = DEFAULT_MONODEPTH_CONFIG_DICT
        depth_pro_config.checkpoint_uri = os.path.join("ml-depth-pro/", depth_pro_config.checkpoint_uri)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # print(depth_pro_config.checkpoint_uri)
        
        self.model = None        
        if load_model:
            # Load the model
            self.model, self.transform = depth_pro.create_model_and_transforms(config=depth_pro_config, device=device)
            self.model.eval()

        self.device = device
        
        # print(self.model)
        print(self.device)
        
        self.calibration_mtx = calibration_mtx
        
        
    def infer(self, image_path, focal_length=None, save_path=None):
        
        scene_img = image_path.split('/')[-2:]
        scene_img = "/".join(scene_img).split(".")[0]
        
        # Load and preprocess an image.
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = self.transform(image)
        
        if focal_length:
            f_px = focal_length

        # Run inference.
        prediction = self.model.infer(image, f_px=f_px)
        
        # Get the depth and focal length
        depth = prediction["depth"].detach().cpu().numpy().squeeze()  # Depth in [m].
        focallength_px = prediction["focallength_px"].cpu()  # Focal length in pixels.
        
        inverse_depth = 1 / depth
        # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )
        
        if save_path:
            # Save as color-mapped "turbo" jpg image.
            cmap = plt.get_cmap("turbo")
            color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
                np.uint8
            )
            np.save(os.path.join(save_path, f"{scene_img}_depth.npy"), depth)
            cv2.imwrite(os.path.join(save_path, f"{scene_img}_normalized_depth.png"), (inverse_depth_normalized * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(save_path, f"{scene_img}_depth_im.png"), color_depth)

        
        return depth, inverse_depth_normalized, focallength_px
        
    
    def get_translation_at_point(self, x, y, depth_image):
        
        fx, fy, cx, cy = self.calibration_mtx[0, 0], self.calibration_mtx[1, 1], self.calibration_mtx[0, 2], self.calibration_mtx[1, 2]
        
        # Get the depth at a specific pixel (x, y)
        z = depth_image[y, x]  # Depth in meters at pixel (x, y)
        if not np.isfinite(z):
            return 0.0
        
        x = z * (x - cx) / fx  # X coordinate in meters
        y = z * (y - cy) / fy
        
        return np.array([x, y, z], dtype=np.float64)

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
    
    @staticmethod
    def get_depth_image_from_path(image_path, output_dir="outputs/depth"):
        
        scene_img = image_path.split('/')[-2:]
        scene_img = "/".join(scene_img).split(".")[0]
        
        im_path = os.path.join(output_dir, f"{scene_img}_depth.npy")
        depth_image = np.load(im_path)
        
        return depth_image
    
    @staticmethod
    def get_RGBD_image(img, depth_img):
        # take RGB image and depth image and return RGBD image
        # stack the RGB image and depth image along the last axis
        rgbd_image = np.dstack((img, depth_img))
        
        return rgbd_image
        
        
if __name__ == '__main__':
    
    # get scene number from cmdline
    import sys
    if len(sys.argv) > 1:
        scene_num = sys.argv[1]
    else:
        print("No scene number provided. Usage: python3 MetricDepth.py <scene_num>")
        sys.exit(1)
    
    # Example usage
    camera_mtx = np.load("P3Data/Calib/calib_mat_front.npy")
    depth_model = MetricDepthModel(camera_mtx)
    
    images_dir = f"P3Data/ExtractedFrames/Undist/scene_{scene_num}"
    images = glob.glob(os.path.join(images_dir, "*.png"))
    
    for image_path in tqdm.tqdm(images):
        depth, normalized_depth, focal = depth_model.infer(image_path=image_path, focal_length=None, save_path="outputs/depth")
        
    # test get_depth_image_from_path
    depth = depth_model.get_depth_image_from_path(images[0])
    
    inverse_depth = 1 / depth
    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
        max_invdepth_vizu - min_invdepth_vizu
    )

    # Save as color-mapped "turbo" jpg image.
    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
        np.uint8
    )
    cv2.imwrite(os.path.join(f"outputs/depth/scene_{scene_num}", "atest_depth_normalized.png"), (inverse_depth_normalized * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(f"outputs/depth/scene_{scene_num}", "atest_depth_color.png"), color_depth)
    