from PIL import Image
import depth_pro

class MetricDepthModel:
    
    def __init__(self, calibration_mtx):
        # Load model and preprocessing transform
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.eval()
        self.device = self.model.device
        
        self.calibration_mtx = calibration_mtx
        
        
    def infer(self, image_path, focal_length=None):
        
        # Load and preprocess an image.
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = self.transform(image)
        
        if focal_length:
            f_px = focal_length

        # Run inference.
        prediction = self.model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"]  # Focal length in pixels.
        
        return depth, focallength_px
        
    def get_world_coords_from_keypoints(self, keypoints):
        
        fx, fy, cx, cy = self.calibration_mtx[0, 0], self.calibration_mtx[1, 1], self.calibration_mtx[0, 2], self.calibration_mtx[1, 2]
        
        