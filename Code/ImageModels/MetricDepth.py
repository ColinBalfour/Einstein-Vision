from PIL import Image
import depth_pro

class MetricDepthModel:
    
    def __init__(self):
        # Load model and preprocessing transform
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.eval()
        
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
        
        
        