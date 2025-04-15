
import sys
import cv2
import torch
import numpy as np

from ultralytics import YOLO


try:
    sys.path.insert(0, 'ImageModels/Detic/third_party/CenterNet2')
    from centernet.config import add_centernet_config

    sys.path.insert(0, 'ImageModels/Detic')
    from detic.config import add_detic_config
    from detic.modeling.text.text_encoder import build_text_encoder
    from detic.modeling.utils import reset_cls_test
except ImportError:
    print("Warning: Detic libraries not found. Make sure to install Detic dependencies.")

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


from Features.Vehicle import Vehicle
from Features.Pedestrian import Pedestrian
from Features.TrafficLight import TrafficLight
from Features.RoadSign import RoadSign

class ObjectDetectionModel:
    
    YOLO_DEFAULT_CLASS_DETECTIONS = [
        'car',
        'person',
        'traffic light',
        'stop sign'
    ]
    
    ALL_CLASSES = [
        *YOLO_DEFAULT_CLASS_DETECTIONS,  # Default classes for YOLO
        # Add more classes as per the model's capability
        "taillight",
        "brake light"
    ]
    
    def __init__(self, model_path='yolov12x.pt', classes=None, conf_threshold=0.65):
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.classes = classes
    
    @staticmethod
    def viz_output(output):
        print()
        for class_name, detections in output.items():
            print(f"Class '{class_name}' has {len(detections)} detections:")
            for det in detections:
                print(f" - {type(det).__name__} with confidence: {det.confidence:.2f}")
        print()
        
    # Default method to be overridden by subclasses
    def infer(self, image_path, save=False):
        return None
    
    def get_outputs(self, image_path, save=False):
        
        boxes, confidences, labels = self.infer(image_path, save=save)
        
        # Process results
        output = {key: [] for key in self.classes}
        for box, conf, class_name in zip(boxes, confidences, labels):
            x1, y1, x2, y2 = box
            print(f"Detected {class_name} with confidence {conf:.2f} at bbox: [{x1}, {y1}, {x2}, {y2}]")
            
            # Get depth for this object
            # Center point of the bounding box
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            if class_name not in self.classes:
                print(f"Class '{class_name}' is not in the specified classes for this model. Must be in {self.classes}. Skipping...")
                continue

            if class_name == 'car':
                # Process vehicle detection
                obj = Vehicle(
                    bbox=[x1, y1, x2, y2],
                    center=[center_x, center_y],
                    confidence=conf,
                    vehicle_type='car', 
                )
            
            elif class_name == 'person':
                # Process pedestrian detection
                obj = Pedestrian(
                    bbox=[x1, y1, x2, y2],
                    center=[center_x, center_y],
                    confidence=conf
                )
            
            elif class_name == 'traffic light':
                # Process traffic light detection
                obj = TrafficLight(
                    bbox=[x1, y1, x2, y2],
                    center=[center_x, center_y],
                    confidence=conf
                )
            
            elif class_name == 'stop sign':
                # Process road sign detection
                obj = RoadSign(
                    bbox=[x1, y1, x2, y2],
                    center=[center_x, center_y],
                    confidence=conf,
                    sign_type='STOP'
                )
            
            else:
                raise ValueError(f"Something went wrong! Class {class_name} is not handled in the infer method. Likely caused by passing an invalid class name to this object at init.")
            
            # Append to the output dictionary
            print(f"Adding {class_name} to output with bbox: [{x1}, {y1}, {x2}, {y2}] and confidence: {conf:.2f}")
            output[class_name].append(obj)
            self.viz_output(output) 

        self.viz_output(output) 
        return output
        
        
    # GPT Prompt: write a visualizer for the above code
    def visualize(self, image, output):
        """
        Draws bounding boxes and labels onto the given image using the detection outputs.
        
        Parameters:
            img (np.ndarray): The original image to draw on (OpenCV image).
            output (dict): A dictionary of detections keyed by class name, where each
                        value is a list of detection objects. For example:
                        {
                            'car': [Vehicle(...), Vehicle(...)],
                            'person': [Pedestrian(...), Pedestrian(...)],
                            ...
                        }
        
        Returns:
            np.ndarray: The image with bounding boxes and labels drawn.
        """
        img = image.copy()  # Make a copy of the image to draw on
        
        # You can define distinct colors for each class if you want
        # Here is an optional color map for illustration:
        class_color_map = {
            'car': (0, 255, 0),             # green
            'person': (255, 0, 0),          # blue
            'traffic light': (0, 0, 255),   # red
            'stop sign': (255, 255, 0)      # cyan
        }
        
        # Font configuration for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Iterate over each class's detections
        for class_name, detections in output.items():
            # If there are no detections for this class, skip
            if not detections:
                print(f"No detections for class '{class_name}'. Skipping visualization.")
                continue
            
            # Pick a color for the current class. Use a default if class not in map.
            color = class_color_map.get(class_name, (0, 255, 255))
            
            print(f"Visualizing {len(detections)} detections for class '{class_name}'")
            
            # Draw each detected object
            for obj in detections:
                # Retrieve bounding box
                x1, y1, x2, y2 = obj.bbox
                # Draw rectangle
                cv2.rectangle(img, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            color, 
                            thickness)
                
                # Prepare label text (class name + confidence)
                label = f"{class_name} {obj.confidence:.2f}"
                
                # Calculate text size to create a background for better visibility
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Optionally create a filled rectangle behind the text label
                cv2.rectangle(img, 
                            (int(x1), int(y1) - text_height - baseline),
                            (int(x1) + text_width, int(y1)), 
                            color, 
                            -1)
                
                # Put the text label slightly above the top-left corner of the bounding box
                cv2.putText(img, 
                            label, 
                            (int(x1), int(y1) - baseline),
                            font, 
                            font_scale, 
                            (0, 0, 0),  # text color (black) for contrast
                            thickness)
                
                # If you also want to visualize the 'center' attribute, you can mark it:
                center_x, center_y = obj.center
                cv2.circle(img, (center_x, center_y), 4, color, -1)
        
        return img
    
    
class YOLODetector(ObjectDetectionModel):
    
    def __init__(self, model_path='yolov12x.pt', classes=None, conf_threshold=0.65):
        
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to('cuda')  # Move model to GPU
            
        self.model.predict()
        
        self.conf_threshold = conf_threshold
        
        self.classes = classes 
        if classes is None:
            self.classes = self.YOLO_DEFAULT_CLASS_DETECTIONS
        if any([c not in self.ALL_CLASSES for c in self.classes]):
            raise ValueError(f"Provided classes {self.classes} are not valid. Must be a subset of {self.ALL_CLASSES}")
    
    
    def infer(self, image_path, save=False):
        
        results = self.model(image_path, conf=self.conf_threshold)
        
        # Process results
        boxes = [] # x1, y1, x2, y2
        confidences = [] # confidence
        labels = [] # class name
        for r in results:
            # Save the image with detections
            if save:
                r.save(filename=f"{image_path.split('.')[-1]}_detected.png") 
            
            # Extract masks, boxes, labels, and confidences
            # masks = r.masks.xy
            # boxes = r.boxes.xyxy  
            # labels = r.boxes.cls
            # confidences = r.boxes.conf
            
            for box in r.boxes:
                x1, y1, x2, y2 = box_xy = box.xyxy[0].tolist()
                
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                # print(f"Detected {class_name} with confidence {conf:.2f} at bbox: [{x1}, {y1}, {x2}, {y2}]")
                
                boxes.append(box_xy)
                confidences.append(conf)
                labels.append(class_name)
    
        return boxes, confidences, labels
    
    
    
class DeticDectector(ObjectDetectionModel):
    
    def __init__(self, model_path=None, classes=None, conf_threshold=0.3, config_path=None, vocabulary=None):
        """
        Initialize the taillight detector

        Args:
            model_path (str): Path to DETIC model weights
            config_path (str): Path to DETIC config file
            confidence_threshold (float): Confidence threshold for detections
            use_gpu (bool): Whether to use GPU for inference
        """
        # icky to read as kwargs because its so long lol
        model_path = model_path or "ImageModels/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        config_path = config_path or "ImageModels/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
        
        super().__init__(model_path=model_path, classes=classes, conf_threshold=conf_threshold)
        
        self.predictor = None

        # Vocabulary for taillight detection
        if list(vocabulary.keys()) != classes:
            raise ValueError(f"Provided classes {classes} do not match the vocabulary keys {list(vocabulary.keys())}.")
        self.vocabulary = vocabulary
        self.config_path = config_path

        # Initialize the model
        self._setup_model()

    def _setup_model(self):
        """
        Set up the DETIC model
        """
        try:
            # Create config
            cfg = get_cfg()
            add_centernet_config(cfg)
            add_detic_config(cfg)

            # Load config from file
            cfg.merge_from_file(self.config_path)

            # Set model weights
            cfg.MODEL.WEIGHTS = self.model_path

            # Configure model parameters
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf_threshold
            cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
            cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
            cfg.MODEL.ROI_HEADS.USE_ZEROSHOT_CLS = True

            # Set device
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            # Create predictor
            self.predictor = DefaultPredictor(cfg)

            # Set custom vocabulary
            self._set_vocabulary(self.vocabulary)

            print("Detic model initialized successfully")
        except Exception as e:
            print(f"Error setting up DETIC model: {e}")
            self.predictor = None
        
    def _set_vocabulary(self, vocabulary):
        """
        Set custom vocabulary for taillight detection
        """
        if self.predictor is None:
            return
        
        # Ensure vocabulary is a list of strings (self.vocabulary is a dict mapping class names to vocab)
        vocabulary = np.array(list(vocabulary.values())).flatten()

        try:
            # Build text encoder
            text_encoder = build_text_encoder(self.predictor.model.cfg)

            # Get text features
            text_features = text_encoder(vocabulary).float()

            # Reset classifier
            reset_cls_test(self.predictor.model, text_features)
        except Exception as e:
            print(f"Error setting vocabulary: {e}")
            
    def class_name_from_vocab(self, vocab):
        for class_name, vocab_list in self.vocabulary.items():
            if vocab in vocab_list:
                return class_name
        return None 

        
    
    def infer(self, image, save=False):
        """
        Detect taillights and brakelights in an image

        Args:
            image: Image as numpy array or path to image file

        Returns:
            list: List of Taillight objects
        """
        if self.predictor is None:
            print("Predictor not initialized. Cannot perform detection.")
            return []

        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                print(f"Could not read image: {image}")
                return []

        # Get image dimensions
        height, width = image.shape[:2]

        boxes = []
        confidences = []
        labels = []
        try:
            # Run inference
            outputs = self.predictor(image)

            # Process predictions
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
            scores = instances.scores.numpy() if instances.has("scores") else []
            classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []

            for box, score, cls_id in zip(boxes, scores, classes):
                # Get class name
                vocab_name = self.vocabulary[cls_id] if cls_id < len(self.vocabulary) else "unknown"

                # Normalize coordinates
                x1, y1, x2, y2 = box

                class_name = self.class_name_from_vocab(vocab_name)
                if class_name is None:
                    print(f"Class name not found for vocab: {vocab_name}. Skipping...")
                    continue
                
                boxes.append(box)
                confidences.append(score)
                labels.append(class_name)
            
            return boxes, confidences, labels

        except Exception as e:
            print(f"Error during detection: {e}")
            return []