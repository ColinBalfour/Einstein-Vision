import sys
import cv2
import torch
import numpy as np
import time

from ultralytics import YOLO

try:
    sys.path.insert(0, 'Detic/third_party/CenterNet2')
    from centernet.config import add_centernet_config

    sys.path.insert(0, 'Detic')
    from detic.config import add_detic_config
    from detic.modeling.text.text_encoder import build_text_encoder
    from detic.modeling.utils import reset_cls_test
except ImportError:
    print("Warning: Detic libraries not found. Make sure to install Detic dependencies.")

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


# Import your Feature classes
from Features.Vehicle import Vehicle
from Features.Pedestrian import Pedestrian
from Features.TrafficLight import TrafficLight
from Features.RoadSign import RoadSign
from Features.Taillight import Taillight


class ObjectDetectionModel:
    """
    Base class for object detection pipelines, with a default get_outputs() 
    and visualize() method that can be used by YOLODetector or DeticDectector.
    """

    YOLO_DEFAULT_CLASS_DETECTIONS = [
        'car',
        'truck',
        'person',
        'traffic light',
        'stop sign'
    ]
    
    ALL_CLASSES = [
        *YOLO_DEFAULT_CLASS_DETECTIONS,  # Default classes for YOLO
        # Add more classes as per the model's capability
        "turn_signal",
        "brake_light"
    ]
    
    # Optional color map (RGB format)
    CLASS_COLOR_MAP = {
        'car': (0, 255, 0),            # green
        'pickup': (0, 255, 0),        # green
        'person': (0, 0, 255),         # blue
        'traffic light': (255, 255, 255),  # white
        'stop sign': (0, 255, 255),    # cyan
        'speed_limit': (255, 255, 255),# white
        'brake': (255, 0, 0),    # red
        'turn': (255, 255, 0),  # yellow
        'taillight': (255, 255, 255),  # white
        
        'green': (0, 255, 0),  # green
        'yellow': (0, 255, 255),  # yellow
        'red': (0, 0, 255),  # red
    }
    
    def __init__(self, model_path='yolov12x.pt', classes=None, conf_threshold=0.65):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.classes = classes
    
    @staticmethod
    def viz_output(output):
        """Utility for printing how many detections per class were found."""
        print()
        for class_name, detections in output.items():
            print(f"Class '{class_name}' has {len(detections)} detections:")
            for det in detections:
                print(f" - {type(det).__name__} with confidence: {det.confidence:.2f}")
        print()
        
    def infer(self, img, image_path="", save=False):
        """
        Default method to be overridden by subclasses.
        Must return (boxes, centers, masks, confidences, labels).
        """
        return [], [], [], [], []
    
    def get_outputs(self, img, image_path="", save=False):
        """
        1) Runs inference -> gets (boxes, centers, masks, confidences, labels).
        2) Instantiates the appropriate Feature objects (Vehicle, Pedestrian, etc.).
        3) Returns a dictionary keyed by class name.
        """
        boxes, centers, masks, confidences, labels = self.infer(img, image_path=image_path, save=save)
        
        # Prepare output: dictionary of lists for each class
        output = {key: [] for key in self.classes}
        
        # Build Feature objects
        for box, center, mask, conf, class_name in zip(boxes, centers, masks, confidences, labels):
            x1, y1, x2, y2 = box
            center_x, center_y = center
            print(f"Detected '{class_name}' with confidence {conf:.2f} at bbox: [{x1}, {y1}, {x2}, {y2}]")
            
            if class_name not in self.classes:
                print(f"Class '{class_name}' is not in the specified classes. Skipping.")
                continue

            if class_name in ['car', 'truck', 'SUV', 'bicycle', 'pickup', 'motorcycle']:
                obj = Vehicle(
                    bbox=[x1, y1, x2, y2],
                    center=[center_x, center_y],
                    mask=mask,
                    confidence=conf,
                    vehicle_type=class_name
                )
            elif class_name == 'person':
                obj = Pedestrian(
                    bbox=[x1, y1, x2, y2],
                    center=[center_x, center_y],
                    confidence=conf
                )
            elif class_name == 'traffic light':
                obj = TrafficLight(
                    bbox=[x1, y1, x2, y2],
                    center=[center_x, center_y],
                    confidence=conf
                )
            elif class_name == 'stop sign':
                obj = RoadSign(
                    bbox=[x1, y1, x2, y2],
                    center=[center_x, center_y],
                    confidence=conf,
                    sign_type='STOP'
                )
            elif class_name == 'speed_limit':
                obj = RoadSign(
                    bbox=[x1, y1, x2, y2],
                    center=[center_x, center_y],
                    confidence=conf,
                    sign_type='SPEED_LIMIT'
                )
            elif class_name in ['brake_light', 'turn_signal', 'taillight']:
                obj = Taillight(
                    bbox=[x1, y1, x2, y2],
                    center=[center_x, center_y],
                    confidence=conf,
                    light_type=class_name
                )
            
            else:
                raise ValueError(
                    f"Unhandled class '{class_name}' in infer. Possibly invalid configuration."
                )
            
            print(f"Adding '{class_name}' to output with bbox: [{x1}, {y1}, {x2}, {y2}] "
                  f"and confidence {conf:.2f}")
            output[class_name].append(obj)
            self.viz_output(output) 

        self.viz_output(output) 
        return output
        
    def visualize(self, image, output):
        """
        Draw bounding boxes, labels, centers, and segmentation masks on the image.
        
        :param image: np.ndarray, the original BGR image
        :param output: dict, e.g. {'car': [Vehicle(), ...], 'person': [Pedestrian(), ...]}
        :return: np.ndarray with drawings
        """
        img = image.copy()  # Work on a copy
        
        # Go through each class's list of detections
        for class_name, detections in output.items():
            if not detections:
                continue
            
            if detections[0].__class__.__name__ == 'Vehicle':
                for obj in detections:
                    if obj.left_taillight:
                        # print(f"Left taillight detected: {obj.left_taillight}")
                        self.draw_viz(img, obj.left_taillight, obj.left_taillight.light_type.split('_')[0])
                    if obj.right_taillight:
                        # print(f"Right taillight detected: {obj.right_taillight}")
                        self.draw_viz(img, obj.right_taillight, obj.left_taillight.light_type.split('_')[0])
            
            for obj in detections:
                if class_name == 'traffic light' and obj.color:
                    print(f"Traffic light color detected: {obj.color}")
                    class_name = obj.color
                
                self.draw_viz(img, obj, class_name)
        
        return img

    def draw_viz(self, img, obj, class_name):
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        def apply_mask(image_bgr, mask_bool, color, alpha=0.4):
            """
            Blend a single-channel boolean mask into 'image_bgr' using 'color'.
            mask_bool: 2D boolean array
            color: (B, G, R) 
            alpha: blending factor
            """
            for c in range(3):
                image_bgr[mask_bool, c] = \
                    image_bgr[mask_bool, c] * (1 - alpha) + alpha * color[c]
        
        color = self.CLASS_COLOR_MAP.get(class_name, (0, 255, 255))  # fallback color
        
        x1, y1, x2, y2 = obj.bbox
        cx, cy = obj.center
                
        # Draw bounding box
        cv2.rectangle(
            img, 
            (int(x1), int(y1)), 
            (int(x2), int(y2)), 
            color, 
            thickness
        )
        
        # Label text
        label = f"{class_name} {obj.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Filled rectangle behind label
        cv2.rectangle(
            img, 
            (int(x1), int(y1) - text_h - baseline),
            (int(x1) + text_w, int(y1)),
            color,
            -1
        )
        
        # Put the label
        cv2.putText(
            img, 
            label,
            (int(x1), int(y1) - baseline),
            font,
            font_scale,
            (0, 0, 0),  # black text
            thickness
        )
        
        # Draw center
        cv2.circle(img, (int(cx), int(cy)), 4, color, -1)

        # If there's a mask, blend it
        if hasattr(obj, "mask") and obj.mask is not None:
            apply_mask(img, obj.mask, color=color, alpha=0.4)

        

class YOLODetector(ObjectDetectionModel):
    """
    A YOLO-based detector for older YOLO versions with .xyn (polygon) data 
    instead of .masks or .masks.data.
    """
    
    def __init__(self, model_path='yolov11x.pt', classes=None, conf_threshold=0.65):
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to('cuda')
        # Force a predict call so the model is "ready"
        self.model.predict()
        
        self.conf_threshold = conf_threshold
        
        # Validate classes
        if classes is None:
            classes = self.YOLO_DEFAULT_CLASS_DETECTIONS
        invalid = [c for c in classes if c not in self.ALL_CLASSES]
        if invalid:
            raise ValueError(
                f"Classes {invalid} not in known list. Must be subset of {self.ALL_CLASSES}"
            )
        
        self.classes = classes

    def infer(self, img, image_path="", save=False):
        """
        1) Reads the image for shape (H, W).
        2) Runs YOLO inference with conf threshold.
        3) Extracts polygons from r.masks.xyn, fill them onto a blank mask with cv2.fillPoly.
        4) Calculates centroid from mask if not empty, else uses bounding box center.
        5) Returns: (boxes, centers, masks, confidences, labels)
        """
        # Read original image
        if isinstance(img, str):
            image_path = img
            img = cv2.imread(img)
        
        if image_path == "":
            save = False
            
        if img is None:
            raise ValueError(f"Could not read image: {img}")
        
        
        H, W = img.shape[:2]

        results = self.model(img, conf=self.conf_threshold)

        # Prepare lists
        boxes = []
        centers = []
        masks = []
        confidences = []
        labels = []

        for r in results:
            if save:
                # Save YOLO's bounding-box annotated image
                r.save(filename=f"{image_path.rsplit('.', 1)[0]}_detected.png")

            # If YOLO can't produce polygons, set to empty
            all_poly_segments = r.masks.xyn if (r.masks and r.masks.xyn is not None) else []

            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box_xy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                class_name = self.model.names[cls_idx]

                # Default center: bounding box
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Build a blank mask
                mask_filled = np.zeros((H, W), dtype=np.uint8)

                # If i < len(all_poly_segments), retrieve polygon data 
                if i < len(all_poly_segments):
                    poly_data = all_poly_segments[i]

                    # Case 1: poly_data is a list of polygons
                    if isinstance(poly_data, list):
                        for single_poly in poly_data:
                            # single_poly should be an np.ndarray of shape (N, 2)
                            if (isinstance(single_poly, np.ndarray) and
                                single_poly.ndim == 2 and
                                single_poly.shape[1] == 2):
                                pts = []
                                for (xn, yn) in single_poly:
                                    px = int(xn * W)
                                    py = int(yn * H)
                                    pts.append([px, py])
                                pts_arr = np.array(pts, dtype=np.int32)
                                cv2.fillPoly(mask_filled, [pts_arr], 1)
                            else:
                                # Unexpected shape; skip
                                pass

                    # Case 2: poly_data is a single polygon array
                    elif isinstance(poly_data, np.ndarray):
                        # e.g. shape (N, 2)
                        if poly_data.ndim == 2 and poly_data.shape[1] == 2:
                            pts = []
                            for (xn, yn) in poly_data:
                                px = int(xn * W)
                                py = int(yn * H)
                                pts.append([px, py])
                            pts_arr = np.array(pts, dtype=np.int32)
                            cv2.fillPoly(mask_filled, [pts_arr], 1)
                        else:
                            # Not an array of shape (N,2)
                            pass
                    else:
                        # poly_data is neither list nor ndarray
                        pass

                # If mask_filled is not empty, compute centroid
                ys, xs = np.where(mask_filled == 1)
                if len(xs) > 0:
                    cx = float(np.mean(xs))
                    cy = float(np.mean(ys))

                center = (int(cx), int(cy))
                mask_bool = (mask_filled == 1) if np.any(mask_filled) else None

                # Store results
                boxes.append(box_xy)
                centers.append(center)
                masks.append(mask_bool)
                confidences.append(conf)
                labels.append(class_name)

        return boxes, centers, masks, confidences, labels


class DeticDectector(ObjectDetectionModel):
    """
    Optional class for DETIC-based detection. 
    Left largely unchanged except for ensuring we return the 
    (boxes, centers, masks, confidences, labels) shape.
    """
    
    def __init__(self, model_path=None, vocabulary=None, conf_threshold=0.3, config_path=None):
        model_path = model_path or "Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        config_path = config_path or "Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
        classes = vocabulary.keys() if vocabulary else None
        
        super().__init__(model_path=model_path, classes=classes, conf_threshold=conf_threshold)
        
        self.predictor = None
        self.vocabulary = vocabulary
        self.flattened_vocab = sum(list(self.vocabulary.values()), [])
        self.config_path = config_path
        self._setup_model()

    def _setup_model(self):
        # try:
            cfg = get_cfg()
            add_centernet_config(cfg)
            add_detic_config(cfg)
            cfg.merge_from_file(self.config_path)
            cfg.MODEL.WEIGHTS = self.model_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf_threshold
            cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
            cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
            # cfg.MODEL.ROI_HEADS.USE_ZEROSHOT_CLS = True
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            cfg.freeze()
            
            self.predictor = DefaultPredictor(cfg)
            self.cfg = cfg

            self._set_vocabulary(self.vocabulary)
            print("Detic model initialized successfully")
        # except Exception as e:
        #     print(f"Error setting up DETIC model: {e}")
        #     self.predictor = None
        
    def _set_vocabulary(self, vocabulary):
        if self.predictor is None or not vocabulary:
            return

        # try:
        metadata = MetadataCatalog.get(str(time.time()))
        metadata.thing_classes = self.flattened_vocab
        classifier = get_clip_embeddings(metadata.thing_classes)
        num_classes = len(metadata.thing_classes)
        reset_cls_test(self.predictor.model, classifier, num_classes)
        # except Exception as e:
        #     print(f"Error setting vocabulary: {e}")
            
    def class_name_from_vocab(self, vocab_item):
        for class_name, vocab_list in self.vocabulary.items():
            if vocab_item in vocab_list:
                return class_name
        return None 
    
    def infer(self, image, image_path="", save=False):
        """
        Returns the standard shape (boxes, centers, masks, confidences, labels).
        Detic might produce masks if you enable them; here we store None for now.
        """
        if self.predictor is None:
            print("Predictor not initialized. Cannot perform detection.")
            return [], [], [], [], []

        if isinstance(image, str):
            image_path = image
            image = cv2.imread(image)
            if image is None:
                print(f"Could not read image: {image_path}")
                return [], [], [], [], []

        boxes_out = []
        centers_out = []
        masks_out = []
        confidences_out = []
        labels_out = []

        # try:
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        
        pred_boxes_np = (instances.pred_boxes.tensor.numpy()
                        if instances.has("pred_boxes") else [])
        pred_scores_np = (instances.scores.numpy()
                        if instances.has("scores") else [])
        pred_classes_np = (instances.pred_classes.numpy()
                        if instances.has("pred_classes") else [])
        
        # If the model had pred_masks, you could store them here too:
        # pred_masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []

        for box, score, cls_id in zip(pred_boxes_np, pred_scores_np, pred_classes_np):
            x1, y1, x2, y2 = box
            
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if cls_id < len(self.flattened_vocab):
                vocab_name = self.flattened_vocab[cls_id]
            else:
                vocab_name = "unknown"

            class_name = self.class_name_from_vocab(vocab_name)
            if class_name is None:
                print(f"Class name not found for vocab: {vocab_name}. Skipping...")
                continue
            
            boxes_out.append([int(x1), int(y1), int(x2), int(y2)])
            centers_out.append((int(cx), int(cy)))
            masks_out.append(None)  # or a real mask if available
            confidences_out.append(float(score))
            labels_out.append(class_name)
        
        return boxes_out, centers_out, masks_out, confidences_out, labels_out

        # except Exception as e:
        #     print(f"Error during DETIC detection: {e}")
        #     return [], [], [], [], []
