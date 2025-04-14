#!/usr/bin/env python3
# Taillight.py - Class for taillight and brakelight detection using DETIC

import os
import json
import cv2
import numpy as np
import sys


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


class Taillight:
    """
    Class to represent a taillight or brakelight in a vehicle
    """

    def __init__(self, class_name="taillight", confidence=0.0,
                 x1=0.0, y1=0.0, x2=0.0, y2=0.0, is_brake=False):
        """
        Initialize a taillight instance

        Args:
            class_name (str): The class name (taillight or brakelight)
            confidence (float): Detection confidence
            x1, y1 (float): Top-left coordinates (normalized 0-1)
            x2, y2 (float): Bottom-right coordinates (normalized 0-1)
            is_brake (bool): Whether this is a brakelight
        """
        self.class_name = class_name
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.is_brake = is_brake

        # Computed properties
        self.width = x2 - x1
        self.height = y2 - y1
        self.center_x = (x1 + x2) / 2
        self.center_y = (y1 + y2) / 2

    def get_center(self):
        """
        Get the center point of the taillight

        Returns:
            list: [center_x, center_y] coordinates
        """
        return [self.center_x, self.center_y]

    def to_dict(self):
        """
        Convert the taillight to a dictionary for JSON serialization
        """
        return {
            "class": self.class_name,
            "is_brake": self.is_brake,
            "confidence": self.confidence,
            "bbox": {
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2
            },
            "center": {
                "x": self.center_x,
                "y": self.center_y
            },
            "dimensions": {
                "width": self.width,
                "height": self.height
            }
        }

    def to_json(self, image_path=None):
        """
        Create a JSON representation of the taillight

        Args:
            image_path (str, optional): Path to the image

        Returns:
            dict: JSON-serializable dictionary
        """
        json_output = self.to_dict()

        if image_path:
            json_output["image_path"] = image_path

        return json_output

    @classmethod
    def from_dict(cls, data):
        """
        Create a Taillight instance from a dictionary
        """
        return cls(
            class_name=data.get("class", "taillight"),
            confidence=data.get("confidence", 0.0),
            x1=data.get("bbox", {}).get("x1", 0.0),
            y1=data.get("bbox", {}).get("y1", 0.0),
            x2=data.get("bbox", {}).get("x2", 0.0),
            y2=data.get("bbox", {}).get("y2", 0.0),
            is_brake=data.get("is_brake", False)
        )


class TaillightDetector:
    """
    Class for detecting taillights and brakelights in images using DETIC
    """

    def __init__(self, model_path=None, config_path=None, confidence_threshold=0.3, use_gpu=True):
        """
        Initialize the taillight detector

        Args:
            model_path (str): Path to DETIC model weights
            config_path (str): Path to DETIC config file
            confidence_threshold (float): Confidence threshold for detections
            use_gpu (bool): Whether to use GPU for inference
        """
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        self.predictor = None

        # Set default paths if not provided
        self.model_path = model_path or "ImageModels/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        self.config_path = config_path or "ImageModels/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"

        # Vocabulary for taillight detection
        self.vocabulary = [
            'taillight',
            'tail light',
            'brake light',
            'brakelight',
            'rear light',
            'car taillight',
            'vehicle light'
        ]

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
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
            cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
            cfg.MODEL.ROI_HEADS.USE_ZEROSHOT_CLS = True

            # Set device
            cfg.MODEL.DEVICE = "cuda" if self.use_gpu else "cpu"

            # Create predictor
            self.predictor = DefaultPredictor(cfg)

            # Set custom vocabulary
            self._set_vocabulary(self.vocabulary)

            print("TaillightDetector initialized successfully")
        except Exception as e:
            print(f"Error setting up DETIC model: {e}")
            self.predictor = None

    def _set_vocabulary(self, vocabulary):
        """
        Set custom vocabulary for taillight detection
        """
        if self.predictor is None:
            return

        try:
            # Build text encoder
            text_encoder = build_text_encoder(self.predictor.model.cfg)

            # Get text features
            text_features = text_encoder(vocabulary).float()

            # Reset classifier
            reset_cls_test(self.predictor.model, text_features)
        except Exception as e:
            print(f"Error setting vocabulary: {e}")

    def detect(self, image):
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

        try:
            # Run inference
            outputs = self.predictor(image)

            # Process predictions
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
            scores = instances.scores.numpy() if instances.has("scores") else []
            classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []

            # Create taillight objects
            taillights = []
            for box, score, cls_id in zip(boxes, scores, classes):
                # Get class name
                class_name = self.vocabulary[cls_id] if cls_id < len(self.vocabulary) else "unknown"

                # Normalize coordinates
                x1, y1, x2, y2 = box
                rel_x1 = float(x1 / width)
                rel_y1 = float(y1 / height)
                rel_x2 = float(x2 / width)
                rel_y2 = float(y2 / height)

                # Determine if it's a brakelight based on class name
                is_brake = "brake" in class_name.lower()

                # Create taillight object
                taillight = Taillight(
                    class_name=class_name,
                    confidence=float(score),
                    x1=rel_x1,
                    y1=rel_y1,
                    x2=rel_x2,
                    y2=rel_y2,
                    is_brake=is_brake
                )

                taillights.append(taillight)

            return taillights

        except Exception as e:
            print(f"Error during detection: {e}")
            return []

    def detect_and_visualize(self, image, output_path=None):
        """
        Detect taillights and brakelights in an image and visualize the results

        Args:
            image: Image as numpy array or path to image file
            output_path (str, optional): Path to save visualization

        Returns:
            list: List of Taillight objects
            numpy.ndarray: Visualization image
        """
        # Load image if path is provided
        if isinstance(image, str):
            original_image = cv2.imread(image)
            vis_image = original_image.copy() if original_image is not None else None
        else:
            original_image = image
            vis_image = image.copy() if image is not None else None

        if vis_image is None:
            print("Could not read image for visualization")
            return [], None

        # Detect taillights
        taillights = self.detect(original_image)

        # Get image dimensions
        height, width = vis_image.shape[:2]

        # Draw bounding boxes
        for taillight in taillights:
            # Convert normalized coordinates to pixel coordinates
            x1 = int(taillight.x1 * width)
            y1 = int(taillight.y1 * height)
            x2 = int(taillight.x2 * width)
            y2 = int(taillight.y2 * height)

            # Choose color based on taillight type (red for brakelight, yellow for taillight)
            color = (0, 0, 255) if taillight.is_brake else (0, 255, 255)

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{taillight.class_name}: {taillight.confidence:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save visualization if output path is provided
        if output_path is not None:
            cv2.imwrite(output_path, vis_image)

        return taillights, vis_image

    def export_to_json(self, taillights, output_path, image_path=None):
        """
        Export taillight detections to JSON

        Args:
            taillights (list): List of Taillight objects
            output_path (str): Path to save JSON file
            image_path (str, optional): Path to the original image
        """
        # Create data structure
        data = {
            "image_path": image_path,
            "detections": [taillight.to_dict() for taillight in taillights]
        }

        # Write to file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Results exported to {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Taillight Detection")
    parser.add_argument("--input", required=True, help="Path to input image or directory")
    parser.add_argument("--output", default="taillight_detections.json", help="Path to output JSON file")
    parser.add_argument("--visualize", action="store_true", help="Visualize detections")
    parser.add_argument("--output-dir", default="output", help="Directory to save visualization results")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    # Initialize detector
    detector = TaillightDetector(confidence_threshold=args.confidence, use_gpu=not args.cpu)

    # Create output directory if it doesn't exist
    if args.visualize and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Process input
    if os.path.isfile(args.input):
        # Process single image
        image_path = args.input
        if args.visualize:
            output_path = os.path.join(args.output_dir, os.path.basename(image_path))
            taillights, _ = detector.detect_and_visualize(image_path, output_path)
        else:
            taillights = detector.detect(image_path)

        # Export to JSON
        detector.export_to_json(taillights, args.output, image_path)
    elif os.path.isdir(args.input):
        # Process directory of images
        all_taillights = {}
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(args.input, filename)
                print(f"Processing {image_path}")

                if args.visualize:
                    output_path = os.path.join(args.output_dir, filename)
                    taillights, _ = detector.detect_and_visualize(image_path, output_path)
                else:
                    taillights = detector.detect(image_path)

                all_taillights[image_path] = [t.to_dict() for t in taillights]

        # Export all results to JSON
        with open(args.output, 'w') as f:
            json.dump(all_taillights, f, indent=4)

        print(f"All results exported to {args.output}")
    else:
        print(f"Input path {args.input} does not exist")
