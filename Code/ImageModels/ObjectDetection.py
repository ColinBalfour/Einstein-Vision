from ultralytics import YOLO
import cv2
import torch
import sys
import os
import numpy as np
from Features import *

from Features.Vehicle import Vehicle
from Features.Pedestrian import Pedestrian
from Features.TrafficLight import TrafficLight
from Features.RoadSign import RoadSign
from Features.Taillight import Taillight, TaillightDetector


class ObjectDetectionModel:
    YOLO_DEFAULT_CLASS_DETECTIONS = [
        'car',
        'person',
        'traffic light',
        'stop sign'
    ]

    ALL_CLASSES = [
        *YOLO_DEFAULT_CLASS_DETECTIONS,  # Default classes for YOLO
        'taillight',
        'brakelight',
        # Add more classes as per the model's capability
    ]

    def __init__(self, model_path='Code/ImageModels/yolo12x.pt', classes=None, conf_threshold=0.65,
                 enable_taillight_detection=False, detic_model_path=None):

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

        # Initialize taillight detector if enabled
        self.enable_taillight_detection = enable_taillight_detection
        self.taillight_detector = None

        if enable_taillight_detection:
            try:
                self.taillight_detector = TaillightDetector(
                    model_path=detic_model_path,
                    confidence_threshold=conf_threshold,
                    use_gpu=torch.cuda.is_available()
                )
                print("Taillight detector initialized successfully")
            except Exception as e:
                print(f"Error initializing taillight detector: {e}")
                self.enable_taillight_detection = False

    @staticmethod
    def viz_output(output):
        print()
        for class_name, detections in output.items():
            print(f"Class '{class_name}' has {len(detections)} detections:")
            for det in detections:
                print(f" - {type(det).__name__} with confidence: {det.confidence:.2f}")
        print()

    def infer(self, image_path, save=False):

        # Load image for later taillight detection if needed
        if self.enable_taillight_detection:
            image = cv2.imread(image_path) if isinstance(image_path, str) else image_path
            if image is None and isinstance(image_path, str):
                print(f"Warning: Could not read image {image_path} for taillight detection")

        # Run YOLO inference
        results = self.model(image_path, conf=self.conf_threshold)

        # Initialize output dictionary with all enabled classes
        output = {key: [] for key in self.classes}

        # Add taillight classes if enabled
        if self.enable_taillight_detection:
            if 'taillight' not in output and 'taillight' in self.classes:
                output['taillight'] = []
            if 'brakelight' not in output and 'brakelight' in self.classes:
                output['brakelight'] = []

        # Process YOLO results
        for r in results:
            # Save the image with detections
            if save and isinstance(image_path, str):
                r.save(filename=f"{image_path.split('.')[-1]}_detected.png")

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                print(f"Detected {class_name} with confidence {conf:.2f} at bbox: [{x1}, {y1}, {x2}, {y2}]")

                # Get depth for this object
                # Center point of the bounding box
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if class_name not in self.classes:
                    print(
                        f"Class '{class_name}' is not in the specified classes for this model. Must be in {self.classes}. Skipping...")
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
                    raise ValueError(
                        f"Something went wrong! Class {class_name} is not handled in the infer method. Likely caused by passing an invalid class name to this object at init.")

                # Append to the output dictionary
                print(f"Adding {class_name} to output with bbox: [{x1}, {y1}, {x2}, {y2}] and confidence: {conf:.2f}")
                output[class_name].append(obj)

        # Perform taillight detection if enabled
        if self.enable_taillight_detection and self.taillight_detector is not None:
            try:
                # Detect taillights using DETIC
                taillights = self.taillight_detector.detect(image if 'image' in locals() else image_path)

                # Add detections to output
                for taillight in taillights:
                    # Create appropriate class name based on detection
                    class_name = 'brakelight' if taillight.is_brake else 'taillight'

                    # Skip if this class is not in requested classes
                    if class_name not in self.classes:
                        continue

                    # Get image dimensions for center point calculation
                    if 'image' in locals() and image is not None:
                        h, w = image.shape[:2]
                    else:
                        # If no image is available, use a standard resolution
                        h, w = 1080, 1920

                    # Calculate center point
                    center_x = int(((taillight.x1 + taillight.x2) / 2) * w)
                    center_y = int(((taillight.y1 + taillight.y2) / 2) * h)

                    # Convert normalized coordinates to pixel values for visualization
                    x1 = int(taillight.x1 * w)
                    y1 = int(taillight.y1 * h)
                    x2 = int(taillight.x2 * w)
                    y2 = int(taillight.y2 * h)

                    print(
                        f"Adding {class_name} to output with bbox: [{x1}, {y1}, {x2}, {y2}] and confidence: {taillight.confidence:.2f}")

                    # Add to output dictionary
                    output[class_name].append(taillight)
            except Exception as e:
                print(f"Error during taillight detection: {e}")

        self.viz_output(output)
        return output

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
            'car': (0, 255, 0),  # green
            'person': (255, 0, 0),  # blue
            'traffic light': (0, 0, 255),  # red
            'stop sign': (255, 255, 0),  # cyan
            'taillight': (0, 255, 255),  # yellow
            'brakelight': (0, 0, 255)  # red (same as traffic light)
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
                # Get bounding box coordinates
                if hasattr(obj, 'bbox'):
                    # For YOLO detections (Vehicle, Pedestrian, etc.)
                    x1, y1, x2, y2 = obj.bbox
                else:
                    # For taillight detections
                    h, w = img.shape[:2]
                    x1 = int(obj.x1 * w)
                    y1 = int(obj.y1 * h)
                    x2 = int(obj.x2 * w)
                    y2 = int(obj.y2 * h)

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

                # Create a filled rectangle behind the text label
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

                # Mark the center point
                if hasattr(obj, 'center'):
                    center_x, center_y = obj.center
                else:
                    # Calculate center for taillights
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                cv2.circle(img, (int(center_x), int(center_y)), 4, color, -1)

        return img
