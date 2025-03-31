from ultralytics import YOLO
import cv2
import torch

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
    ]
    
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
    
    @staticmethod
    def viz_output(output):
        print()
        for class_name, detections in output.items():
            print(f"Class '{class_name}' has {len(detections)} detections:")
            for det in detections:
                print(f" - {type(det).__name__} with confidence: {det.confidence:.2f}")
        print()
        
    def infer(self, image_path, save=False):
        
        results = self.model(image_path, conf=self.conf_threshold)
        
        # Process results
        output = {key: [] for key in self.classes}
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
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
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
