from ultralytics import YOLO
import os
import glob

from Features.Object import Object

class Pedestrian(Object):
    """
    This class represents a pedestrian detected by the YOLO model.
    It inherits from the Object class and can be used to represent a pedestrian object.
    """

    def __init__(self, bbox, center, confidence, pose=None):
        # Initialize the Object class with the bounding box, center, confidence, and pose
        super().__init__(bbox, center, confidence, pose)

    def __repr__(self):
        return f"Pedestrian(bbox={self.bbox}, center={self.center}, confidence={self.confidence}, pose={self.pose})"

    def __str__(self):
        return self.__repr__()

if __name__ == "__main__":
    # Load model
    model = YOLO('yolo11x.pt')
    # Input and output directories
    data_dir = 'Code/P3Data'
    output_dir = 'Code/outputs/scene_1'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each frame in the directory
    for frame_file in os.listdir(data_dir):
        if frame_file.endswith(('.jpg', '.png', '.jpeg')):
            # Full path to the frame
            frame_path = os.path.join(data_dir, frame_file)

            # Run inference
            results = model(frame_path, conf = 0.65)

            # Save results with visualization
            output_path = os.path.join(output_dir, frame_file)

            # Plot results and save
            for r in results:
                im_array = r.plot()  # Plot with detections
                boxes = r.boxes
                masks = r.masks
                keypoints = r.keypoints
                probs = r.probs
                obb = r.obb
                r.save(filename=output_path)  # Save image with detections

            print(f"Processed {frame_file}")
