from ultralytics import YOLO
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


class PoseEstimationModel:
    def __init__(self, model_path='D:/3. Computer vision/Homeworks/Einstein-Vision/Code/ImageModels/yolo11x-pose.pt', conf_threshold=0.4):
        """
        Initialize the pose estimation model

        Args:
            model_path: Path to the YOLO pose model
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to('cuda')


        self.model.predict(torch.zeros((1, 3, 64, 64), device=self.model.device))
        self.conf_threshold = conf_threshold

    def infer(self, image_path, show=False, save_path=None):
        """
        Run pose estimation on an image

        Args:
            image_path: Path to the image or an image array
            show: Whether to display the results
            save_path: Directory to save visualization results

        Returns:
            List of keypoints for each detected person
        """
        # Run pose estimation
        results = self.model(image_path, conf=self.conf_threshold)

        # Process results
        pose_data = []
        for r in results:
            if hasattr(r, 'keypoints') and r.keypoints is not None:
                # Extract keypoints
                for i, keypoints in enumerate(r.keypoints.data):
                    # Get bounding box if available
                    if hasattr(r, 'boxes') and len(r.boxes) > i:
                        box = r.boxes[i].xyxy[0].tolist()
                        conf = float(r.boxes[i].conf[0])
                    else:
                        # Calculate bounding box from keypoints
                        valid_kpts = keypoints[keypoints[:, 2] > 0]
                        if len(valid_kpts) > 0:
                            x1, y1 = valid_kpts[:, 0].min(), valid_kpts[:, 1].min()
                            x2, y2 = valid_kpts[:, 0].max(), valid_kpts[:, 1].max()
                            box = [x1, y1, x2, y2]
                            conf = np.mean(keypoints[:, 2])
                        else:
                            continue

                    # Calculate center
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2

                    # Process keypoints into list format
                    kpts_list = keypoints.tolist()

                    # Add to results
                    pose_data.append({
                        'keypoints': kpts_list,
                        'bbox': box,
                        'center': [int(center_x), int(center_y)],
                        'confidence': conf
                    })

        # Visualize if requested
        if show or save_path:
            # Load the image if path is given
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = image_path.copy()

            vis_img = self.visualize(img, pose_data)

            if show:
                plt.figure(figsize=(12, 8))
                plt.imshow(vis_img)
                plt.title("Detected Poses")
                plt.show()

            if save_path:
                save_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                filename = f"{save_path}/pose_detection.png"
                cv2.imwrite(filename, save_img)
                print(f"Saved pose visualization to {filename}")

        return pose_data

    def visualize(self, image, pose_data):
        """
        Draw pose keypoints and skeletons on image

        Args:
            image: Image to draw on
            pose_data: List of pose dictionaries

        Returns:
            Image with poses drawn
        """
        img = image.copy()

        # Define COCO skeleton connections
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
        ]

        # Draw each pose
        for pose in pose_data:
            keypoints = np.array(pose['keypoints'])

            # Draw skeleton lines first (so they're behind the points)
            for limb in skeleton:
                idx1, idx2 = limb[0] - 1, limb[1] - 1
                if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                        keypoints[idx1, 2] > 0.4 and keypoints[idx2, 2] > 0.4):
                    pt1 = (int(keypoints[idx1, 0]), int(keypoints[idx1, 1]))
                    pt2 = (int(keypoints[idx2, 0]), int(keypoints[idx2, 1]))

                    cv2.line(img, pt1, pt2, (0, 0, 255), 2)

            # Draw keypoints
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.4:  # Only draw high-confidence keypoints
                    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

        return img

    def match_poses_to_pedestrians(self, pose_data, pedestrians):
        """
        Match detected poses to pedestrian objects using IoU

        Args:
            pose_data: List of pose dictionaries from infer()
            pedestrians: List of Pedestrian objects

        Returns:
            Dictionary mapping pedestrian objects to keypoints
        """
        matches = {}

        for person in pedestrians:
            person_bbox = person.bbox
            best_match = None
            best_iou = 0

            for pose in pose_data:
                pose_bbox = pose['bbox']

                # Calculate IoU
                x1 = max(person_bbox[0], pose_bbox[0])
                y1 = max(person_bbox[1], pose_bbox[1])
                x2 = min(person_bbox[2], pose_bbox[2])
                y2 = min(person_bbox[3], pose_bbox[3])

                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    person_area = (person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1])
                    pose_area = (pose_bbox[2] - pose_bbox[0]) * (pose_bbox[3] - pose_bbox[1])
                    union = person_area + pose_area - intersection
                    iou = intersection / union

                    if iou > best_iou:
                        best_iou = iou
                        best_match = pose

            # If a good match was found, associate the pose with the pedestrian
            if best_match and best_iou > 0.4:
                matches[person] = best_match['keypoints']

        return matches

    def estimate_3d_poses(self, pose_data, depth_model, depth_map):
        """
        Estimate 3D positions of keypoints using depth data

        Args:
            pose_data: List of pose dictionaries
            depth_model: MetricDepthModel instance
            depth_map: Depth map from depth model

        Returns:
            List of dictionaries with 3D keypoints
        """
        poses_3d = []

        for pose in pose_data:
            keypoints = pose['keypoints']
            keypoints_3d = []

            for kp in keypoints:
                x, y, conf = kp
                if conf > 0.4:  # Only process high-confidence keypoints
                    # Get 3D position
                    pos_3d = depth_model.get_translation_at_point(int(x), int(y), depth_map)
                    keypoints_3d.append([*pos_3d, conf])
                else:
                    keypoints_3d.append([0, 0, 0, conf])

            # Create 3D pose data
            pose_3d = {
                'keypoints_2d': keypoints,
                'keypoints_3d': keypoints_3d,
                'bbox': pose['bbox'],
                'center': pose['center'],
                'confidence': pose['confidence']
            }
            poses_3d.append(pose_3d)

        return poses_3d
