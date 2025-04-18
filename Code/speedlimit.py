
import cv2
import numpy as np
import pytesseract
import argparse
import os
import sys
import torch
import time
import json
import re
from collections import Counter

# Configuration and Model Paths (default paths)
CONFIG_PATH = "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH = "/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

# Add necessary paths for imports
sys.path.insert(0, 'third_party/CenterNet2/')
sys.path.insert(0, 'D:/3. Computer vision/Homeworks/Einstein-Vision/detectron2')

# Import Detic and Detectron2 modules
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# Import CenterNet config
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.text.text_encoder import build_text_encoder
from detic.modeling.utils import reset_cls_test

# Constants for OCR
PYTESSARACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSDATA_CONFIG = r'C:\Program Files\Tesseract-OCR\tessdata'
FONT = cv2.FONT_HERSHEY_SIMPLEX
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
THICKNESS = 2

# Detic categories to detect
DETIC_CATEGORIES = ["street_sign"]


def ml_enhanced_ocr(sign_image):
    """
    Advanced OCR method with machine learning enhanced preprocessing and validation

    Args:
        sign_image (numpy.ndarray): Input sign image

    Returns:
        int or None: Detected speed limit or None if detection fails
    """

    def preprocess_for_ml_ocr(image):
        """
        Advanced image preprocessing for OCR

        Args:
            image (numpy.ndarray): Input image

        Returns:
            list: List of preprocessed images
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(gray)

        # Multiple image transformations
        transforms = [
            # Basic binary thresholding
            cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],

            # Adaptive thresholding
            cv2.adaptiveThreshold(normalized, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2),

            # Enhance contrast
            cv2.equalizeHist(normalized)
        ]

        return transforms

    def ensemble_ocr_extraction(preprocessed_images):
        """
        Ensemble OCR extraction with multiple configurations

        Args:
            preprocessed_images (list): List of preprocessed images

        Returns:
            list: Detected numbers
        """
        results = []

        # Tesseract configurations for different scenarios
        ocr_configs = [
            '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789',
            '--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789',
            '--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789',
            '--psm 7 -c tessedit_char_whitelist=0123456789'
        ]

        for image in preprocessed_images:
            for config in ocr_configs:
                try:
                    # Extract text
                    text = pytesseract.image_to_string(image, config=config)

                    # Extract only digits
                    digits = re.findall(r'\d+', text)
                    results.extend(digits)

                except Exception as e:
                    print(f"OCR extraction error: {e}")

        return results

    def validate_speed_limit(detected_numbers):
        """
        Validate and filter speed limit numbers

        Args:
            detected_numbers (list): List of detected number strings

        Returns:
            int or None: Most likely speed limit
        """
        # Typical North American speed limits
        valid_limits = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

        # Convert to integers and filter
        valid_detected = []
        for num_str in detected_numbers:
            try:
                num = int(num_str)
                if num in valid_limits:
                    valid_detected.append(num)
            except ValueError:
                continue

        # Use statistical voting
        if not valid_detected:
            return None

        # Count occurrences
        number_counts = Counter(valid_detected)

        # Return most common valid number
        most_common = number_counts.most_common(1)[0]

        # Optional: Add confidence threshold
        total_detections = sum(number_counts.values())
        confidence_ratio = most_common[1] / total_detections

        # Return number if it appears in more than 30% of detections
        return most_common[0] if confidence_ratio > 0.3 else None

    # Combine all steps
    preprocessed_images = preprocess_for_ml_ocr(sign_image)
    detected_numbers = ensemble_ocr_extraction(preprocessed_images)
    final_speed_limit = validate_speed_limit(detected_numbers)

    # Additional logging for debugging
    print(f"Detected numbers: {detected_numbers}")
    print(f"Final speed limit: {final_speed_limit}")

    return final_speed_limit


def detect_speed_hump(sign):
    """
    Advanced speed hump detection method

    Args:
        sign (numpy.ndarray): Potential speed hump sign image

    Returns:
        bool: Whether a speed hump sign is detected
    """
    # Convert to HSV color space for better color detection
    hsv = cv2.cvtColor(sign, cv2.COLOR_BGR2HSV)

    # Define color ranges for yellow and diamond-like shapes
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    # Create yellow color mask
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Shape detection
    gray = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check for diamond-like shapes
    diamond_detected = False
    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # Diamond has 4 vertices and is roughly square
        if len(approx) == 4:
            # Calculate aspect ratio and area
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            area_ratio = cv2.contourArea(contour) / (w * h)

            # Check for diamond-like shape
            if 0.8 < aspect_ratio < 1.2 and 0.7 < area_ratio < 1.3:
                diamond_detected = True
                break

    # OCR for additional confirmation
    try:
        # Try different OCR configurations
        ocr_configs = [
            '--psm 6',
            '--psm 11',
            '--psm 3'
        ]

        for config in ocr_configs:
            text = pytesseract.image_to_string(sign, config=config)
            text_upper = text.upper()

            # Look for speed hump keywords
            if "HUMP" in text_upper or "SPEED" in text_upper:
                return True
    except Exception as e:
        print(f"Speed hump OCR error: {e}")

    # Yellow color and diamond shape detection
    yellow_ratio = np.sum(yellow_mask > 0) / (sign.shape[0] * sign.shape[1])

    # Combine multiple detection criteria
    return (yellow_ratio > 0.3 and diamond_detected)


def process_image(image_path, predictor, metadata, output_dir=None, verbose=False):
    """Process an image to detect speed limit signs using Detic"""
    # Read the image
    img = read_image(image_path, format="BGR")
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Set tesseract configuration
    pytesseract.pytesseract.tesseract_cmd = PYTESSARACT_PATH

    # Run Detic detection
    outputs = predictor(img)

    # Create a copy of the image for visualization
    visualized_img = img.copy()
    result_img = visualized_img.copy()

    # Get instances
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
    scores = instances.scores.numpy() if instances.has("scores") else []
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []

    # Variables to store detection results
    speed_limit = 'Unknown'
    speed_hump_detected = False

    # Create a dictionary to store results for JSON
    result_data = {
        "filename": os.path.basename(image_path),
        "speed_limit": "Unknown",
        "speed_limit_range": "",
        "speed_hump_detected": False,
        "detection_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detected_objects": []
    }

    # Look for speed limit sign and speed hump
    for i, (box, score) in enumerate(zip(boxes, scores)):
        if score > 0.5:
            x1, y1, x2, y2 = box.astype(int)

            # Extract the sign region
            if y1 < 0: y1 = 0
            if x1 < 0: x1 = 0
            if y2 >= img.shape[0]: y2 = img.shape[0] - 1
            if x2 >= img.shape[1]: x2 = img.shape[1] - 1

            # Skip invalid boxes
            if y1 >= y2 or x1 >= x2:
                continue

            sign = img[y1:y2, x1:x2]

            # Calculate aspect ratio
            width, height = x2 - x1, y2 - y1
            aspect_ratio = height / width if width > 0 else 0

            # Speed Limit Sign Detection
            if aspect_ratio > 1.2:
                # Use the enhanced ML-based OCR method
                detected_limit = ml_enhanced_ocr(sign)

                if detected_limit:
                    speed_limit = str(detected_limit)
                    result_data["speed_limit"] = speed_limit

                    # Determine speed limit range
                    if 10 <= int(speed_limit) <= 20:
                        result_data["speed_limit_range"] = "Slow zone "
                    elif 20 < int(speed_limit) <= 35:
                        result_data["speed_limit_range"] = "Urban area"
                    elif 35 < int(speed_limit) <= 55:
                        result_data["speed_limit_range"] = "Suburban road (35-55 mph)"
                    elif 55 < int(speed_limit) <= 70:
                        result_data["speed_limit_range"] = "Highway (55-70 mph)"
                    else:
                        result_data["speed_limit_range"] = "High-speed zone (> 70 mph)"

                    # Annotate the image
                    cv2.putText(result_img, f"Speed Limit: {speed_limit}",
                                (x1, y1 - 10), FONT, 0.8, BLUE_COLOR, THICKNESS)

            # Speed Hump Detection
            # Check for diamond-like shape and color
            if 0.8 < aspect_ratio < 1.2:
                # Detect speed hump
                is_speed_hump = detect_speed_hump(sign)

                if is_speed_hump:
                    speed_hump_detected = True
                    result_data["speed_hump_detected"] = True
                    cv2.putText(result_img, "SPEED HUMP",
                                (x1, y1 - 10), FONT, 0.8, (0, 255, 0), THICKNESS)

    # Add information overlay
    cv2.putText(result_img, f'Speed limit: {speed_limit}', (50, 50),
                FONT, 1, BLUE_COLOR, THICKNESS)

    if result_data["speed_limit"] != "Unknown":
        cv2.putText(result_img, f'Range: {result_data["speed_limit_range"]}', (50, 90),
                    FONT, 1, BLUE_COLOR, THICKNESS)

    # Add speed hump detection to overlay
    if speed_hump_detected:
        cv2.putText(result_img, 'Speed Hump: Detected', (50, 130),
                    FONT, 1, (0, 255, 0), THICKNESS)

    # Display the result
    cv2.imshow('Detic Detections', result_img)

    # Save output if requested
    if output_dir:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get base filename from the input path
        base_filename = os.path.basename(image_path)
        filename, ext = os.path.splitext(base_filename)

        # Create output path with detected speed limit in filename
        output_path = os.path.join(output_dir, f"{filename}_speed_{speed_limit}{ext}")
        json_path = os.path.join(output_dir, f"{filename}_results.json")

        # Save the processed image
        cv2.imwrite(output_path, result_img)
        print(f"Output image saved to: {output_path}")

        # Save JSON results
        with open(json_path, 'w') as json_file:
            json.dump(result_data, json_file, indent=4)
        print(f"Detection results saved to: {json_path}")

    # Wait for a key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return speed_limit


# Rest of the main() function remains the same as in the previous script
def main():
    global PYTESSARACT_PATH

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Speed limit sign detection using Detic')
    parser.add_argument('--config-file', type=str, default=CONFIG_PATH,
                        help=f'Path to the config file (default: {CONFIG_PATH})')
    parser.add_argument('--path', type=str, required=True, help='Path to image file')
    parser.add_argument('--output', type=str, default='output', help='Directory to save output images and JSON results')
    parser.add_argument('--verbose', action='store_true', help='Show additional debug information')
    parser.add_argument('--categories', type=str, default=','.join(DETIC_CATEGORIES),
                        help=f'Categories to detect, comma-separated (default: {",".join(DETIC_CATEGORIES)})')
    parser.add_argument('--tesseract-path', type=str, default=PYTESSARACT_PATH,
                        help=f'Path to tesseract executable (default: {PYTESSARACT_PATH})')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, help='Additional options to pass to the model')
    args = parser.parse_args()

    # Update Tesseract path if provided
    PYTESSARACT_PATH = args.tesseract_path

    # Parse categories if provided
    categories = args.categories.split(',')

    # Setup Detic model using the provided config file
    print("Setting up Detic model...")

    # Configure the model
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)

    # Apply any additional options (like model weights)
    if args.opts and len(args.opts) > 0:
        # Make sure we're not passing --verbose as part of the opts
        filtered_opts = [opt for opt in args.opts if opt != '--verbose']
        if len(filtered_opts) > 0:
            print(f"Applying additional options: {filtered_opts}")
            cfg.merge_from_list(filtered_opts)

    # Set detection threshold and other parameters
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True

    # Set up text encoder
    text_encoder = build_text_encoder(cfg)

    try:
        # Reset the classifier with our custom categories
        reset_cls_test(cfg, categories, text_encoder)
    except AttributeError as e:
        print(f"Warning: Could not reset classifier - {e}")
        print("Continuing with default classifier...")

    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Create metadata
    metadata = MetadataCatalog.get("__unused")
    metadata.thing_classes = categories

    # Process the image
    print(f"Processing image: {args.path}")
    speed_limit = process_image(args.path, predictor, metadata, args.output, args.verbose)
    print(f"Detected speed limit: {speed_limit}")


if __name__ == '__main__':
    main()
