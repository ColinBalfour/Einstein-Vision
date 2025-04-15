import cv2
import numpy as np
import pytesseract
from process_for_ocr import process_speed_limit_sign, detect_rectangular_sign
import argparse
import os
from consts_sample import *
import torch
# In your Python script or __init__.py
import sys
import os

# Import Detic and Detectron2 modules
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


from detic.config import add_detic_config
from detic.modeling.text.text_encoder import build_text_encoder
from detic.modeling.utils import reset_cls_test


def setup_detic():
    """Set up the Detic model for street sign detection"""
    cfg = get_cfg()
    add_detic_config(cfg)
    cfg.merge_from_file(CONFIG_PATH)
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set detection threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True

    # Set up text encoder for Detic
    text_encoder = build_text_encoder(cfg)

    # Detic can detect open-vocabulary categories
    categories = DETIC_CATEGORIES
    vocabulary = categories

    # Reset the classifier with our custom categories
    reset_cls_test(cfg, categories, text_encoder)

    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Create metadata
    metadata = MetadataCatalog.get("__unused")
    metadata.thing_classes = categories

    return predictor, metadata


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

    # Create visualization
    v = Visualizer(visualized_img, metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    result = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_img = result.get_image()[:, :, ::-1]  # Convert to BGR for OpenCV

    # Create window for display
    cv2.namedWindow("Detic Detections", cv2.WINDOW_NORMAL)
    if verbose:
        cv2.namedWindow("Sign Processing", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sign Processing", 300, 300)

    # Variable to store the detected speed limit
    speed_limit = 'Unknown'

    # Get instances
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
    scores = instances.scores.numpy() if instances.has("scores") else []
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []

    # Process each detection
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        # Only process high confidence detections
        if score > 0.5:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.astype(int)

            # Crop the sign based on Detic detection
            cropped_img = img[y1:y2, x1:x2]

            # Refine the sign detection using the rectangular detector
            refined_sign = detect_rectangular_sign(cropped_img)

            # Use either the refined sign or original crop
            sign_to_process = refined_sign if refined_sign is not None else cropped_img

            # Process the sign for OCR
            processed_img = process_speed_limit_sign(sign_to_process)

            if processed_img is not None:
                try:
                    # First check if it contains "SPEED" or "LIMIT" text
                    general_config = r'--oem 3 --psm 6'
                    full_text = pytesseract.image_to_string(sign_to_process, lang='eng',
                                                            config=general_config)

                    # Check if this might be a speed limit sign based on text
                    is_speed_limit = "SPEED" in full_text.upper() or "LIMIT" in full_text.upper()

                    # Apply OCR to extract the number
                    digit_text = pytesseract.image_to_string(processed_img, lang='eng',
                                                             config=CUSTOM_CONFIG)

                    # Clean and validate the text
                    digit_text = digit_text.strip()

                    # Check if we found a valid speed limit
                    if digit_text and digit_text.isdigit():
                        speed_val = int(digit_text)
                        if 5 <= speed_val <= 120:  # Typical speed limit range
                            speed_limit = digit_text
                            # Add a label to the image
                            cv2.putText(result_img, f"Speed Limit: {speed_limit}",
                                        (x1, y1 - 10), FONT, 0.8, BLUE_COLOR, THICKNESS)

                    # Display the processed image if verbose mode is on
                    if verbose:
                        if processed_img.size > 0:
                            cv2.imshow('Sign Processing', processed_img)
                            if is_speed_limit or digit_text.isdigit():
                                print(f"Detected text: {full_text.strip()}")
                                print(f"Detected digits: {digit_text}")

                except Exception as e:
                    print(f"OCR Error: {e}")

    # Add information overlay
    cv2.putText(result_img, f'Speed limit: {speed_limit}', (50, 50),
                FONT, 1, BLUE_COLOR, THICKNESS)

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

        # Save the processed image
        cv2.imwrite(output_path, result_img)
        print(f"Output saved to: {output_path}")

    # Wait for a key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return speed_limit


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Speed limit sign detection using Detic')
    parser.add_argument('--path', type=str, required=True, help='Path to image file')
    parser.add_argument('--output', type=str, default='output', help='Directory to save output images')
    parser.add_argument('--verbose', action='store_true', help='Show additional debug information')
    args = parser.parse_args()

    # Setup Detic model
    print("Setting up Detic model...")
    predictor, metadata = setup_detic()

    # Process the image
    print(f"Processing image: {args.path}")
    speed_limit = process_image(args.path, predictor, metadata, args.output, args.verbose)
    print(f"Detected speed limit: {speed_limit}")


if __name__ == '__main__':
    main()