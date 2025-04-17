import cv2
import numpy as np
import pytesseract
import os

class OCRModel:
    
    CUSTOM_CONFIG = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
    
    def __init__(self, pytessaract_path=None, verbose=False, output_folder='outputs'):
        
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.verbose = verbose
        # Set tesseract configuration
        if pytessaract_path:
            pytesseract.pytesseract.tesseract_cmd = pytessaract_path
    
    @staticmethod
    def euclidian_distance(self, y1, x1, y2, x2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((y1 - x1) ** 2 + (y2 - x2) ** 2)
    
    def detect_rectangular_sign(self, img):
        """Detect rectangular North American speed limit signs"""
        # Check if image is valid
        if img is None or img.size == 0:
            return None

        # Convert to grayscale if not already
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Apply thresholding to create binary image
        ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Blur using 3 * 3 kernel to reduce noise
        thresh = cv2.blur(thresh, (3, 3))

        # Find contours - looking for rectangles
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Variables to store the best matching rectangle
        best_rect = None
        max_score = 0

        # Center of the image
        center_y = img.shape[0] / 2
        center_x = img.shape[1] / 2

        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the polygon has 4 sides (rectangle)
            if len(approx) == 4:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)

                # Calculate aspect ratio (speed limit signs are typically taller than wide)
                aspect_ratio = float(h) / w

                # Speed limit signs typically have aspect ratio around 1.3-1.5
                if 1.2 < aspect_ratio < 1.6:
                    # Check size - filter out very small rectangles
                    if w > 20 and h > 30:
                        # Calculate distance from center
                        rect_center_x = x + w / 2
                        rect_center_y = y + h / 2
                        dist = self.euclidian_distance(center_y, rect_center_x, center_x, rect_center_y)

                        # Calculate a score based on distance and size (larger signs closer to center are preferred)
                        size_factor = w * h
                        distance_factor = 1.0 / (dist + 1)  # Add 1 to avoid division by zero
                        score = size_factor * distance_factor

                        if score > max_score:
                            max_score = score
                            best_rect = (x, y, w, h)

        if best_rect is not None:
            x, y, w, h = best_rect

            # Create mask for the rectangle
            mask = np.zeros(gray.shape, dtype="uint8")
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

            # Apply mask to get just the sign
            masked = cv2.bitwise_and(gray, gray, mask=mask)

            # Crop to just the sign area for OCR processing
            sign_roi = masked[y:y + h, x:x + w]

            return sign_roi

        return None
    
    def process_speed_limit_sign(self, cropped_img):
        """Process a rectangular North American speed limit sign for OCR"""
        # Handle empty images
        if cropped_img is None or cropped_img.size == 0:
            return None

        # Convert to grayscale if needed
        if len(cropped_img.shape) == 3:
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_img.copy()

        # Resize to a standard height for more consistent OCR
        height = 200
        aspect_ratio = gray.shape[1] / gray.shape[0]
        width = int(height * aspect_ratio)
        gray = cv2.resize(gray, (width, height))

        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

        # Clean up the image with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Focus on the bottom half of the sign where the number is
        h, w = thresh.shape
        number_region = thresh[h // 2:, :]

        return number_region
    
    def get_number(self, img, bbox):
        if isinstance(img, str):
            img = cv2.imread(img)
        
        result_img = img.copy()
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Crop the sign based on Detic detection
        cropped_img = img[y1:y2, x1:x2]
        
        # Refine the sign detection using the rectangular detector
        refined_sign = self.detect_rectangular_sign(cropped_img)

        # Use either the refined sign or original crop
        sign_to_process = refined_sign if refined_sign is not None else cropped_img

        # Process the sign for OCR
        processed_img = self.process_speed_limit_sign(sign_to_process)
        
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
                                                            config=self.CUSTOM_CONFIG)

                # Clean and validate the text
                digit_text = digit_text.strip()

                # Check if we found a valid speed limit
                if digit_text and digit_text.isdigit():
                    speed_val = int(digit_text)
                    if 5 <= speed_val <= 120:  # Typical speed limit range
                        speed_limit = digit_text
                        # Add a label to the image
                        cv2.putText(result_img, f"Speed Limit: {speed_limit}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        
                cv2.imwrite(os.path.join(self.output_folder, 'speed_debug.png'), result_img)
                # Display the processed image if verbose mode is on
                if self.verbose:
                    if processed_img.size > 0:
                        cv2.imshow('Sign Processing', processed_img)
                        if is_speed_limit or digit_text.isdigit():
                            print(f"Detected text: {full_text.strip()}")
                            print(f"Detected digits: {digit_text}")
                            
                return speed_limit

            except Exception as e:
                print(f"OCR Error: {e}")
    
    
def detect_traffic_light_arrows(image, obj, imtype='bgr'):
    # Load image
    if isinstance(image, str):
        image_path = image
        image = cv2.imread(image_path)
        
    if image is None:
        print("Error: Could not load image")
        return None
    
    if imtype == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Step 1: Create synthetic arrow templates
    # Left arrow template
    # left_arrow = np.zeros((20, 20), dtype=np.uint8)
    # cv2.arrowedLine(left_arrow, (15, 10), (5, 10), 255, 2, tipLength=0.5)

    left_arrow = np.zeros((30, 30), dtype=np.uint8)
    cv2.arrowedLine(left_arrow, (22, 15), (8, 15), 255, 3, tipLength=0.4)  # Thicker line, adjusted position
    # cv2.imshow("Left", left_arrow)

    # Right arrow template
    # right_arrow = np.zeros((20, 20), dtype=np.uint8)
    # cv2.arrowedLine(right_arrow, (5, 10), (15, 10), 255, 2, tipLength=0.5)
    right_arrow = np.zeros((30, 30), dtype=np.uint8)
    cv2.arrowedLine(right_arrow, (8, 15), (22, 15), 255, 3, tipLength=0.4)  # Thicker line
    # cv2.imshow("right", right_arrow)

    # Straight arrow template
    # straight_arrow = np.zeros((20, 20), dtype=np.uint8)
    # cv2.arrowedLine(straight_arrow, (10, 15), (10, 5), 255, 2, tipLength=0.5)
    straight_arrow = np.zeros((30, 30), dtype=np.uint8)
    cv2.arrowedLine(straight_arrow, (15, 22), (15, 8), 255, 3, tipLength=0.4)  # Thicker line
    # cv2.imshow("straight", straight_arrow)

    # Store templates with their labels
    templates = [
        (left_arrow, "left"),
        (right_arrow, "right"),
        (straight_arrow, "straight")
    ]

    # Step 3: Process each traffic light for arrows

    x1, y1, x2, y2 = obj.bbox
    
    
    # Add padding to capture the whole traffic light
    padding = 5
    x1 = int(max(0, x1 - padding))
    y1 = int(max(0, y1 - padding))
    x2 = int(min(image.shape[1], x2 + padding))
    y2 = int(min(image.shape[0], y2 + padding))

    # Crop the traffic light region
    traffic_light_img = image[y1:y2, x1:x2]

    # Skip if the crop is too small
    if traffic_light_img.shape[0] < 15 or traffic_light_img.shape[1] < 15:
        return None, None

    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(traffic_light_img, cv2.COLOR_BGR2HSV)

    # Create masks for all possible active colors
    # Green
    lower_green = np.array([50, 10, 50])  # Lowered thresholds
    upper_green = np.array([110, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Blue (for some arrow lights)
    # lower_blue = np.array([90, 30, 30])  # Lowered thresholds
    # upper_blue = np.array([140, 255, 255])
    # red_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # # Cyan/light blue (common in arrow lights)
    # lower_cyan = np.array([80, 30, 150])  # Specific for bright cyan
    # upper_cyan = np.array([100, 255, 255])
    # yellow_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
    
    # Red (for some arrow lights)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    
    # # Yellow (for some arrow lights)
    lower_yellow = np.array([12, 85, 100])  # Lowered thresholds
    upper_yellow = np.array([32, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # if we see too much yellow, its probably traffic light itself
    if cv2.countNonZero(yellow_mask) > .1 * cv2.countNonZero(np.ones_like(yellow_mask)):
        yellow_mask = np.zeros_like(yellow_mask)
    
    
    # find mask with max area (actual light color)
    print(cv2.countNonZero(green_mask), cv2.countNonZero(red_mask), cv2.countNonZero(yellow_mask))
    light_color = np.argmax([cv2.countNonZero(green_mask), cv2.countNonZero(red_mask), cv2.countNonZero(yellow_mask)])
    light_color = ['green', 'red', 'yellow'][light_color]
    
    # cv2.imwrite(f"outputs/green_mask_{i}.png", np.hstack([traffic_light_img, cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)]))
    # cv2.imwrite(f"outputs/red_mask_{i}.png", np.hstack([traffic_light_img, cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)]))
    # cv2.imwrite(f"outputs/yellow_mask_{i}.png", np.hstack([traffic_light_img, cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)]))

    # Combine all color masks
    color_mask = cv2.bitwise_or(cv2.bitwise_or(green_mask, red_mask), yellow_mask)

    # Apply morphological operations
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours, check if there's any signal at all with a more lenient approach
    if not contours:
        # Try with adjusted brightness/contrast and a more generous threshold
        gray = cv2.cvtColor(traffic_light_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return light_color, None

    # Keep only contours with reasonable size
    significant_contours = [c for c in contours if cv2.contourArea(c) > 15]

    if not significant_contours:
        return light_color, None

    # Find the largest contour
    largest_contour = max(significant_contours, key=cv2.contourArea)

    # Create a mask with just the largest contour
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)

    # Calculate shape features
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    # Approximate the contour
    epsilon = 0.04 * perimeter
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Attempt template matching
    best_match_score = 0
    best_match_type = None

    # Try template matching with all templates at multiple scales
    for template, label in templates:
        for scale in np.linspace(0.5, 4.0, 15):  # More scales to try
            # Resize template
            width = int(template.shape[1] * scale)
            height = int(template.shape[0] * scale)

            if width >= contour_mask.shape[1] or height >= contour_mask.shape[0]:
                continue

            resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_LINEAR)

            # Try template matching
            try:
                result = cv2.matchTemplate(contour_mask, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val > best_match_score:
                    best_match_score = max_val
                    best_match_type = label
            except:
                continue

    # Determine if it's an arrow based on shape metrics and template matching
    is_arrow = False

    # Less strict criteria for arrows
    shape_score = 0

    # Check number of approximation points
    if 5 <= len(approx) <= 12:
        shape_score += 0.3

    # Check solidity
    if 0.4 <= solidity <= 0.9:
        shape_score += 0.3

    # Check area relative to bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    extent = float(area) / (w * h)
    if 0.4 <= extent <= 0.9:
        shape_score += 0.3

    # Either good template match or good shape features indicate an arrow
    if best_match_score > 0.2 or shape_score > 0.5:
        is_arrow = True
        if best_match_score <= 0.2:
            best_match_type = None # Arrow (Shape)
            best_match_score = shape_score

    if is_arrow and best_match_score > .5:
        return light_color, best_match_type if best_match_type else None
    else:
        return light_color, None