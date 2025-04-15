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
    
    