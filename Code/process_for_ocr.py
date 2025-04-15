import cv2
import numpy as np


def euclidian_distance(y1, x1, y2, x2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((y1 - x1) ** 2 + (y2 - x2) ** 2)


def detect_rectangular_sign(img):
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
                    dist = euclidian_distance(center_y, rect_center_x, center_x, rect_center_y)

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


def process_speed_limit_sign(cropped_img):
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


# For backward compatibility
crop_circle_with_text = detect_rectangular_sign