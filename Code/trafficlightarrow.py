import cv2
import numpy as np
from ultralytics import YOLO


def detect_traffic_light_arrows(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return None

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
        (left_arrow, "Left Arrow"),
        (right_arrow, "Right Arrow"),
        (straight_arrow, "Straight Arrow")
    ]

    # Step 2: Detect traffic lights using YOLO
    model = YOLO("yolo11x.pt")
    results = model(image)


    traffic_lights = []

    # Extract traffic light detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            # Traffic light class in COCO is 9
            if cls == 9:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Only consider high confidence detections
                if confidence > 0.3:  # Lower threshold to detect more traffic lights
                    traffic_lights.append((x1, y1, x2, y2))

    # Step 3: Process each traffic light for arrows
    arrow_lights = []
    regular_lights = []

    for i, (x1, y1, x2, y2) in enumerate(traffic_lights):
        # Add padding to capture the whole traffic light
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)

        # Crop the traffic light region
        traffic_light_img = image[y1:y2, x1:x2]

        # Skip if the crop is too small
        if traffic_light_img.shape[0] < 15 or traffic_light_img.shape[1] < 15:
            continue

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
        
        cv2.imwrite(f"outputs/green_mask_{i}.png", np.hstack([traffic_light_img, cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)]))
        cv2.imwrite(f"outputs/red_mask_{i}.png", np.hstack([traffic_light_img, cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)]))
        cv2.imwrite(f"outputs/yellow_mask_{i}.png", np.hstack([traffic_light_img, cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)]))

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
                regular_lights.append((x1, y1, x2, y2))
                continue

        # Keep only contours with reasonable size
        significant_contours = [c for c in contours if cv2.contourArea(c) > 15]

        if not significant_contours:
            regular_lights.append((x1, y1, x2, y2))
            continue

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
                best_match_type = "Arrow (Shape)"
                best_match_score = shape_score

        if is_arrow:
            arrow_lights.append((x1, y1, x2, y2, best_match_type if best_match_type else "Arrow", best_match_score))
        else:
            regular_lights.append((x1, y1, x2, y2))

    # Remove filtering by position to allow detection of multiple arrows

    # Draw results
    result_image = image.copy()

    # Draw regular traffic lights in blue
    for x1, y1, x2, y2 in regular_lights:
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw arrow traffic lights in green
    for x1, y1, x2, y2, arrow_type, score in arrow_lights:
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{arrow_type}: {score:.2f}"
        cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return result_image, regular_lights, arrow_lights, light_color


# Example usage
if __name__ == "__main__":
    image_path = "P3Data/ExtractedFrames/Undist/scene_6/frame_000116.png"
    image_path = "P3Data/ExtractedFrames/Undist/scene_4/frame_000045.png"
    result, regular_lights, arrow_lights, light_color = detect_traffic_light_arrows(image_path)

    # cv2.imshow("Traffic Light Arrows", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(light_color)
    print(f"Detected {len(arrow_lights)} arrow traffic lights and {len(regular_lights)} regular traffic lights")
    for i, (_, _, _, _, arrow_type, score) in enumerate(arrow_lights):
        print(f"Arrow {i + 1}: {arrow_type} (Score: {score:.2f})")



