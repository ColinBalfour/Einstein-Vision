import cv2

MODEL_PATH = 'D:/3. Computer vision/Homeworks/Einstein-Vision/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
CONFIG_PATH = 'D:/3. Computer vision/Homeworks/Einstein-Vision/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
PYTESSARACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSDATA_CONFIG = r'C:\Program Files\Tesseract-OCR\tessdata'
FONT = cv2.FONT_HERSHEY_SIMPLEX
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
THICKNESS = 2
CUSTOM_CONFIG = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'

# For Detic, define categories to detect instead of fixed classes
DETIC_CATEGORIES = ["street sign", "traffic sign", "speed limit sign", "road sign"]