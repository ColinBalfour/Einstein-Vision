# Einstein-Vision

## Usage Instructions

1. Install all dependencies (core depdendencies listed below) - NOTE: please review script before running, depthpro in particular has a large install size
    Run ./install_dependencies.sh
2. Parse data into proper format. Using P3Data with (Assets, Calib, Sequences) subfolders:
    Run ./Code/parse_data.sh

## Dependencies:

Depth Pro: https://github.com/apple/ml-depth-pro/
Lane Segmentation (only need checkpoint): 
    Article, https://debuggercafe.com/lane-detection-using-mask-rcnn/
    Download, https://drive.usercontent.google.com/download?id=1WRu0e5GYlsKmQp0UR_ZyuYoXQmQWR4Pj&export=download&authuser=0

HumanPose - "yolo11x-pose.pt"

Vehicle classification - https://github.com/MaryamBoneh/Vehicle-Detection
yolov11 - https://docs.ultralytics.com/models/yolo11/#performance-metrics

Data (name P3Data, place in root): https://app.box.com/s/zjys9xcefyqfj2oxkwsm5irgz6g3hp1l

Github repo Traffic sign detection:https://github.com/bhaskrr/traffic-sign-detection-using-yolov11/blob/main/process_video.py
Pretrained Model - https://github.com/bhaskrr/traffic-sign-detection-using-yolov11/tree/main/model
