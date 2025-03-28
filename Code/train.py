from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# This is the critical part for Windows
if __name__ == '__main__':
    # Add freeze_support for Windows
    freeze_support()

    # Load a model
    model = YOLO('yolo11x.pt')  # load a pretrained model

    # Train the model
    results = model.train(
        data='C:/Users/simra/Projects/Code/datasets/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=30, device = 'cuda'
    )
