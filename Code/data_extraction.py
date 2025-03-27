import cv2
import os
import re
import glob
import numpy as np
print(np.version.version)

import scipy.io
# from scipy.io.matlab.mio5 import 

def extract_frames(video_path, output_folder, sample_rate=5):
    """Extract every 5th frame from video"""
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    # Read and process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every 5th frame
        if frame_count % sample_rate == 0:
            output_path = f"{output_folder}/frame_{saved_count:06d}.png"
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Total frames: {frame_count}")
    print(f"Saved frames: {saved_count}")
    return saved_count


# Process all videos in scene folders
def process_scene_videos(base_folder, image_type):
    # Get all directories that match the pattern "scene X"
    output_base = os.path.join(base_folder, f"ExtractedFrames/{image_type}")
    
    scenes_path = os.path.join(base_folder, "Sequences")
    for item in os.listdir(scenes_path):
        item_path = os.path.join(scenes_path, item)
        print(item_path)

        # Check if it's a directory and matches our naming pattern
        if os.path.isdir(item_path) and item.lower().startswith("scene"):
            # Extract scene number using regex
            match = re.search(r'\d+', item)
            if match:
                scene_num = match.group(0)
                scene_path = os.path.join(item_path, image_type)
                print(scene_path)
                
                # Create output folder with same scene numbering
                output_folder = os.path.join(output_base, f"scene_{scene_num}")

                # Process all videos in this scene folder
                for video_path in glob.glob(os.path.join(scene_path, "*front*.mp4")):

                    print(f"Processing {video_path} from {item}...")
                    frames = extract_frames(video_path, output_folder, sample_rate=5)
                    print(f"Extracted {frames} frames from {video_path}")

def get_calibration_matrix(path, cam_id):
    """
    Read the camera calibration matrix from .mat file.
    
    Args:
        path (str): Path to the matlab calibration file.
    
    Returns:
        numpy.ndarray: The camera calibration matrix.
    """
    
    # Load the .mat file
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    
    # Extract the camera's calibration struct
    cam_struct = mat[cam_id]

    # Get the intrinsic matrix (in MATLAB it's 3x3 in column-major)
    return cam_struct.K

if __name__ == "__main__":
    # Define base folders
    base_folder = "P3Data"
    image_type = "Undist"
    
    if not os.path.exists(os.path.join(base_folder, "ExtractedFrames")):
        print("ExtractedFrames folder not found. Extracting frames...")
        process_scene_videos(base_folder, image_type)
    
    if 'calib_mat_front.npy' not in os.listdir(os.path.join(base_folder, "Calib")):
        print("Extracting calibration matrix...")
        K = get_calibration_matrix(os.path.join(base_folder, "Calib/calibration_struct"), 'front')
        
        # Save the calibration matrix as a numpy file
        np.save(os.path.join(base_folder, "Calib/calib_mat_front.npy"), K)
    