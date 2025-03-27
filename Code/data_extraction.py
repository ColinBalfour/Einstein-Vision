import cv2
import os
import re
import glob
import numpy
print(numpy.version.version)

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

if __name__ == "__main__":
    # Define base folders
    base_folder = "P3Data"
    image_type = "Undist"
    
    process_scene_videos(base_folder, image_type)