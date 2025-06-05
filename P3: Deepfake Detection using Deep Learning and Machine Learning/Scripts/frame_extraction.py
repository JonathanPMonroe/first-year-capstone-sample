import os
import cv2
import multiprocessing

video_dir = "Data/extracted_videos/"
frame_output_dir = "Data/frames/"
os.makedirs(frame_output_dir, exist_ok=True)

#I wrote code to extract frames from each mp4. Number of frames extracted per video: 11
#This provides a good representation of the video without needing to process too much data

def extract_frames(video_path):
    """Extracts frames from a video at approximately 1 frame per second and saves them."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_path = os.path.join(frame_output_dir, video_name)
    os.makedirs(video_output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"❌ Could not retrieve FPS for video: {video_path}. Skipping...")
        cap.release()
        return

    fps = int(fps)
    frame_interval = max(1, fps)
    frame_count = 0
    saved_count = 0

    print(f"Extracting frames from {video_name}...")

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(video_output_path, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Finished {video_name}: {saved_count} frames saved.")

def get_video_files(video_dir):
    """
    Recursively retrieves all .mp4 files (case-insensitive) from the given directory using os.walk.
    Debug statements list each directory and file encountered.
    """
    video_files = []
    print("Scanning directories using os.walk:")
    for root, dirs, files in os.walk(video_dir):
        print("Directory:", root)
        for file in files:
            print("  Found file:", file)
            if file.lower().endswith(".mp4"):
                video_files.append(os.path.join(root, file))
    print(f"Found {len(video_files)} video files.")
    return video_files

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    print("Looking for videos in:", os.path.abspath(video_dir))
    
    videos = get_video_files(video_dir)
    
    print(f"Found {len(videos)} videos. Starting multiprocessing...")
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(extract_frames, videos)

    print("Frame extraction complete!")
