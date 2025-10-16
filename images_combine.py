import cv2
import os

def create_video_from_frames(frames_dir, output_video_path, fps=50):
    frames = [os.path.join(frames_dir, frame) for frame in sorted(os.listdir(frames_dir)) if frame.endswith(".jpg")]
    if not frames:
        print("No frames found in the directory.")
        return

    # Read the first frame to get the size
    first_frame = cv2.imread(frames[0])
    if first_frame is None:
        print("Error processing the first frame.")
        return
    height, width, layers = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use the correct codec
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_path in frames:
        img = cv2.imread(frame_path)
        if img is not None:
            video.write(img)

    video.release()
    print(f"Video saved at {output_video_path}")

# Example usage
frames_directory = r"C:\Users\mmert\PycharmProjects\ObjectTrackingProject\images"  # Directory containing images
output_video_file = r"C:\Users\mmert\PycharmProjects\ObjectTrackingProject\output.mp4"  # Output video file path

create_video_from_frames(frames_directory, output_video_file)
