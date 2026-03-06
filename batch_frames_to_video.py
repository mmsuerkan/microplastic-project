"""Batch frame to video converter"""
import cv2
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

FRAMES_DIR = r"C:\Users\mmert\PycharmProjects\ObjectTrackingProject\temp_frames"
FPS = 50

def create_video_from_frames(frames_dir, output_video_path, fps=50):
    """Frame'lerden video olustur"""
    frames = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir)) if f.lower().endswith(".jpg")]
    if not frames:
        return False

    first_frame = cv2.imread(frames[0])
    if first_frame is None:
        return False

    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_path in frames:
        img = cv2.imread(frame_path)
        if img is not None:
            video.write(img)

    video.release()
    return True

def main():
    print("=" * 60)
    print("BATCH FRAMES TO VIDEO CONVERTER")
    print("=" * 60)

    # Tum klasorleri tara
    to_process = []

    for root, dirs, files in os.walk(FRAMES_DIR):
        jpgs = [f for f in files if f.lower().endswith('.jpg')]
        if jpgs and 'output_video.mp4' not in files:
            to_process.append(root)

    print(f"Islenecek klasor: {len(to_process)}")

    success = 0
    fail = 0

    for i, folder in enumerate(to_process):
        rel_path = os.path.relpath(folder, FRAMES_DIR)
        print(f"\n[{i+1}/{len(to_process)}] {rel_path}")

        output_path = os.path.join(folder, "output_video.mp4")

        if create_video_from_frames(folder, output_path, FPS):
            print(f"  -> OK")
            success += 1
        else:
            print(f"  -> FAIL")
            fail += 1

    print("\n" + "=" * 60)
    print(f"SONUC: {success} basarili, {fail} basarisiz")
    print("=" * 60)

if __name__ == "__main__":
    main()
