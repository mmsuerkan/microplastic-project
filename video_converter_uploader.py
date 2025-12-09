import os
import sys
import subprocess
import imageio_ffmpeg

sys.stdout.reconfigure(encoding='utf-8')

# ==============================================
# VIDEO CONVERTER + S3 UPLOADER
# S3'te olanlari atlar, sadece yenileri yukler
# FFmpeg ile H.264 donusumu
# ==============================================

PROCESSED_DIR = "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results"
SOURCE_DIR = "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/source_videos"
S3_BUCKET = "microplastic-experiments"
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def get_s3_videos():
    """S3'te zaten yuklu videolari bul"""
    videos = set()
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", f"s3://{S3_BUCKET}/", "--recursive"],
            capture_output=True, text=True, timeout=120
        )
        for line in result.stdout.split('\n'):
            if 'output_video.mp4' in line:
                # Path'i cikart
                parts = line.split()
                if len(parts) >= 4:
                    path = parts[-1]
                    # success/06.11.2024/MAK/... veya fail/lost_early/06.11.2024/...
                    videos.add(path.replace('/output_video.mp4', ''))
    except Exception as e:
        print(f"S3 kontrol hatasi: {e}")
    return videos


def find_source_video(exp_path):
    """Deney icin kaynak videoyu bul"""
    # exp_path: success/06.11.2024/MAK/FIRST/ABS C/C-13
    # veya: fail/insufficient_movement/06.11.2024/MAK/FIRST/ABS C/C-14

    parts = exp_path.split('/')

    if parts[0] == 'success':
        # success/tarih/view/repeat/category/code
        date, view, repeat, category, code = parts[1], parts[2], parts[3], parts[4], parts[5]
    else:
        # fail/reason/tarih/view/repeat/category/code
        date, view, repeat, category, code = parts[2], parts[3], parts[4], parts[5], parts[6]

    # Kaynak klasoru bul
    for folder in os.listdir(SOURCE_DIR):
        if folder.startswith("IC CAPTURE") and date in folder and view in folder:
            video_path = os.path.join(SOURCE_DIR, folder, repeat, category, code, "output_video.mp4")
            if os.path.exists(video_path):
                return video_path

    return None


def convert_to_h264(input_path, output_path):
    """FFmpeg ile H.264 formatina donustur"""
    try:
        result = subprocess.run(
            [
                FFMPEG_EXE, "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-crf", "23",
                "-preset", "medium",
                "-an",  # Ses yok
                output_path
            ],
            capture_output=True, text=True, timeout=300
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Donusum hatasi: {e}")
        return False


def upload_to_s3(local_path, s3_path):
    """S3'e yukle"""
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", local_path, f"s3://{S3_BUCKET}/{s3_path}"],
            capture_output=True, text=True, timeout=300
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Upload hatasi: {e}")
        return False


def find_local_experiments():
    """Lokal deneyleri bul"""
    experiments = []

    # Success
    success_dir = os.path.join(PROCESSED_DIR, "success")
    if os.path.exists(success_dir):
        for root, dirs, files in os.walk(success_dir):
            if "auto_tracking_results.csv" in files:
                rel_path = os.path.relpath(root, PROCESSED_DIR)
                experiments.append(rel_path.replace('\\', '/'))

    # Fail
    fail_dir = os.path.join(PROCESSED_DIR, "fail")
    if os.path.exists(fail_dir):
        for root, dirs, files in os.walk(fail_dir):
            if "auto_tracking_results.csv" in files:
                rel_path = os.path.relpath(root, PROCESSED_DIR)
                experiments.append(rel_path.replace('\\', '/'))

    return experiments


def main():
    print("=" * 60)
    print("VIDEO CONVERTER + S3 UPLOADER")
    print("S3'te olanlari atlar, sadece yenileri yukler")
    print("=" * 60)
    print()

    # S3'te yuklu videolar
    print("S3'teki videolar kontrol ediliyor...")
    s3_videos = get_s3_videos()
    print(f"S3'te yuklu: {len(s3_videos)} video")

    # Lokal deneyler
    print("Lokal deneyler taraniyor...")
    local_exps = find_local_experiments()
    print(f"Lokal deney: {len(local_exps)}")

    # Yuklenmemisleri filtrele
    to_upload = [e for e in local_exps if e not in s3_videos]
    print(f"Yuklenecek: {len(to_upload)}")
    print()

    if not to_upload:
        print("Tum deneyler zaten S3'te!")
        return

    # Islem
    success_count = 0
    fail_count = 0

    for i, exp_path in enumerate(to_upload):
        print(f"\n[{i+1}/{len(to_upload)}] {exp_path}")

        # Kaynak videoyu bul
        source_video = find_source_video(exp_path)
        if not source_video:
            print("  -> Kaynak video bulunamadi, ATLANIYOR")
            fail_count += 1
            continue

        # Hedef klasor
        dest_dir = os.path.join(PROCESSED_DIR, exp_path)
        h264_path = os.path.join(dest_dir, "output_video.mp4")

        # H.264'e donustur
        print(f"  H.264 donusturuluyor...")
        if not convert_to_h264(source_video, h264_path):
            print("  -> Donusum BASARISIZ")
            fail_count += 1
            continue

        # S3'e yukle - tum dosyalar
        print(f"  S3'e yukleniyor...")
        all_uploaded = True

        for fname in os.listdir(dest_dir):
            local_file = os.path.join(dest_dir, fname)
            s3_path = f"{exp_path}/{fname}"

            if not upload_to_s3(local_file, s3_path):
                all_uploaded = False
                break

        if all_uploaded:
            print("  -> BASARILI")
            success_count += 1
        else:
            print("  -> Upload BASARISIZ")
            fail_count += 1

    # Ozet
    print("\n" + "=" * 60)
    print("OZET")
    print("=" * 60)
    print(f"Basarili: {success_count}")
    print(f"Basarisiz: {fail_count}")


if __name__ == "__main__":
    main()
