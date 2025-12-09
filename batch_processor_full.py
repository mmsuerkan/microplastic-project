import cv2
import numpy as np
import csv
import os
import sys
import shutil
import re
import subprocess
import imageio.v3 as iio
from auto_particle_tracker import auto_track_particle

# UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# ==============================================
# BATCH PROCESSOR - Tum deneyleri isle + H.264 + S3
# ==============================================

SOURCE_DIR = "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/source_videos"
OUTPUT_DIR = "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results"
S3_BUCKET = "s3://microplastic-experiments"

# Basari kriterleri
MIN_VERTICAL_MOVEMENT = 400  # piksel
MAX_HORIZONTAL_RATIO = 0.5   # yatay/dikey orani


def parse_folder_name(folder_name):
    """IC CAPTURE 01.11.23(MAK) -> (01.11.23, MAK)"""
    match = re.search(r'IC CAPTURE (\d+\.\d+\.?\d*)\s*[\(\-]?(MAK|ANG)', folder_name, re.IGNORECASE)
    if match:
        date = match.group(1)
        view = match.group(2).upper()
        return date, view
    return None, None


def classify_result(results, video_path):
    """Sonucu siniflandir: success veya fail kategorisi"""

    if results is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return "fail", "video_error"
        cap.release()
        return "fail", "not_found"

    vertical = results.get('vertical_pixels', 0)
    horizontal = abs(results.get('horizontal_drift', 0))

    if vertical < 100:
        return "fail", "not_trimmed"

    if vertical < MIN_VERTICAL_MOVEMENT:
        return "fail", "lost_early"

    if vertical > 0 and horizontal / vertical > MAX_HORIZONTAL_RATIO:
        return "fail", "wrong_detection"

    return "success", None


def convert_to_h264(video_path):
    """Video'yu H.264 formatina donustur"""
    try:
        meta = iio.immeta(video_path, plugin='pyav')
        if meta.get('codec') == 'h264':
            return True  # Zaten H.264

        frames = iio.imread(video_path, plugin='pyav')
        fps = meta.get('fps', 50)

        temp_path = video_path + '.h264.mp4'
        iio.imwrite(temp_path, frames, fps=fps, codec='h264', plugin='pyav')

        os.remove(video_path)
        os.rename(temp_path, video_path)
        return True
    except Exception as e:
        print(f"  H.264 donusum hatasi: {e}")
        return False


def upload_to_s3(local_path, s3_path):
    """Dosyayi S3'e yukle"""
    try:
        result = subprocess.run(
            ['aws', 's3', 'cp', local_path, s3_path],
            capture_output=True,
            check=True
        )
        return True
    except Exception as e:
        print(f"  S3 yukleme hatasi: {e}")
        return False


def process_single_experiment(video_path, output_base_dir, date, view, repeat, category, code):
    """Tek bir deneyi isle, H.264'e cevir ve S3'e yukle"""

    # Tracker calistir
    temp_output = "temp_output"
    os.makedirs(temp_output, exist_ok=True)

    try:
        results = auto_track_particle(video_path, output_dir=temp_output, show_video=False, debug=False)
    except Exception as e:
        print(f"  Tracker hatasi: {e}")
        results = None

    # Sonucu siniflandir
    status, fail_reason = classify_result(results, video_path)

    # Cikti klasorunu olustur
    if status == "success":
        output_path = os.path.join(output_base_dir, "success", date, view, repeat, category, code)
        s3_base_path = f"success/{date}/{view}/{repeat}/{category}/{code}"
    else:
        output_path = os.path.join(output_base_dir, "fail", fail_reason, date, view, repeat, category, code)
        s3_base_path = f"fail/{fail_reason}/{date}/{view}/{repeat}/{category}/{code}"

    os.makedirs(output_path, exist_ok=True)

    # Video'yu kopyala
    output_video = os.path.join(output_path, "output_video.mp4")
    shutil.copy2(video_path, output_video)

    # H.264'e donustur
    convert_to_h264(output_video)

    # S3'e yukle - video
    s3_video_path = f"{S3_BUCKET}/{s3_base_path}/output_video.mp4"
    upload_to_s3(output_video, s3_video_path)

    # Sonuc dosyalarini kopyala ve S3'e yukle
    for filename in os.listdir(temp_output):
        src = os.path.join(temp_output, filename)
        dst = os.path.join(output_path, filename)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            # S3'e yukle
            s3_file_path = f"{S3_BUCKET}/{s3_base_path}/{filename}"
            upload_to_s3(dst, s3_file_path)

    # Temp klasoru temizle
    shutil.rmtree(temp_output, ignore_errors=True)

    return status, fail_reason, results


def get_processed_experiments(processed_dir):
    """Islenmis deneylerin listesini dondur"""
    processed = set()
    for root, dirs, files in os.walk(processed_dir):
        if 'output_video.mp4' in files:
            parts = root.replace('\\', '/').split('/')
            for i, p in enumerate(parts):
                if p in ['success', 'fail']:
                    if p == 'success':
                        key = '/'.join(parts[i+1:])
                    else:
                        key = '/'.join(parts[i+2:])
                    processed.add(key)
                    break
    return processed


def find_unprocessed_experiments(source_dir, processed):
    """Islenmemis deneyleri bul"""
    experiments = []

    for date_folder in os.listdir(source_dir):
        if not date_folder.startswith("IC CAPTURE"):
            continue

        date, view = parse_folder_name(date_folder)
        if not date or not view:
            continue

        date_path = os.path.join(source_dir, date_folder)

        for repeat in ["FIRST", "SECOND", "THIRD"]:
            repeat_path = os.path.join(date_path, repeat)
            if not os.path.isdir(repeat_path):
                continue

            for category in os.listdir(repeat_path):
                category_path = os.path.join(repeat_path, category)
                if not os.path.isdir(category_path):
                    continue

                for code in os.listdir(category_path):
                    code_path = os.path.join(category_path, code)
                    if not os.path.isdir(code_path):
                        continue

                    video_path = os.path.join(code_path, "output_video.mp4")
                    if os.path.exists(video_path):
                        key = f"{date}/{view}/{repeat}/{category}/{code}"
                        if key not in processed:
                            experiments.append({
                                'video_path': video_path,
                                'date': date,
                                'view': view,
                                'repeat': repeat,
                                'category': category,
                                'code': code,
                                'key': key
                            })

    return experiments


def main():
    print("=" * 60)
    print("BATCH PROCESSOR - Mikroplastik Deney Analizi")
    print("Islem + H.264 Donusum + S3 Yukleme")
    print("=" * 60)

    # Onceki sonuclari kontrol et
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processed = get_processed_experiments(OUTPUT_DIR)
    print(f"\nOnceden islenmis: {len(processed)} deney")

    # Islenmemis deneyleri bul
    print(f"Kaynak: {SOURCE_DIR}")
    print("Islenmemis deneyler taraniyor...")

    experiments = find_unprocessed_experiments(SOURCE_DIR, processed)
    total = len(experiments)

    print(f"Islenmemis deney: {total}\n")

    if total == 0:
        print("Tum deneyler islenmis!")
        return

    # Istatistikler
    stats = {
        'success': 0,
        'fail': {
            'not_found': 0,
            'not_trimmed': 0,
            'lost_early': 0,
            'wrong_detection': 0,
            'video_error': 0
        }
    }

    # Her deneyi isle
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{total}] {exp['key']}")

        status, fail_reason, results = process_single_experiment(
            exp['video_path'],
            OUTPUT_DIR,
            exp['date'],
            exp['view'],
            exp['repeat'],
            exp['category'],
            exp['code']
        )

        if status == "success":
            stats['success'] += 1
            print(f"  -> BASARILI")
        else:
            stats['fail'][fail_reason] += 1
            print(f"  -> BASARISIZ ({fail_reason})")

        # Her 10 deneyde bir ozet
        if i % 10 == 0:
            print(f"\n--- Ara Ozet: {i}/{total} tamamlandi ---")
            print(f"    Basarili: {stats['success']}, Basarisiz: {i - stats['success']}")

    # Ozet rapor
    print("\n" + "=" * 60)
    print("OZET RAPOR")
    print("=" * 60)
    print(f"Islenen deney: {total}")
    print(f"Basarili: {stats['success']} ({100*stats['success']/total:.1f}%)")
    print(f"Basarisiz: {total - stats['success']} ({100*(total-stats['success'])/total:.1f}%)")
    print("\nBasarisizlik dagilimi:")
    for reason, count in stats['fail'].items():
        if count > 0:
            print(f"  - {reason}: {count}")

    print("\nTAMAMLANDI!")


if __name__ == "__main__":
    main()
