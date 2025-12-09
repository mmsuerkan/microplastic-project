import cv2
import numpy as np
import csv
import os
import sys
import re
import subprocess
import shutil
from auto_particle_tracker import auto_track_particle

sys.stdout.reconfigure(encoding='utf-8')

# ==============================================
# BATCH TRACKER V2 - S3 yapisiyla uyumlu
# ==============================================

SOURCE_DIR = "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/source_videos"
OUTPUT_DIR = "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results"
S3_BUCKET = "microplastic-experiments"

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

    # Kisa takip
    if results.get('total_frames', 0) < 50:
        return "fail", "lost_early"

    # Yetersiz dikey hareket
    if vertical < MIN_VERTICAL_MOVEMENT:
        return "fail", "insufficient_movement"

    # Cok fazla yatay sapma
    if vertical > 0 and horizontal / vertical > MAX_HORIZONTAL_RATIO:
        return "fail", "horizontal_drift"

    return "success", None


def get_s3_processed():
    """S3'te islenmis deneylerin ID'lerini al"""
    processed = set()

    try:
        result = subprocess.run(
            ["aws", "s3", "ls", f"s3://{S3_BUCKET}/", "--recursive"],
            capture_output=True, text=True, timeout=120
        )

        for line in result.stdout.split('\n'):
            if 'auto_tracking_results.csv' in line:
                # Ornek: fail/lost_early/01.11.23/ANG/THIRD/C/C-2/auto_tracking_results.csv
                # veya: success/01.11.23/MAK/FIRST/BSP/BSP-1/auto_tracking_results.csv
                parts = line.split()
                if len(parts) >= 4:
                    path = parts[-1]
                    # success/ veya fail/reason/ sonrasini al
                    if path.startswith('success/'):
                        # success/01.11.23/MAK/FIRST/BSP/BSP-1/...
                        exp_path = path.replace('success/', '').rsplit('/', 1)[0]
                        processed.add(exp_path)
                    elif path.startswith('fail/'):
                        # fail/lost_early/01.11.23/MAK/FIRST/HC/HC-23/...
                        # fail_reason'i atla, gerisi exp_id
                        parts = path.split('/', 2)  # ['fail', 'lost_early', '01.11.23/...']
                        if len(parts) >= 3:
                            exp_path = parts[2].rsplit('/', 1)[0]
                            processed.add(exp_path)
    except Exception as e:
        print(f"S3 kontrol hatasi: {e}")

    return processed


def get_local_processed():
    """Lokal islenmis deneylerin ID'lerini al"""
    processed = set()

    # success klasoru
    success_dir = os.path.join(OUTPUT_DIR, "success")
    if os.path.exists(success_dir):
        for root, dirs, files in os.walk(success_dir):
            if "auto_tracking_results.csv" in files:
                # success/01.11.23/MAK/FIRST/BSP/BSP-1 -> 01.11.23/MAK/FIRST/BSP/BSP-1
                rel_path = os.path.relpath(root, success_dir)
                processed.add(rel_path.replace('\\', '/'))

    # fail klasoru
    fail_dir = os.path.join(OUTPUT_DIR, "fail")
    if os.path.exists(fail_dir):
        for reason_folder in os.listdir(fail_dir):
            reason_path = os.path.join(fail_dir, reason_folder)
            if os.path.isdir(reason_path):
                for root, dirs, files in os.walk(reason_path):
                    if "auto_tracking_results.csv" in files:
                        # fail/lost_early/01.11.23/MAK/... -> 01.11.23/MAK/...
                        rel_path = os.path.relpath(root, reason_path)
                        processed.add(rel_path.replace('\\', '/'))

    return processed


def find_all_experiments():
    """Tum deneyleri bul"""
    experiments = []

    for date_folder in os.listdir(SOURCE_DIR):
        if not date_folder.startswith("IC CAPTURE"):
            continue

        date, view = parse_folder_name(date_folder)
        if not date or not view:
            continue

        date_path = os.path.join(SOURCE_DIR, date_folder)
        if not os.path.isdir(date_path):
            continue

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
                        # exp_id: 01.11.23/MAK/FIRST/BSP/BSP-1
                        exp_id = f"{date}/{view}/{repeat}/{category}/{code}"
                        experiments.append({
                            'date': date,
                            'view': view,
                            'repeat': repeat,
                            'category': category,
                            'code': code,
                            'video_path': video_path,
                            'exp_id': exp_id
                        })

    return experiments


def process_experiment(exp):
    """Tek bir deneyi isle"""

    video_path = exp['video_path']
    exp_id = exp['exp_id']

    print(f"Video: {video_path}")

    # Gecici klasor
    temp_dir = "temp_output"
    os.makedirs(temp_dir, exist_ok=True)

    # Parcacik takibi
    results = auto_track_particle(video_path, temp_dir)

    # Siniflandir
    status, fail_reason = classify_result(results, video_path)

    # Hedef klasor - S3 yapisiyla ayni
    # success/01.11.23/MAK/FIRST/BSP/BSP-1/
    # fail/lost_early/01.11.23/MAK/FIRST/HC/HC-23/
    if status == "success":
        dest_dir = os.path.join(OUTPUT_DIR, "success", exp_id)
    else:
        dest_dir = os.path.join(OUTPUT_DIR, "fail", fail_reason or "unknown", exp_id)

    os.makedirs(dest_dir, exist_ok=True)

    # Sonuclari kopyala
    for fname in ["auto_tracking_results.csv", "auto_tracked_path.jpg", "summary.csv", "optical_flow_vectors.png"]:
        src = os.path.join(temp_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest_dir, fname))

    return status, fail_reason


def main():
    print("=" * 60)
    print("BATCH TRACKER V2 - S3 Yapisiyla Uyumlu")
    print("Video donusum ve S3 yukleme YAPILMAYACAK")
    print("=" * 60)
    print()

    # S3'te islenmisleri al
    print("S3'teki islenmis deneyler kontrol ediliyor...")
    s3_processed = get_s3_processed()
    print(f"S3'te islenmis: {len(s3_processed)} deney")

    # Lokalde islenmisleri al
    print("Lokal islenmis deneyler kontrol ediliyor...")
    local_processed = get_local_processed()
    print(f"Lokalde islenmis: {len(local_processed)} deney")

    # Birlesik set
    all_processed = s3_processed | local_processed
    print(f"Toplam islenmis: {len(all_processed)} deney")

    # Tum deneyler
    print(f"\nKaynak: {SOURCE_DIR}")
    print("Tum deneyler taraniyor...")
    all_experiments = find_all_experiments()
    print(f"Toplam deney: {len(all_experiments)}")

    # Islenmemisleri filtrele
    to_process = [e for e in all_experiments if e['exp_id'] not in all_processed]
    print(f"Islenmemis deney: {len(to_process)}")
    print()

    if not to_process:
        print("Tum deneyler islenmis!")
        return

    # Istatistikler
    stats = {'success': 0, 'fail': 0}
    fail_reasons = {}

    for i, exp in enumerate(to_process):
        print(f"\n[{i+1}/{len(to_process)}] {exp['exp_id']}")

        try:
            status, reason = process_experiment(exp)
            stats[status] += 1

            if reason:
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

            print(f"  -> {'BASARILI' if status == 'success' else 'BASARISIZ'}" +
                  (f" ({reason})" if reason else ""))

        except Exception as e:
            print(f"  -> HATA: {e}")
            stats['fail'] += 1
            fail_reasons['error'] = fail_reasons.get('error', 0) + 1

    # Ozet
    print("\n" + "=" * 60)
    print("OZET")
    print("=" * 60)
    print(f"Basarili: {stats['success']}")
    print(f"Basarisiz: {stats['fail']}")
    print("\nBasarisizlik nedenleri:")
    for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
