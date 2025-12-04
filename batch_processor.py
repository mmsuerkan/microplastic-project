import cv2
import numpy as np
import csv
import os
import sys
import shutil
import re
from auto_particle_tracker import auto_track_particle

# UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# ==============================================
# BATCH PROCESSOR - Tum deneyleri isle
# ==============================================

SOURCE_DIR = "D:/MERT-DUZENLENMIS"
OUTPUT_DIR = "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results"

# Basari kriterleri
MIN_VERTICAL_MOVEMENT = 400  # piksel
MAX_HORIZONTAL_RATIO = 0.5   # yatay/dikey orani


def parse_folder_name(folder_name):
    """IC CAPTURE 01.11.23(MAK) -> (01.11.23, MAK)"""
    # Tarih ve gorusu cikar
    match = re.search(r'IC CAPTURE (\d+\.\d+\.?\d*)\s*[\(\-]?(MAK|ANG)', folder_name, re.IGNORECASE)
    if match:
        date = match.group(1)
        view = match.group(2).upper()
        return date, view
    return None, None


def classify_result(results, video_path):
    """Sonucu siniflandir: success veya fail kategorisi"""

    if results is None:
        # Video acilamadi veya parcacik bulunamadi
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return "fail", "video_error"
        cap.release()
        return "fail", "not_found"

    vertical = results.get('vertical_pixels', 0)
    horizontal = abs(results.get('horizontal_drift', 0))
    total_frames = results.get('total_frames', 0)

    # Video kirpilmamis - cok az hareket
    if vertical < 100:
        return "fail", "not_trimmed"

    # Erken kayip - video uzun ama hareket az
    if vertical < MIN_VERTICAL_MOVEMENT:
        return "fail", "lost_early"

    # Yanlis tespit - yatay sapma cok fazla
    if vertical > 0 and horizontal / vertical > MAX_HORIZONTAL_RATIO:
        return "fail", "wrong_detection"

    # Basarili
    return "success", None


def process_single_experiment(video_path, output_base_dir, date, view, repeat, category, code):
    """Tek bir deneyi isle"""

    print(f"\n{'='*60}")
    print(f"Isleniyor: {date}/{view}/{repeat}/{category}/{code}")
    print(f"{'='*60}")

    # Tracker calistir
    temp_output = "temp_output"
    os.makedirs(temp_output, exist_ok=True)

    try:
        results = auto_track_particle(video_path, output_dir=temp_output, show_video=False, debug=False)
    except Exception as e:
        print(f"HATA: {e}")
        results = None

    # Sonucu siniflandir
    status, fail_reason = classify_result(results, video_path)

    # Cikti klasorunu olustur
    if status == "success":
        output_path = os.path.join(output_base_dir, "success", date, view, repeat, category, code)
    else:
        output_path = os.path.join(output_base_dir, "fail", fail_reason, date, view, repeat, category, code)

    os.makedirs(output_path, exist_ok=True)

    # Dosyalari kopyala
    # Video
    shutil.copy2(video_path, os.path.join(output_path, "output_video.mp4"))

    # Sonuc dosyalari
    for filename in os.listdir(temp_output):
        src = os.path.join(temp_output, filename)
        dst = os.path.join(output_path, filename)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # Temp klasoru temizle
    shutil.rmtree(temp_output, ignore_errors=True)

    print(f"Sonuc: {status.upper()}" + (f" ({fail_reason})" if fail_reason else ""))
    print(f"Kaydedildi: {output_path}")

    return status, fail_reason, results


def find_all_experiments(source_dir):
    """Tum deneyleri bul"""
    experiments = []

    for date_folder in os.listdir(source_dir):
        if not date_folder.startswith("IC CAPTURE"):
            continue

        date, view = parse_folder_name(date_folder)
        if not date or not view:
            continue

        date_path = os.path.join(source_dir, date_folder)

        # FIRST, SECOND, THIRD
        for repeat in ["FIRST", "SECOND", "THIRD"]:
            repeat_path = os.path.join(date_path, repeat)
            if not os.path.isdir(repeat_path):
                continue

            # Kategoriler (BSP, C, HC, WSP, ABS C, etc.)
            for category in os.listdir(repeat_path):
                category_path = os.path.join(repeat_path, category)
                if not os.path.isdir(category_path):
                    continue

                # Parcacik kodlari (BSP-1, C-2, etc.)
                for code in os.listdir(category_path):
                    code_path = os.path.join(category_path, code)
                    if not os.path.isdir(code_path):
                        continue

                    # Video dosyasi ara
                    video_path = os.path.join(code_path, "output_video.mp4")
                    if os.path.exists(video_path):
                        experiments.append({
                            'video_path': video_path,
                            'date': date,
                            'view': view,
                            'repeat': repeat,
                            'category': category,
                            'code': code
                        })

    return experiments


def main():
    print("="*60)
    print("BATCH PROCESSOR - Mikroplastik Deney Analizi")
    print("="*60)

    # Onceki sonuclari temizle
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Tum deneyleri bul
    print(f"\nKaynak: {SOURCE_DIR}")
    print("Deneyler taraniyor...")

    experiments = find_all_experiments(SOURCE_DIR)
    total = len(experiments)

    print(f"Toplam {total} deney bulundu.\n")

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
        print(f"\n[{i}/{total}]", end=" ")

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
        else:
            stats['fail'][fail_reason] += 1

    # Ozet rapor
    print("\n" + "="*60)
    print("OZET RAPOR")
    print("="*60)
    print(f"Toplam deney: {total}")
    print(f"Basarili: {stats['success']} ({100*stats['success']/total:.1f}%)")
    print(f"Basarisiz: {total - stats['success']} ({100*(total-stats['success'])/total:.1f}%)")
    print("\nBasarisizlik dagilimi:")
    for reason, count in stats['fail'].items():
        if count > 0:
            print(f"  - {reason}: {count}")

    # Raporu kaydet
    report_path = os.path.join(OUTPUT_DIR, "batch_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("BATCH PROCESSING REPORT\n")
        f.write("="*40 + "\n")
        f.write(f"Toplam deney: {total}\n")
        f.write(f"Basarili: {stats['success']}\n")
        f.write(f"Basarisiz: {total - stats['success']}\n\n")
        f.write("Basarisizlik dagilimi:\n")
        for reason, count in stats['fail'].items():
            f.write(f"  {reason}: {count}\n")

    print(f"\nRapor kaydedildi: {report_path}")
    print("TAMAMLANDI!")


if __name__ == "__main__":
    main()
