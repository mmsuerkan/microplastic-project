import cv2
import numpy as np
import csv
import os
import sys
import re
import shutil
from auto_particle_tracker_v3 import auto_track_particle

sys.stdout.reconfigure(encoding='utf-8')

# ==============================================
# BATCH TRACKER V4 - Third Iteration
# Sadece insufficient_movement deneyleri yeniden tarar
# V3 tracker kullanir (aralikli frame tarama)
# ==============================================

SOURCE_DIR = "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/source_videos"
OUTPUT_DIR = "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results/third_iteration"

# insufficient_movement deneyleri - hem ilk iterasyondan hem second iterasyondan
FAIL_DIRS = [
    "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results/fail/insufficient_movement",
    "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results/second_iteration/fail/insufficient_movement"
]

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


def get_insufficient_movement_experiments():
    """insufficient_movement altindaki tum deneyleri bul"""
    experiments = []
    seen_exp_ids = set()  # Tekrarlari onle

    for fail_dir in FAIL_DIRS:
        if not os.path.exists(fail_dir):
            print(f"UYARI: {fail_dir} bulunamadi, atlaniyor...")
            continue

        print(f"Taraniyor: {fail_dir}")

        # Klasor yapisi: fail/insufficient_movement/DATE/VIEW/REPEAT/CATEGORY/CODE/
        for date_folder in os.listdir(fail_dir):
            date_path = os.path.join(fail_dir, date_folder)
            if not os.path.isdir(date_path):
                continue

            for view_folder in os.listdir(date_path):
                view_path = os.path.join(date_path, view_folder)
                if not os.path.isdir(view_path):
                    continue

                for repeat_folder in os.listdir(view_path):
                    repeat_path = os.path.join(view_path, repeat_folder)
                    if not os.path.isdir(repeat_path):
                        continue

                    for category_folder in os.listdir(repeat_path):
                        category_path = os.path.join(repeat_path, category_folder)
                        if not os.path.isdir(category_path):
                            continue

                        for code_folder in os.listdir(category_path):
                            code_path = os.path.join(category_path, code_folder)
                            if not os.path.isdir(code_path):
                                continue

                            exp_id = f"{date_folder}/{view_folder}/{repeat_folder}/{category_folder}/{code_folder}"

                            # Tekrar kontrolu
                            if exp_id in seen_exp_ids:
                                continue
                            seen_exp_ids.add(exp_id)

                            # Source video yolunu bul
                            source_video = find_source_video(date_folder, view_folder, repeat_folder, category_folder, code_folder)

                            if source_video and os.path.exists(source_video):
                                experiments.append({
                                    'date': date_folder,
                                    'view': view_folder,
                                    'repeat': repeat_folder,
                                    'category': category_folder,
                                    'code': code_folder,
                                    'video_path': source_video,
                                    'exp_id': exp_id,
                                    'original_fail_path': code_path
                                })

    return experiments


def find_source_video(date, view, repeat, category, code):
    """Source video yolunu bul"""
    for folder in os.listdir(SOURCE_DIR):
        if not folder.startswith("IC CAPTURE"):
            continue

        parsed_date, parsed_view = parse_folder_name(folder)
        if parsed_date == date and parsed_view == view:
            video_path = os.path.join(SOURCE_DIR, folder, repeat, category, code, "output_video.mp4")
            if os.path.exists(video_path):
                return video_path

    return None


def process_experiment(exp):
    """Tek bir deneyi isle"""

    video_path = exp['video_path']
    exp_id = exp['exp_id']

    print(f"Video: {video_path}")

    # Gecici klasor
    temp_dir = "temp_output"
    os.makedirs(temp_dir, exist_ok=True)

    # Parcacik takibi (V3 - aralikli frame tarama)
    results = auto_track_particle(video_path, temp_dir, show_video=False)

    # Siniflandir
    status, fail_reason = classify_result(results, video_path)

    # Hedef klasor - third_iteration altinda
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

    # V3 ek bilgi: detection_frame
    if results and 'detection_frame' in results:
        print(f"  Tespit frame: {results['detection_frame']}")

    return status, fail_reason, results


def main():
    print("=" * 60)
    print("BATCH TRACKER V4 - Third Iteration")
    print("Sadece insufficient_movement deneyleri V3 tracker ile taranacak")
    print("V3: Aralikli frame tarama (video basi kirpilmamis durumlar icin)")
    print("=" * 60)
    print()

    # insufficient_movement deneylerini bul
    print("insufficient_movement deneyleri araniyor...")
    experiments = get_insufficient_movement_experiments()
    print(f"Toplam insufficient_movement deney: {len(experiments)}")
    print()

    if not experiments:
        print("Islenecek deney bulunamadi!")
        return

    # Istatistikler
    stats = {'success': 0, 'fail': 0}
    fail_reasons = {}
    converted = []  # insufficient_movement -> success donusenler
    improved = []   # Daha iyi sonuc alanlar

    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {exp['exp_id']}")

        try:
            status, reason, results = process_experiment(exp)
            stats[status] += 1

            if status == "success":
                converted.append({
                    'exp_id': exp['exp_id'],
                    'vertical': results.get('vertical_pixels', 0) if results else 0,
                    'detection_frame': results.get('detection_frame', 0) if results else 0
                })
                print(f"  -> BASARILI! (onceden insufficient_movement idi)")
                if results:
                    print(f"     Dikey hareket: {results.get('vertical_pixels', 0)} piksel")
            else:
                if reason:
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                print(f"  -> BASARISIZ ({reason})")
                if results:
                    print(f"     Dikey hareket: {results.get('vertical_pixels', 0)} piksel")

        except Exception as e:
            print(f"  -> HATA: {e}")
            stats['fail'] += 1
            fail_reasons['error'] = fail_reasons.get('error', 0) + 1

    # Ozet
    print("\n" + "=" * 60)
    print("OZET - THIRD ITERATION (insufficient_movement)")
    print("=" * 60)
    print(f"Toplam islenen: {len(experiments)}")
    print(f"Basarili (yeni): {stats['success']}")
    print(f"Hala basarisiz: {stats['fail']}")
    if len(experiments) > 0:
        print(f"\nDonusum orani: {stats['success']}/{len(experiments)} ({100*stats['success']/len(experiments):.1f}%)")

    if converted:
        print(f"\n--- INSUFFICIENT_MOVEMENT -> SUCCESS DONUSENLER ({len(converted)}) ---")
        for item in converted:
            print(f"  + {item['exp_id']}")
            print(f"    Dikey: {item['vertical']} px, Tespit frame: {item['detection_frame']}")

    if fail_reasons:
        print("\nHala basarisiz olanlar:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Sonuclari dosyaya kaydet
    report_path = os.path.join(OUTPUT_DIR, "iteration_report.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("THIRD ITERATION REPORT (insufficient_movement)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Tracker: V3 (Aralikli frame tarama)\n")
        f.write(f"Toplam islenen: {len(experiments)}\n")
        f.write(f"Basarili (yeni): {stats['success']}\n")
        f.write(f"Hala basarisiz: {stats['fail']}\n")
        if len(experiments) > 0:
            f.write(f"Donusum orani: {100*stats['success']/len(experiments):.1f}%\n\n")

        if converted:
            f.write(f"INSUFFICIENT_MOVEMENT -> SUCCESS ({len(converted)}):\n")
            for item in converted:
                f.write(f"  {item['exp_id']}\n")
                f.write(f"    Dikey: {item['vertical']} px, Tespit frame: {item['detection_frame']}\n")

        if fail_reasons:
            f.write(f"\nHala basarisiz:\n")
            for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
                f.write(f"  {reason}: {count}\n")

    print(f"\nRapor kaydedildi: {report_path}")


if __name__ == "__main__":
    main()
