"""
S3 Results Updater - Sadece yeni sonuclari S3'e yukler ve JSON'i gunceller
Videolar zaten S3'te oldugu icin sadece CSV + JPG yukler
"""
import os
import sys
import json
import boto3
import shutil
from pathlib import Path
import mimetypes

sys.stdout.reconfigure(encoding='utf-8')

# S3 ayarlari
BUCKET_NAME = "microplastic-experiments"
REGION = "us-east-1"
RESULTS_DIR = "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results"

# S3 client
s3 = boto3.client('s3', region_name=REGION)


def merge_iteration_results():
    """2nd ve 3rd iteration success sonuclarini ana success klasorune kopyala"""

    iterations = [
        ("second_iteration/success", "2nd iteration"),
        ("third_iteration/success", "3rd iteration")
    ]

    merged_count = 0

    for iter_path, iter_name in iterations:
        src_dir = os.path.join(RESULTS_DIR, iter_path)
        if not os.path.exists(src_dir):
            print(f"{iter_name}: Klasor bulunamadi, atlaniyor...")
            continue

        print(f"\n{iter_name} sonuclari kopyalaniyor...")

        # Her bir deney klasorunu kopyala
        for root, dirs, files in os.walk(src_dir):
            if 'summary.csv' in files or 'auto_tracking_results.csv' in files:
                # Relative path: second_iteration/success/DATE/VIEW/...
                rel_path = os.path.relpath(root, src_dir)
                dest_path = os.path.join(RESULTS_DIR, "success", rel_path)

                # Hedef klasor yoksa olustur
                os.makedirs(dest_path, exist_ok=True)

                # Dosyalari kopyala
                for f in files:
                    src_file = os.path.join(root, f)
                    dst_file = os.path.join(dest_path, f)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)

                merged_count += 1

    print(f"\nToplam {merged_count} deney ana success klasorune eklendi")
    return merged_count


def collect_all_experiments():
    """Tum deneyleri topla (success + fail) - SADECE ana klasorler"""
    experiments = []
    seen_ids = set()  # Tekrarlari onle

    # SUCCESS - sadece ana success klasoru (iteration sonuclari zaten buraya kopyalandi)
    success_dir = os.path.join(RESULTS_DIR, "success")
    if os.path.exists(success_dir):
        for root, dirs, files in os.walk(success_dir):
            if 'summary.csv' in files or 'auto_tracking_results.csv' in files:
                rel_path = os.path.relpath(root, RESULTS_DIR)
                parts = rel_path.replace('\\', '/').split('/')

                if len(parts) >= 6:
                    exp_id = f"{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}"

                    if exp_id in seen_ids:
                        continue
                    seen_ids.add(exp_id)

                    # Video S3'te var ama lokalde yok, listeye ekle
                    files_with_video = list(files)
                    if 'output_video.mp4' not in files_with_video:
                        files_with_video.append('output_video.mp4')

                    exp = {
                        'id': exp_id,
                        'status': 'success',
                        'fail_reason': None,
                        'date': parts[1],
                        'view': parts[2],
                        'repeat': parts[3],
                        'category': parts[4],
                        'code': parts[5],
                        'path': rel_path.replace('\\', '/'),
                        'files': files_with_video
                    }

                    summary_path = os.path.join(root, 'summary.csv')
                    if os.path.exists(summary_path):
                        exp['metrics'] = parse_summary(summary_path)

                    experiments.append(exp)

    # FAIL - sadece ana fail klasoru
    # Iteration fail'leri dahil etme cunku:
    # 1. Success olanlar zaten ana success'e kopyalandi
    # 2. Hala fail olanlar iteration klasorlerinde, ana fail'de zaten var
    fail_dir = os.path.join(RESULTS_DIR, "fail")
    if os.path.exists(fail_dir):
        for root, dirs, files in os.walk(fail_dir):
            if 'summary.csv' in files or 'auto_tracking_results.csv' in files:
                rel_path = os.path.relpath(root, RESULTS_DIR)
                parts = rel_path.replace('\\', '/').split('/')

                # fail/reason/date/view/repeat/category/code
                if len(parts) >= 7:
                    exp_id = f"{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}_{parts[6]}"

                    # Bu deney artik success olduysa atlayalim
                    if exp_id in seen_ids:
                        continue
                    seen_ids.add(exp_id)

                    # Video S3'te var ama lokalde yok, listeye ekle
                    files_with_video = list(files)
                    if 'output_video.mp4' not in files_with_video:
                        files_with_video.append('output_video.mp4')

                    exp = {
                        'id': exp_id,
                        'status': 'fail',
                        'fail_reason': parts[1],
                        'date': parts[2],
                        'view': parts[3],
                        'repeat': parts[4],
                        'category': parts[5],
                        'code': parts[6],
                        'path': rel_path.replace('\\', '/'),
                        'files': files_with_video
                    }

                    summary_path = os.path.join(root, 'summary.csv')
                    if os.path.exists(summary_path):
                        exp['metrics'] = parse_summary(summary_path)

                    experiments.append(exp)

    return experiments


def parse_summary(summary_path):
    """Summary.csv dosyasini parse et"""
    metrics = {}
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ',' in line:
                    key, value = line.strip().split(',', 1)
                    metrics[key] = value
    except:
        pass
    return metrics


def upload_new_results():
    """Sadece yeni sonuc dosyalarini S3'e yukle"""

    # Iteration success klasorlerinden dosyalari yukle
    iterations = [
        "second_iteration/success",
        "third_iteration/success"
    ]

    uploaded_count = 0

    for iter_path in iterations:
        src_dir = os.path.join(RESULTS_DIR, iter_path)
        if not os.path.exists(src_dir):
            continue

        print(f"\n{iter_path} S3'e yukleniyor...")

        for root, dirs, files in os.walk(src_dir):
            for filename in files:
                # Sadece sonuc dosyalarini yukle (video haric)
                if filename.endswith(('.csv', '.jpg', '.png')):
                    local_path = os.path.join(root, filename)

                    # S3 key: success/DATE/VIEW/... formatinda
                    rel_from_iter = os.path.relpath(root, src_dir)
                    s3_key = f"success/{rel_from_iter}/{filename}".replace('\\', '/')

                    try:
                        content_type, _ = mimetypes.guess_type(local_path)
                        if content_type is None:
                            content_type = 'application/octet-stream'

                        s3.upload_file(
                            local_path,
                            BUCKET_NAME,
                            s3_key,
                            ExtraArgs={'ContentType': content_type}
                        )
                        uploaded_count += 1
                    except Exception as e:
                        print(f"Hata: {s3_key} - {e}")

        print(f"  {uploaded_count} dosya yuklendi")

    return uploaded_count


def update_experiments_json():
    """experiments.json'i guncelle"""

    print("\nTum deneyler toplanıyor...")
    experiments = collect_all_experiments()

    success_count = len([e for e in experiments if e['status'] == 'success'])
    fail_count = len([e for e in experiments if e['status'] == 'fail'])

    print(f"Toplam: {len(experiments)} deney")
    print(f"  Success: {success_count}")
    print(f"  Fail: {fail_count}")

    # Fail sebeplerini say
    fail_reasons = {}
    for exp in experiments:
        if exp['status'] == 'fail':
            reason = exp.get('fail_reason', 'unknown')
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    print("\nFail sebepleri:")
    for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # Metadata olustur
    metadata = {
        'total': len(experiments),
        'success': success_count,
        'fail': fail_count,
        'fail_reasons': fail_reasons,
        'experiments': experiments
    }

    # S3'e yukle
    print("\nexperiments.json S3'e yukleniyor...")
    metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key='experiments.json',
        Body=metadata_json.encode('utf-8'),
        ContentType='application/json'
    )

    # Lokale de kaydet
    local_json = os.path.join(RESULTS_DIR, 'experiments.json')
    with open(local_json, 'w', encoding='utf-8') as f:
        f.write(metadata_json)

    print(f"\nS3 URL: https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/experiments.json")

    return metadata


def main():
    print("=" * 60)
    print("S3 RESULTS UPDATER")
    print("Sadece yeni sonuclari yukler, videolar zaten S3'te")
    print("=" * 60)

    # 1. Iteration sonuclarini ana klasore kopyala
    print("\n1. Iteration sonuclari ana success klasorune kopyalaniyor...")
    merge_iteration_results()

    # 2. Yeni sonuc dosyalarini S3'e yukle
    print("\n2. Yeni sonuc dosyalari S3'e yukleniyor...")
    uploaded = upload_new_results()
    print(f"\nToplam {uploaded} dosya yuklendi")

    # 3. experiments.json'i guncelle
    print("\n3. experiments.json guncelleniyor...")
    metadata = update_experiments_json()

    print("\n" + "=" * 60)
    print("TAMAMLANDI!")
    print(f"Success: {metadata['success']}")
    print(f"Fail: {metadata['fail']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
