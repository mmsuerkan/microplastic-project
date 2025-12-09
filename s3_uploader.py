"""
S3 Uploader - Deney sonuclarini S3'e yukler
"""
import os
import sys
import json
import boto3
from pathlib import Path
import mimetypes

# UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# S3 ayarlari
BUCKET_NAME = "microplastic-experiments"
REGION = "us-east-1"
RESULTS_DIR = "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results"

# S3 client
s3 = boto3.client('s3', region_name=REGION)

def create_bucket_if_not_exists():
    """Bucket yoksa olustur"""
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket mevcut: {BUCKET_NAME}")
    except:
        print(f"Bucket olusturuluyor: {BUCKET_NAME}")
        if REGION == 'us-east-1':
            s3.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3.create_bucket(
                Bucket=BUCKET_NAME,
                CreateBucketConfiguration={'LocationConstraint': REGION}
            )

        # Public access block kaldir
        s3.delete_public_access_block(Bucket=BUCKET_NAME)

        # Public access icin policy
        bucket_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Sid": "PublicReadGetObject",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{BUCKET_NAME}/*"
            }]
        }
        s3.put_bucket_policy(Bucket=BUCKET_NAME, Policy=json.dumps(bucket_policy))

        # CORS ayarlari
        cors_config = {
            'CORSRules': [{
                'AllowedHeaders': ['*'],
                'AllowedMethods': ['GET', 'HEAD'],
                'AllowedOrigins': ['*'],
                'ExposeHeaders': []
            }]
        }
        s3.put_bucket_cors(Bucket=BUCKET_NAME, CORSConfiguration=cors_config)
        print("Bucket olusturuldu ve yapilandirildi")

def upload_file(local_path, s3_key):
    """Dosyayı S3'e yükle"""
    content_type, _ = mimetypes.guess_type(local_path)
    if content_type is None:
        content_type = 'application/octet-stream'

    extra_args = {'ContentType': content_type}

    s3.upload_file(local_path, BUCKET_NAME, s3_key, ExtraArgs=extra_args)

def collect_experiments():
    """Tüm deneyleri topla ve metadata oluştur"""
    experiments = []

    for status in ['success', 'fail']:
        status_dir = os.path.join(RESULTS_DIR, status)
        if not os.path.exists(status_dir):
            continue

        if status == 'success':
            # success/date/view/repeat/category/code
            for root, dirs, files in os.walk(status_dir):
                if 'summary.csv' in files:
                    rel_path = os.path.relpath(root, RESULTS_DIR)
                    parts = rel_path.replace('\\', '/').split('/')

                    if len(parts) >= 6:
                        exp = {
                            'id': parts[-1],
                            'status': 'success',
                            'fail_reason': None,
                            'date': parts[1],
                            'view': parts[2],
                            'repeat': parts[3],
                            'category': parts[4],
                            'code': parts[5],
                            'path': rel_path.replace('\\', '/'),
                            'files': files
                        }

                        # Summary.csv'den verileri oku
                        summary_path = os.path.join(root, 'summary.csv')
                        if os.path.exists(summary_path):
                            exp['metrics'] = parse_summary(summary_path)

                        experiments.append(exp)
        else:
            # fail/reason/date/view/repeat/category/code
            for root, dirs, files in os.walk(status_dir):
                if 'summary.csv' in files or 'auto_tracking_results.csv' in files:
                    rel_path = os.path.relpath(root, RESULTS_DIR)
                    parts = rel_path.replace('\\', '/').split('/')

                    if len(parts) >= 7:
                        exp = {
                            'id': parts[-1],
                            'status': 'fail',
                            'fail_reason': parts[1],
                            'date': parts[2],
                            'view': parts[3],
                            'repeat': parts[4],
                            'category': parts[5],
                            'code': parts[6],
                            'path': rel_path.replace('\\', '/'),
                            'files': files
                        }

                        summary_path = os.path.join(root, 'summary.csv')
                        if os.path.exists(summary_path):
                            exp['metrics'] = parse_summary(summary_path)

                        experiments.append(exp)

    return experiments

def parse_summary(summary_path):
    """Summary.csv dosyasını parse et"""
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

def upload_experiments():
    """Tüm deneyleri S3'e yükle"""
    print("Bucket kontrol ediliyor...")
    create_bucket_if_not_exists()

    print("\nDeneyler toplanıyor...")
    experiments = collect_experiments()
    print(f"Toplam {len(experiments)} deney bulundu")

    # Metadata JSON oluştur
    metadata = {
        'total': len(experiments),
        'success': len([e for e in experiments if e['status'] == 'success']),
        'fail': len([e for e in experiments if e['status'] == 'fail']),
        'experiments': experiments
    }

    # Experiments JSON'ı yükle
    print("\nMetadata yükleniyor...")
    metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key='experiments.json',
        Body=metadata_json.encode('utf-8'),
        ContentType='application/json'
    )

    # Dosyaları yükle
    print("\nDosyalar yükleniyor...")
    total_files = 0
    for i, exp in enumerate(experiments, 1):
        local_dir = os.path.join(RESULTS_DIR, exp['path'])

        for filename in exp['files']:
            local_path = os.path.join(local_dir, filename)
            s3_key = f"{exp['path']}/{filename}"

            try:
                upload_file(local_path, s3_key)
                total_files += 1
            except Exception as e:
                print(f"Hata: {s3_key} - {e}")

        if i % 10 == 0:
            print(f"[{i}/{len(experiments)}] yüklendi...")

    print(f"\n{total_files} dosya yüklendi!")
    print(f"\nS3 URL: https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/experiments.json")

    return metadata

if __name__ == "__main__":
    upload_experiments()
