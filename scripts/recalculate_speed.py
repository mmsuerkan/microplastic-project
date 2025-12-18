"""
Tum deneylerin hiz hesabini yeniden yapar.
Mevcut summary.csv dosyalarindan verileri okur ve dogru formulu uygular.
"""
import os
import sys
import csv
import json
import boto3
from io import StringIO

sys.stdout.reconfigure(encoding='utf-8')

# S3 ayarlari
S3_BUCKET = 'microplastic-experiments'
s3 = boto3.client('s3')

# Sabitler
COLUMN_HEIGHT_M = 0.285  # 28.5 cm kolon yuksekligi
DEFAULT_FRAME_HEIGHT = 1080  # Varsayilan video yuksekligi

def get_all_experiments():
    """S3'den tum deneyleri listele"""
    experiments = []

    for prefix in ['success/', 'fail/']:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('summary.csv'):
                    experiments.append(obj['Key'])

    return experiments

def read_summary_from_s3(key):
    """S3'den summary.csv oku"""
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=key)
        content = response['Body'].read().decode('utf-8')

        reader = csv.reader(StringIO(content))
        data = {}
        for row in reader:
            if len(row) >= 2:
                data[row[0]] = row[1]
        return data
    except Exception as e:
        print(f"Hata okurken {key}: {e}")
        return None

def calculate_new_speed(data, frame_height=DEFAULT_FRAME_HEIGHT):
    """Yeni hiz hesabi"""
    try:
        dikey_hareket = float(data.get('Dikey Hareket', 0))
        toplam_frame = int(data.get('Toplam Frame', 0))
        video_suresi = float(data.get('Video Suresi', 0))

        if toplam_frame == 0 or video_suresi == 0:
            return None, None, None

        fps = toplam_frame / video_suresi
        pixels_per_meter = frame_height / COLUMN_HEIGHT_M

        # Takip frame sayisi - eger varsa kullan, yoksa toplam frame
        takip_frame = int(data.get('Takip Frame', toplam_frame))
        tracking_duration = takip_frame / fps

        actual_distance_m = abs(dikey_hareket) / pixels_per_meter

        if tracking_duration > 0:
            new_speed_mps = actual_distance_m / tracking_duration
        else:
            new_speed_mps = 0

        return new_speed_mps, actual_distance_m, tracking_duration

    except Exception as e:
        print(f"Hesaplama hatasi: {e}")
        return None, None, None

def update_summary_csv(key, data, new_speed_mps, actual_distance_m, tracking_duration):
    """Summary CSV'yi guncelle ve S3'e yukle"""
    try:
        # Yeni CSV olustur
        output = StringIO()
        writer = csv.writer(output)

        # Mevcut verileri koru, hiz bilgilerini guncelle
        writer.writerow(['Metrik', 'Deger', 'Birim'])
        writer.writerow(['Video', data.get('Video', ''), ''])

        if 'Tracker Version' in data:
            writer.writerow(['Tracker Version', data.get('Tracker Version', ''), ''])
        if 'Tespit Frame' in data:
            writer.writerow(['Tespit Frame', data.get('Tespit Frame', ''), 'frame'])

        writer.writerow(['Toplam Frame', data.get('Toplam Frame', ''), 'frame'])

        # Takip frame ekle (eger yoksa toplam frame kullan)
        takip_frame = data.get('Takip Frame', data.get('Toplam Frame', ''))
        writer.writerow(['Takip Frame', takip_frame, 'frame'])

        writer.writerow(['Video Suresi', data.get('Video Suresi', ''), 'saniye'])
        writer.writerow(['Takip Suresi', f'{tracking_duration:.2f}', 'saniye'])
        writer.writerow(['Baslangic X', data.get('Baslangic X', ''), 'piksel'])
        writer.writerow(['Baslangic Y', data.get('Baslangic Y', ''), 'piksel'])
        writer.writerow(['Bitis X', data.get('Bitis X', ''), 'piksel'])
        writer.writerow(['Bitis Y', data.get('Bitis Y', ''), 'piksel'])
        writer.writerow(['Dikey Hareket', data.get('Dikey Hareket', ''), 'piksel'])
        writer.writerow(['Kat Edilen Mesafe', f'{actual_distance_m*100:.2f}', 'cm'])
        writer.writerow(['Yatay Sapma', data.get('Yatay Sapma', ''), 'piksel'])
        writer.writerow(['Mean Magnitude', data.get('Mean Magnitude', ''), 'px/frame'])

        if 'Ortalama dX' in data:
            writer.writerow(['Ortalama dX', data.get('Ortalama dX', ''), 'px/frame'])
        writer.writerow(['Ortalama dY', data.get('Ortalama dY', ''), 'px/frame'])

        # Yeni hiz
        writer.writerow(['Hiz', f'{new_speed_mps*100:.2f}', 'cm/s'])

        # S3'e yukle
        csv_content = output.getvalue()
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=csv_content.encode('utf-8'),
            ContentType='text/csv'
        )

        return True
    except Exception as e:
        print(f"Guncelleme hatasi {key}: {e}")
        return False

def main():
    print("Tum deneylerin hiz hesabi guncelleniyor...")
    print(f"Kolon yuksekligi: {COLUMN_HEIGHT_M*100} cm")
    print(f"Varsayilan frame yuksekligi: {DEFAULT_FRAME_HEIGHT} px")
    print("="*60)

    experiments = get_all_experiments()
    print(f"Toplam {len(experiments)} deney bulundu")

    updated = 0
    failed = 0
    skipped = 0

    speed_changes = []

    for i, key in enumerate(experiments):
        if (i + 1) % 100 == 0:
            print(f"Ilerleme: {i+1}/{len(experiments)}")

        data = read_summary_from_s3(key)
        if data is None:
            failed += 1
            continue

        new_speed, actual_distance, tracking_duration = calculate_new_speed(data)
        if new_speed is None:
            skipped += 1
            continue

        # Eski hizi al (varsa)
        old_speed_str = data.get('Tahmini Hiz', data.get('Hiz', '0'))
        try:
            # cm/s veya m/s olabilir
            old_speed = float(old_speed_str.replace(',', '.'))
            if old_speed < 1:  # m/s ise cm/s'e cevir
                old_speed *= 100
        except:
            old_speed = 0

        # Degisimi kaydet
        speed_changes.append({
            'key': key,
            'old': old_speed,
            'new': new_speed * 100,  # cm/s
            'diff': (new_speed * 100) - old_speed
        })

        # Guncelle
        if update_summary_csv(key, data, new_speed, actual_distance, tracking_duration):
            updated += 1
        else:
            failed += 1

    print("="*60)
    print(f"SONUC:")
    print(f"  Guncellenen: {updated}")
    print(f"  Basarisiz: {failed}")
    print(f"  Atlanan: {skipped}")

    # Istatistikler
    if speed_changes:
        diffs = [s['diff'] for s in speed_changes]
        avg_diff = sum(diffs) / len(diffs)
        print(f"\nHiz Degisimi Istatistikleri:")
        print(f"  Ortalama fark: {avg_diff:.2f} cm/s")
        print(f"  Min fark: {min(diffs):.2f} cm/s")
        print(f"  Max fark: {max(diffs):.2f} cm/s")

        # Ornekler
        print(f"\nOrnek degisimler (ilk 5):")
        for s in speed_changes[:5]:
            print(f"  {s['old']:.2f} -> {s['new']:.2f} cm/s (fark: {s['diff']:.2f})")

if __name__ == '__main__':
    main()
