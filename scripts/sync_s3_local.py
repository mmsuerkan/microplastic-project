import os
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8')

PROCESSED_DIR = "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results"
S3_BUCKET = "microplastic-experiments"

def get_local_experiments():
    """Lokal deneyleri bul"""
    experiments = set()
    for root, dirs, files in os.walk(PROCESSED_DIR):
        if "auto_tracking_results.csv" in files:
            rel_path = os.path.relpath(root, PROCESSED_DIR).replace('\\', '/')
            experiments.add(rel_path)
    return experiments

def get_s3_experiments():
    """S3'teki deneyleri bul"""
    experiments = set()
    result = subprocess.run(
        ["aws", "s3", "ls", f"s3://{S3_BUCKET}/", "--recursive"],
        capture_output=True, text=True, timeout=300
    )
    for line in result.stdout.split('\n'):
        if 'output_video.mp4' in line:
            parts = line.split()
            if len(parts) >= 4:
                path = ' '.join(parts[3:])
                path = path.replace('/output_video.mp4', '')
                experiments.add(path)
    return experiments

def delete_s3_folder(path):
    """S3'ten klasor sil"""
    s3_path = f"s3://{S3_BUCKET}/{path}/"
    result = subprocess.run(
        ["aws", "s3", "rm", s3_path, "--recursive"],
        capture_output=True, text=True, timeout=60
    )
    return result.returncode == 0

def main():
    print("=" * 60)
    print("S3 VE LOKAL SENKRONIZASYONU")
    print("=" * 60)

    print("\nLokal deneyler taraniyor...")
    local = get_local_experiments()
    print(f"Lokal: {len(local)}")

    print("S3 deneyleri taraniyor...")
    s3 = get_s3_experiments()
    print(f"S3: {len(s3)}")

    # S3'te olup lokalde olmayan
    only_s3 = s3 - local
    print(f"\nSilinecek (sadece S3'te): {len(only_s3)}")

    if not only_s3:
        print("Silinecek bir sey yok!")
        return

    # Silme islemi
    deleted = 0
    failed = 0

    for i, path in enumerate(sorted(only_s3)):
        print(f"[{i+1}/{len(only_s3)}] Siliniyor: {path}")
        if delete_s3_folder(path):
            deleted += 1
        else:
            print(f"  HATA: Silinemedi!")
            failed += 1

    print("\n" + "=" * 60)
    print("OZET")
    print("=" * 60)
    print(f"Silinen: {deleted}")
    print(f"Basarisiz: {failed}")

    # Dogrulama
    print("\nDogrulama yapiliyor...")
    new_s3 = get_s3_experiments()
    print(f"Yeni S3 sayisi: {len(new_s3)}")
    print(f"Lokal sayisi: {len(local)}")

if __name__ == "__main__":
    main()
