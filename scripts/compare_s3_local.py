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
                # Son kisim path
                path = ' '.join(parts[3:])  # Bosluklu path'ler icin
                path = path.replace('/output_video.mp4', '')
                experiments.add(path)
    return experiments

def main():
    print("Lokal deneyler taraniyor...")
    local = get_local_experiments()
    print(f"Lokal: {len(local)}")

    print("\nS3 deneyleri taraniyor...")
    s3 = get_s3_experiments()
    print(f"S3: {len(s3)}")

    # Farklar
    only_s3 = s3 - local
    only_local = local - s3

    print(f"\n=== SADECE S3'TE OLAN ({len(only_s3)} adet) ===")
    for p in sorted(only_s3):
        print(f"  {p}")

    print(f"\n=== SADECE LOKALDE OLAN ({len(only_local)} adet) ===")
    for p in sorted(only_local):
        print(f"  {p}")

    print(f"\n=== OZET ===")
    print(f"Lokal: {len(local)}")
    print(f"S3: {len(s3)}")
    print(f"Sadece S3'te: {len(only_s3)}")
    print(f"Sadece lokalde: {len(only_local)}")
    print(f"Ortak: {len(local & s3)}")

if __name__ == "__main__":
    main()
