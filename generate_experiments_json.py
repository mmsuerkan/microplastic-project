import os
import json
import csv
import sys

sys.stdout.reconfigure(encoding='utf-8')

PROCESSED_DIR = "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results"
OUTPUT_FILE = "C:/Users/mmert/PycharmProjects/ObjectTrackingProject/processed_results/experiments.json"

def parse_summary_csv(csv_path):
    """summary.csv dosyasından metrikleri oku"""
    metrics = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    key = row[0].strip()
                    value = row[1].strip() if len(row) > 1 else ""
                    unit = row[2].strip() if len(row) > 2 else ""
                    metrics[key] = f"{value} {unit}".strip()
    except Exception as e:
        pass
    return metrics

def parse_tracking_csv(csv_path):
    """auto_tracking_results.csv'den özet metrikleri çıkar"""
    metrics = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last = rows[-1]
                metrics['total_frames'] = len(rows)
                metrics['vertical_pixels'] = last.get('vertical_pixels', '')
                metrics['horizontal_drift'] = last.get('horizontal_drift', '')
                metrics['dominant_frequency_hz'] = last.get('dominant_frequency_hz', '')
                metrics['oscillation_amplitude'] = last.get('oscillation_amplitude', '')
                metrics['oscillation_detected'] = last.get('oscillation_detected', '')
    except Exception as e:
        pass
    return metrics

def scan_experiments():
    """Tüm deneyleri tara"""
    experiments = []

    # Success klasörü
    success_dir = os.path.join(PROCESSED_DIR, "success")
    if os.path.exists(success_dir):
        for root, dirs, files in os.walk(success_dir):
            if "auto_tracking_results.csv" in files:
                rel_path = os.path.relpath(root, PROCESSED_DIR).replace('\\', '/')
                parts = rel_path.split('/')
                # success/tarih/view/repeat/category/code
                if len(parts) >= 6:
                    exp = {
                        'status': 'success',
                        'fail_reason': None,
                        'date': parts[1],
                        'view': parts[2],
                        'repeat': parts[3],
                        'category': parts[4],
                        'code': parts[5],
                        'path': rel_path,
                        'files': files
                    }

                    # Metrikleri oku
                    summary_path = os.path.join(root, "summary.csv")
                    if os.path.exists(summary_path):
                        exp['metrics'] = parse_summary_csv(summary_path)

                    tracking_path = os.path.join(root, "auto_tracking_results.csv")
                    tracking_metrics = parse_tracking_csv(tracking_path)
                    if 'metrics' not in exp:
                        exp['metrics'] = {}
                    exp['metrics'].update(tracking_metrics)

                    experiments.append(exp)

    # Fail klasörü
    fail_dir = os.path.join(PROCESSED_DIR, "fail")
    if os.path.exists(fail_dir):
        for reason_folder in os.listdir(fail_dir):
            reason_path = os.path.join(fail_dir, reason_folder)
            if not os.path.isdir(reason_path):
                continue

            for root, dirs, files in os.walk(reason_path):
                if "auto_tracking_results.csv" in files:
                    rel_path = os.path.relpath(root, PROCESSED_DIR).replace('\\', '/')
                    parts = rel_path.split('/')
                    # fail/reason/tarih/view/repeat/category/code
                    if len(parts) >= 7:
                        exp = {
                            'status': 'fail',
                            'fail_reason': parts[1],
                            'date': parts[2],
                            'view': parts[3],
                            'repeat': parts[4],
                            'category': parts[5],
                            'code': parts[6],
                            'path': rel_path,
                            'files': files
                        }

                        # Metrikleri oku
                        summary_path = os.path.join(root, "summary.csv")
                        if os.path.exists(summary_path):
                            exp['metrics'] = parse_summary_csv(summary_path)

                        tracking_path = os.path.join(root, "auto_tracking_results.csv")
                        tracking_metrics = parse_tracking_csv(tracking_path)
                        if 'metrics' not in exp:
                            exp['metrics'] = {}
                        exp['metrics'].update(tracking_metrics)

                        experiments.append(exp)

    return experiments

def main():
    print("Deneyler taraniyor...")
    experiments = scan_experiments()

    success_count = sum(1 for e in experiments if e['status'] == 'success')
    fail_count = sum(1 for e in experiments if e['status'] == 'fail')

    # Tarihe ve koda göre sırala
    experiments.sort(key=lambda x: (x['date'], x['category'], x['code']))

    # ID ekle
    for i, exp in enumerate(experiments):
        exp['id'] = f"{exp['code']}-{exp['date']}-{exp['view']}-{exp['repeat']}"

    data = {
        'total': len(experiments),
        'success': success_count,
        'fail': fail_count,
        'generated_at': __import__('datetime').datetime.now().isoformat(),
        'experiments': experiments
    }

    # JSON'a yaz
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nTamamlandi!")
    print(f"Toplam: {len(experiments)}")
    print(f"Basarili: {success_count}")
    print(f"Basarisiz: {fail_count}")
    print(f"\nKaydedildi: {OUTPUT_FILE}")

    # Fail reason dağılımı
    fail_reasons = {}
    for e in experiments:
        if e['status'] == 'fail':
            reason = e['fail_reason']
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    if fail_reasons:
        print("\nBasarisizlik nedenleri:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

if __name__ == "__main__":
    main()
