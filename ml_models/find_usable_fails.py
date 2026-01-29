"""Fail deneylerden kullanılabilir olanları bul"""
import os
import csv
import sys
sys.stdout.reconfigure(encoding='utf-8')

fail_dir = 'processed_results/fail'
usable = []
unusable = 0

for root, dirs, files in os.walk(fail_dir):
    if 'summary.csv' in files:
        summary_path = os.path.join(root, 'summary.csv')
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2 and row[0] == 'Hiz':
                        velocity = float(row[1])
                        # Kategori ve kodu bul
                        parts = root.replace('\\', '/').split('/')
                        # BSP/BSP-1 gibi yapıdan çıkar
                        for i, p in enumerate(parts):
                            if p in ['BSP', 'C', 'HC', 'WSP'] and i+1 < len(parts):
                                cat = p
                                code = parts[i+1]
                                if velocity > 0.5:  # 0.5 cm/s üzeri kullanılabilir
                                    usable.append((cat, code, velocity, root))
                                else:
                                    unusable += 1
                                break
        except Exception as e:
            pass

print(f'Kullanılamaz (velocity <= 0.5): {unusable}')
print(f'Kullanılabilir (velocity > 0.5): {len(usable)}')

if usable:
    print(f'\n--- Kullanılabilir Fail Deneyler (velocity > 0.5 cm/s) ---')
    for cat, code, vel, path in sorted(usable, key=lambda x: -x[2])[:30]:
        print(f'  {cat}/{code}: {vel:.2f} cm/s')

    # Kategori bazlı özet
    print(f'\n--- Kategori Bazlı Özet ---')
    cat_counts = {}
    for cat, code, vel, path in usable:
        if cat not in cat_counts:
            cat_counts[cat] = set()
        cat_counts[cat].add(code)

    for cat in sorted(cat_counts.keys()):
        print(f'  {cat}: {len(cat_counts[cat])} unique parçacık')
