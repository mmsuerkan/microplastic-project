"""Fail deneylerden kullanılabilir olanları training data'ya ekle"""
import os
import csv
import pandas as pd
import numpy as np
import json
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ================================================================
# 1. FAIL DENEYLERDEN VELOCİTY ÇEK
# ================================================================
print("="*70)
print("1. FAIL DENEYLERDEN VELOCITY ÇEK")
print("="*70)

fail_dir = 'processed_results/fail'
pmma_cats = ['BSP', 'C', 'HC', 'WSP']

fail_experiments = []  # [(cat, code, velocity, path), ...]

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
                        for i, p in enumerate(parts):
                            if p in pmma_cats and i+1 < len(parts):
                                cat = p
                                code = parts[i+1]
                                if velocity > 0.5:  # 0.5 cm/s üzeri kullanılabilir
                                    fail_experiments.append({
                                        'category': cat,
                                        'code': code,
                                        'velocity': velocity,
                                        'path': root
                                    })
                                break
        except:
            pass

print(f"Toplam kullanılabilir fail deney: {len(fail_experiments)}")

# Kategori bazlı özet
cat_summary = {}
for exp in fail_experiments:
    cat = exp['category']
    if cat not in cat_summary:
        cat_summary[cat] = []
    cat_summary[cat].append(exp)

for cat in pmma_cats:
    if cat in cat_summary:
        unique_codes = set(e['code'] for e in cat_summary[cat])
        print(f"  {cat}: {len(cat_summary[cat])} deney, {len(unique_codes)} unique parçacık")

# ================================================================
# 2. EXCEL'DEN BOYUT/DENSITY BİLGİLERİNİ AL
# ================================================================
print("\n" + "="*70)
print("2. EXCEL'DEN BOYUT/DENSITY BİLGİLERİNİ AL")
print("="*70)

excel_path = r'c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS (2).xlsx'

# PMMA kolon konfigürasyonu
pmma_config = {
    'BSP': {
        'sheet': 'PMMA BSP',
        'shape': 'Box Shape Prism',
        'shape_enc': 4,
        'a_col': 'a (mm).3',
        'b_col': 'b (mm).3',
        'c_col': 'c (mm).3',
    },
    'C': {
        'sheet': 'PMMA Cylinder',
        'shape': 'Cylinder',
        'shape_enc': 0,
        'a_col': 'Diameter (mm)',
        'b_col': 'Diameter (mm)',
        'c_col': 'Height(mm)',
    },
    'HC': {
        'sheet': 'PMMA Half Cylinder',
        'shape': 'Half Cylinder',
        'shape_enc': 1,
        'a_col': 'Diameter (mm).1',
        'b_col': 'Diameter (mm).1',
        'c_col': 'Height (mm).1',
    },
    'WSP': {
        'sheet': 'PMMA Wedge-Shaped',
        'shape': 'Wedge Shape Prism',
        'shape_enc': 3,
        'a_col': 'a (mm).3',
        'b_col': 'b (mm).3',
        'c_col': 'c (mm).3',
    },
}

# Excel'den tüm PMMA verilerini oku
excel_data = {}  # {cat: {num: {a, b, c, density}}}

xl = pd.ExcelFile(excel_path)

for cat, config in pmma_config.items():
    sheet_name = config['sheet']
    df = pd.read_excel(xl, sheet_name=sheet_name, header=1)
    cols = df.columns.tolist()

    # Kolon indekslerini bul
    a_idx = None
    b_idx = None
    c_idx = None
    density_idx = None

    for i, col in enumerate(cols):
        col_str = str(col).strip()
        if col_str == config['a_col']:
            a_idx = i
        if col_str == config['b_col']:
            b_idx = i
        if col_str == config['c_col']:
            c_idx = i
        if 'Density (kg/m' in col_str:
            density_idx = i

    excel_data[cat] = {}

    for idx, row in df.iterrows():
        code = row.iloc[1]  # Shape kolonu
        if pd.notna(code):
            match = re.search(r'(\d+)', str(code))
            if match:
                num = int(match.group(1))
                a = row.iloc[a_idx] if a_idx and pd.notna(row.iloc[a_idx]) else None
                b = row.iloc[b_idx] if b_idx and pd.notna(row.iloc[b_idx]) else None
                c = row.iloc[c_idx] if c_idx and pd.notna(row.iloc[c_idx]) else None
                density = row.iloc[density_idx] if density_idx and pd.notna(row.iloc[density_idx]) else None

                if a and b and c and density:
                    excel_data[cat][num] = {
                        'a': float(a),
                        'b': float(b),
                        'c': float(c),
                        'density': float(density)
                    }

for cat in pmma_cats:
    print(f"  {cat}: {len(excel_data.get(cat, {}))} parçacık Excel'de")

# ================================================================
# 3. PARÇACIK BAZLI ORTALAMA VELOCITY HESAPLA
# ================================================================
print("\n" + "="*70)
print("3. PARÇACIK BAZLI ORTALAMA VELOCITY HESAPLA")
print("="*70)

# Aynı parçacığın birden fazla ölçümünü ortala
particle_velocities = {}  # {(cat, num): [velocities]}

for exp in fail_experiments:
    cat = exp['category']
    code = exp['code']
    velocity = exp['velocity']

    match = re.search(r'(\d+)', code)
    if match:
        num = int(match.group(1))
        key = (cat, num)
        if key not in particle_velocities:
            particle_velocities[key] = []
        particle_velocities[key].append(velocity)

print(f"Toplam unique parçacık (fail'den): {len(particle_velocities)}")

# ================================================================
# 4. MEVCUT TRAİNİNG DATA'YI OKU
# ================================================================
print("\n" + "="*70)
print("4. MEVCUT TRAINING DATA'YI OKU")
print("="*70)

existing_df = pd.read_csv('data/training_data_particle_avg.csv')
print(f"Mevcut training data: {len(existing_df)} parçacık")

# Mevcut PMMA parçacıkları
existing_pmma = set()
for idx, row in existing_df.iterrows():
    if row['category'] in ['BSP', 'C', 'HC', 'WSP']:
        match = re.search(r'(\d+)', row['code'])
        if match:
            num = int(match.group(1))
            existing_pmma.add((row['category'], num))

print(f"Mevcut PMMA parçacık: {len(existing_pmma)}")

# ================================================================
# 5. YENİ PARÇACIKLARI EKLE
# ================================================================
print("\n" + "="*70)
print("5. YENİ PARÇACIKLARI EKLE")
print("="*70)

new_rows = []
skipped_existing = 0
skipped_no_excel = 0

for (cat, num), velocities in particle_velocities.items():
    # Zaten var mı?
    if (cat, num) in existing_pmma:
        skipped_existing += 1
        continue

    # Excel'de var mı?
    if cat not in excel_data or num not in excel_data[cat]:
        skipped_no_excel += 1
        continue

    # Velocity ortalaması
    velocity_mean = np.mean(velocities)
    velocity_std = np.std(velocities) if len(velocities) > 1 else 0

    # Excel'den boyut/density
    excel_info = excel_data[cat][num]
    a = excel_info['a']
    b = excel_info['b']
    c = excel_info['c']
    density = excel_info['density']

    # Türetilmiş özellikler
    config = pmma_config[cat]

    # Volume hesapla (şekle göre)
    if config['shape'] == 'Cylinder':
        volume = np.pi * (a/2)**2 * c  # π * r² * h
        surface_area = 2 * np.pi * (a/2) * c + 2 * np.pi * (a/2)**2
    elif config['shape'] == 'Half Cylinder':
        volume = 0.5 * np.pi * (a/2)**2 * c
        surface_area = np.pi * (a/2) * c + np.pi * (a/2)**2 + a * c
    elif config['shape'] == 'Box Shape Prism':
        volume = a * b * c
        surface_area = 2 * (a*b + b*c + a*c)
    elif config['shape'] == 'Wedge Shape Prism':
        volume = 0.5 * a * b * c
        surface_area = a*b + a*c + b*c + np.sqrt(a**2 + b**2) * c
    else:
        volume = a * b * c
        surface_area = 2 * (a*b + b*c + a*c)

    aspect_ratio = max(a, b, c) / min(a, b, c) if min(a, b, c) > 0 else 1
    vol_surf_ratio = volume / surface_area if surface_area > 0 else 0

    new_rows.append({
        'category': cat,
        'code': f"{cat}-{num}",
        'shape_name': config['shape'],
        'shape_enc': config['shape_enc'],
        'a': a,
        'b': b,
        'c': c,
        'density': density,
        'volume': volume,
        'surface_area': surface_area,
        'aspect_ratio': aspect_ratio,
        'vol_surf_ratio': vol_surf_ratio,
        'velocity_mean': velocity_mean,
        'velocity_std': velocity_std,
        'measurement_count': len(velocities),
        'source': 'fail_recovery'  # Nereden geldiğini işaretle
    })

print(f"Eklenen yeni parçacık: {len(new_rows)}")
print(f"Atlandı (zaten var): {skipped_existing}")
print(f"Atlandı (Excel'de yok): {skipped_no_excel}")

# Kategori bazlı
new_by_cat = {}
for row in new_rows:
    cat = row['category']
    if cat not in new_by_cat:
        new_by_cat[cat] = 0
    new_by_cat[cat] += 1

print("\nKategori bazlı eklenen:")
for cat in pmma_cats:
    print(f"  {cat}: {new_by_cat.get(cat, 0)} yeni parçacık")

# ================================================================
# 6. YENİ TRAINING DATA KAYDET
# ================================================================
print("\n" + "="*70)
print("6. YENİ TRAINING DATA KAYDET")
print("="*70)

# Mevcut verilere source kolonu ekle
existing_df['source'] = 'original'

# Yeni satırları DataFrame'e çevir
if new_rows:
    new_df = pd.DataFrame(new_rows)

    # Kolonları eşleştir
    for col in existing_df.columns:
        if col not in new_df.columns:
            new_df[col] = None

    # Birleştir
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
else:
    combined_df = existing_df

print(f"Toplam parçacık: {len(combined_df)}")
print(f"  - Mevcut: {len(existing_df)}")
print(f"  - Yeni: {len(new_rows)}")

# Kaydet
output_path = 'data/training_data_with_fails.csv'
combined_df.to_csv(output_path, index=False)
print(f"\n✓ Kaydedildi: {output_path}")

# Şekil dağılımı
print("\n--- Şekil Dağılımı (Güncel) ---")
shape_dist = combined_df.groupby('shape_name').size()
print(shape_dist.to_string())

# PMMA özet
print("\n--- PMMA Özet ---")
pmma_df = combined_df[combined_df['category'].isin(pmma_cats)]
print(f"Toplam PMMA parçacık: {len(pmma_df)}")
print(pmma_df.groupby('category').size().to_string())
