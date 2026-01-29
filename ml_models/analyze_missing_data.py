"""Eksik veri analizi - Numara bazlı eşleştirme"""
import pandas as pd
import json
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ================================================================
# 1. EXCEL'DEN VERİ ÇEK (Density bilgisi var)
# ================================================================
print("="*70)
print("1. EXCEL'DEKİ PARÇACIKLAR (Density var)")
print("="*70)

excel_path = r'c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS (2).xlsx'
xl = pd.ExcelFile(excel_path)

# Sheet -> Category eşleştirmesi
sheet_mapping = {
    'ABS CYLINDER': ('ABS', 'Cylinder'),
    'ABS HC': ('ABS', 'Half Cylinder'),
    'PLA CYLINDER ': ('PLA', 'Cylinder'),
    'PLA CUBE ': ('PLA', 'Cube'),
    'PLA HC ': ('PLA', 'Half Cylinder'),
    'PS EC ': ('PS', 'Elliptic Cylinder'),
    'RESIN (a=9 mm)': ('RESIN_9', 'Mixed'),
    'RESIN (a=6 mm) ': ('RESIN_6', 'Mixed'),
    'P6 BSP ': ('PA6', 'Box Shape Prism'),
    'P6 HC ': ('PA6', 'Half Cylinder'),
    'P6 CYLINDER ': ('PA6', 'Cylinder'),
    'PMMA BSP': ('PMMA', 'Box Shape Prism'),
    'PMMA Cylinder': ('PMMA', 'Cylinder'),
    'PMMA Wedge-Shaped': ('PMMA', 'Wedge Shape Prism'),
    'PMMA Half Cylinder': ('PMMA', 'Half Cylinder'),
}

excel_particles = []

for sheet in xl.sheet_names:
    sheet_clean = sheet.strip()
    if sheet_clean not in [k.strip() for k in sheet_mapping.keys()]:
        continue

    # Mapping bul
    for k, v in sheet_mapping.items():
        if k.strip() == sheet_clean:
            material, shape = v
            break

    df = pd.read_excel(xl, sheet_name=sheet, header=None)

    for idx in range(2, len(df)):
        code = df.iloc[idx, 1]  # Shape/Code kolonu
        if pd.notna(code) and str(code).strip():
            code_str = str(code).strip()

            # Numarayı çıkar
            match = re.search(r'(\d+)', code_str)
            if match:
                num = int(match.group(1))
                excel_particles.append({
                    'sheet': sheet.strip(),
                    'material': material,
                    'shape': shape,
                    'code': code_str,
                    'number': num,
                })

print(f"Excel'de toplam: {len(excel_particles)} parçacık")

# Material bazlı özet
excel_by_material = {}
for p in excel_particles:
    key = f"{p['material']}_{p['shape']}"
    if key not in excel_by_material:
        excel_by_material[key] = []
    excel_by_material[key].append(p['number'])

print("\nMaterial/Shape bazlı:")
for key in sorted(excel_by_material.keys()):
    nums = sorted(excel_by_material[key])
    print(f"  {key}: {len(nums)} parçacık ({min(nums)}-{max(nums)})")

# ================================================================
# 2. EXPERIMENTS.JSON'DAN VERİ ÇEK (Velocity var)
# ================================================================
print("\n" + "="*70)
print("2. EXPERIMENTS.JSON'DAKİ PARÇACIKLAR (Velocity var)")
print("="*70)

with open('processed_results/experiments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Category -> (Material, Shape) eşleştirmesi
# PMMA kategorileri: BSP, C, HC, WSP
category_mapping = {
    'ABS C': ('ABS', 'Cylinder'),
    'ABS CUBE': ('ABS', 'Cube'),
    'ABS EC': ('ABS', 'Elliptic Cylinder'),
    'ABS HC': ('ABS', 'Half Cylinder'),
    'BSP': ('PMMA', 'Box Shape Prism'),      # PMMA BSP
    'C': ('PMMA', 'Cylinder'),                # PMMA Cylinder
    'HC': ('PMMA', 'Half Cylinder'),          # PMMA Half Cylinder
    'WSP': ('PMMA', 'Wedge Shape Prism'),     # PMMA Wedge Shape Prism
    'PA 6': ('PA6', 'Mixed'),
    'PLA C': ('PLA', 'Cylinder'),
    'PLA CUBE': ('PLA', 'Cube'),
    'PLA HC': ('PLA', 'Half Cylinder'),
    'PS': ('PS', 'Elliptic Cylinder'),
    'RESIN (a=6 r=3)': ('RESIN_6', 'Mixed'),
    'RESIN (a=9 r=4.5)': ('RESIN_9', 'Mixed'),
}

exp_particles = {}

for exp in data['experiments']:
    if exp['status'] != 'success':
        continue

    cat = exp['category']
    code = exp['code']

    if cat not in category_mapping:
        continue

    material, shape = category_mapping[cat]

    # Numarayı çıkar
    match = re.search(r'(\d+)', code)
    if match:
        num = int(match.group(1))

        # Shape'i code'dan belirle (PA6 ve RESIN için)
        if material in ['PA6', 'RESIN_6', 'RESIN_9']:
            if 'BSP' in code:
                shape = 'Box Shape Prism'
            elif 'HC' in code:
                shape = 'Half Cylinder'
            elif 'CUBE' in code:
                shape = 'Cube'
            elif 'SP' in code:
                shape = 'Sphere'
            elif 'C-' in code or code.startswith('C'):
                shape = 'Cylinder'
            elif 'D-' in code:
                shape = 'Elliptic Cylinder'

        key = f"{material}_{shape}"
        if key not in exp_particles:
            exp_particles[key] = set()
        exp_particles[key].add(num)

print(f"Experiments'da toplam unique parçacık sayısı:")
total_exp = sum(len(v) for v in exp_particles.values())
print(f"  {total_exp} parçacık")

print("\nMaterial/Shape bazlı:")
for key in sorted(exp_particles.keys()):
    nums = sorted(exp_particles[key])
    print(f"  {key}: {len(nums)} parçacık ({min(nums)}-{max(nums) if nums else 0})")

# ================================================================
# 3. EKSİK VERİLERİ BUL
# ================================================================
print("\n" + "="*70)
print("3. EKSİK VERİLER (Excel'de var, Velocity yok)")
print("="*70)

missing = []
matched = []

for p in excel_particles:
    key = f"{p['material']}_{p['shape']}"
    num = p['number']

    if key in exp_particles and num in exp_particles[key]:
        matched.append(p)
    else:
        missing.append(p)

print(f"\nEşleşen: {len(matched)} parçacık (velocity ölçümü VAR)")
print(f"Eksik:   {len(missing)} parçacık (velocity ölçümü YOK)")

# Eksikleri material bazlı grupla
missing_by_material = {}
for p in missing:
    key = f"{p['material']}_{p['shape']}"
    if key not in missing_by_material:
        missing_by_material[key] = []
    missing_by_material[key].append(p['number'])

print("\n--- Eksik Parçacıklar (Material/Shape bazlı) ---")
for key in sorted(missing_by_material.keys()):
    nums = sorted(missing_by_material[key])
    print(f"\n{key}: {len(nums)} eksik")
    if len(nums) <= 20:
        print(f"  Numaralar: {nums}")
    else:
        print(f"  Numaralar: {nums[:10]}...{nums[-5:]}")

# ================================================================
# 4. ÖZET TABLO
# ================================================================
print("\n" + "="*70)
print("4. ÖZET TABLO")
print("="*70)

print(f"\n{'Material/Shape':<30} {'Excel':<8} {'Velocity':<10} {'Eksik':<8} {'%':<8}")
print("-" * 65)

all_keys = set(excel_by_material.keys()) | set(exp_particles.keys())
total_excel = 0
total_velocity = 0
total_missing = 0

for key in sorted(all_keys):
    excel_count = len(excel_by_material.get(key, []))
    velocity_count = len(exp_particles.get(key, set()))
    missing_count = len(missing_by_material.get(key, []))

    if excel_count > 0:
        pct = (velocity_count / excel_count) * 100
        print(f"{key:<30} {excel_count:<8} {velocity_count:<10} {missing_count:<8} {pct:.0f}%")
        total_excel += excel_count
        total_velocity += velocity_count
        total_missing += missing_count

print("-" * 65)
pct_total = (total_velocity / total_excel) * 100 if total_excel > 0 else 0
print(f"{'TOPLAM':<30} {total_excel:<8} {total_velocity:<10} {total_missing:<8} {pct_total:.0f}%")

# ================================================================
# 5. ÖNCELİKLİ EKSİKLER
# ================================================================
print("\n" + "="*70)
print("5. ÖNCELİKLİ EKSİKLER (Model için değerli)")
print("="*70)

print("""
Öncelik sırası (model iyileştirme için):

1. 🔴 PMMA parçacıkları - En fazla eksik (200 parçacık!)
   - Model PMMA verisi az gördü, bu kategoride zayıf

2. 🟡 PA6 parçacıkları - Şuan 11 parçacık var
   - Daha fazla PA6 verisi model çeşitliliğini artırır

3. 🟢 ABS parçacıkları - Bazı eksikler var
   - Model ABS'yi iyi öğrenmiş, öncelik düşük
""")

# Sadece PMMA eksiklerini detaylı göster
print("\n--- PMMA Eksik Detay ---")
pmma_missing = {k: v for k, v in missing_by_material.items() if 'PMMA' in k}
for key, nums in sorted(pmma_missing.items()):
    print(f"\n{key}:")
    print(f"  Toplam eksik: {len(nums)}")
    print(f"  Numaralar: {sorted(nums)[:15]}..." if len(nums) > 15 else f"  Numaralar: {sorted(nums)}")
