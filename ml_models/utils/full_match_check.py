"""
TAM EŞLEŞME KONTROLÜ
Excel'deki tüm parçacıklar vs Training Data
"""
import pandas as pd
import json
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

excel_path = r'c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS-updated.xlsx'
json_path = r'C:\Users\mmert\PycharmProjects\ObjectTrackingProject\processed_results\experiments.json'
train_path = r'C:\Users\mmert\PycharmProjects\ObjectTrackingProject\data\training_data_v2.csv'

print("=" * 80)
print("TAM EŞLEŞME KONTROLÜ")
print("Excel'deki TÜM parçacıklar eşleşiyor mu?")
print("=" * 80)

# 1. Excel'deki tüm parçacıkları say
print("\n1. EXCEL'DEKİ TÜM PARÇACIKLAR")
print("-" * 80)

xl = pd.ExcelFile(excel_path)
excel_particles = {}

for sheet in xl.sheet_names:
    df = pd.read_excel(xl, sheet_name=sheet, header=None)

    if len(df) < 2:
        continue

    row1 = [str(c).strip() if pd.notna(c) else '' for c in df.iloc[1]]

    # Shape kolonunu bul
    shape_idx = None
    for i, c in enumerate(row1):
        if c == 'Shape':
            shape_idx = i
            break

    if shape_idx is None:
        continue

    # Type kolonunu bul
    type_idx = None
    for i, c in enumerate(row1):
        if c == 'Type':
            type_idx = i
            break

    particles = []
    for idx in range(2, len(df)):
        row = df.iloc[idx]
        shape = str(row.iloc[shape_idx]).strip() if pd.notna(row.iloc[shape_idx]) else ''
        type_val = str(row.iloc[type_idx]).strip() if type_idx is not None and pd.notna(row.iloc[type_idx]) else ''

        if shape and shape.lower() != 'nan':
            particles.append({'shape': shape, 'type': type_val})

    if particles:
        excel_particles[sheet] = particles
        print(f"  {sheet:25s}: {len(particles):3d} parçacık")

total_excel = sum(len(p) for p in excel_particles.values())
print(f"\n  TOPLAM: {total_excel} parçacık")

# 2. Experiments.json'daki başarılı deneyler
print("\n2. EXPERIMENTS.JSON - BAŞARILI DENEYLER")
print("-" * 80)

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

exp_by_category = {}
for exp in data['experiments']:
    if exp['status'] == 'success':
        cat = exp['category']
        code = exp['code']
        if cat not in exp_by_category:
            exp_by_category[cat] = set()
        exp_by_category[cat].add(code)

for cat, codes in sorted(exp_by_category.items()):
    print(f"  {cat:25s}: {len(codes):3d} benzersiz parçacık")

total_exp_unique = sum(len(codes) for codes in exp_by_category.values())
print(f"\n  TOPLAM benzersiz: {total_exp_unique} parçacık")

# 3. Training data'daki parçacıklar
print("\n3. TRAINING DATA - EŞLEŞTİRİLEN")
print("-" * 80)

train_df = pd.read_csv(train_path)
train_by_shape = train_df.groupby('shape_name').agg({
    'code': 'nunique',
    'velocity_ms': 'count'
}).rename(columns={'code': 'unique_particles', 'velocity_ms': 'total_records'})

for shape, row in train_by_shape.iterrows():
    print(f"  {shape:25s}: {row['unique_particles']:3d} parçacık, {row['total_records']:3d} kayıt")

print(f"\n  TOPLAM: {train_df['code'].nunique()} benzersiz parçacık, {len(train_df)} kayıt")

# 4. Excel sheet -> Category mapping kontrolü
print("\n4. SHEET-CATEGORY EŞLEŞTİRME KONTROLÜ")
print("-" * 80)

# Excel sheet'ler ve experiments kategorileri karşılaştırması
print("\n  Excel Sheet'leri:")
for sheet in sorted(excel_particles.keys()):
    print(f"    - {sheet}")

print("\n  Experiment Kategorileri:")
for cat in sorted(exp_by_category.keys()):
    print(f"    - {cat}")

# 5. Eşleşmeyen parçacıklar
print("\n5. DETAYLI EŞLEŞTİRME ANALİZİ")
print("-" * 80)

# Her sheet için hangi kodların eşleştiğini kontrol et
sheet_to_categories = {
    'ABS CYLINDER': ['ABS C'],
    'PLA CYLINDER ': ['PLA C'],
    'PMMA Cylinder': ['C'],
    'ABS HC': ['ABS HC'],
    'PLA HC ': ['PLA HC'],
    'PMMA Half Cylinder': ['HC'],
    'PLA CUBE ': ['PLA CUBE', 'ABS CUBE'],
    'PMMA BSP': ['BSP'],
    'PA6 BSP ': ['PA 6'],
    'PMMA Wedge-Shaped': ['WSP'],
    'RESIN SPHERE': ['RESIN (a=6 r=3)', 'RESIN (a=9 r=4.5)'],
    'RESIN CYLINDER': ['RESIN (a=6 r=3)', 'RESIN (a=9 r=4.5)'],
    'RESIN CUBE': ['RESIN (a=6 r=3)', 'RESIN (a=9 r=4.5)'],
    'RESIN EC': ['RESIN (a=6 r=3)', 'RESIN (a=9 r=4.5)', 'PS', 'ABS EC'],
}

for sheet, particles in excel_particles.items():
    categories = sheet_to_categories.get(sheet, [])

    # Bu sheet'e ait deneyler
    exp_codes = set()
    for cat in categories:
        if cat in exp_by_category:
            exp_codes.update(exp_by_category[cat])

    # Excel'deki parçacık kodları
    excel_codes = set()
    for p in particles:
        code = p['shape'].upper()
        # Normalize
        code = code.replace('BOX-SHAPED PRISM-', 'BSP-')
        code = code.replace('WEDGE-SHAPED-', 'WSP-')
        code = code.replace('CYLINDER-', 'C-')
        code = code.replace('HALF  CYLINDER-', 'HC-')
        code = code.replace('HALF CYLINDER-', 'HC-')
        excel_codes.add(code)

    # Karşılaştır
    matched = excel_codes & set(c.upper() for c in exp_codes)
    only_excel = excel_codes - set(c.upper() for c in exp_codes)
    only_exp = set(c.upper() for c in exp_codes) - excel_codes

    print(f"\n  {sheet}:")
    print(f"    Excel'de: {len(excel_codes)} parçacık")
    print(f"    Deneyde: {len(exp_codes)} benzersiz kod")
    print(f"    Eşleşen: {len(matched)}")

    if only_excel and len(only_excel) <= 10:
        print(f"    Sadece Excel'de: {sorted(only_excel)}")
    elif only_excel:
        print(f"    Sadece Excel'de: {len(only_excel)} adet")

    if only_exp and len(only_exp) <= 10:
        print(f"    Sadece deneyde: {sorted(only_exp)}")
    elif only_exp:
        print(f"    Sadece deneyde: {len(only_exp)} adet")

print("\n" + "=" * 80)
print("SONUÇ")
print("=" * 80)
print(f"""
Excel'de toplam: {total_excel} parçacık
Deneylerde benzersiz: {total_exp_unique} parçacık
Training'de eşleşen: {train_df['code'].nunique()} benzersiz parçacık

NOT: Bazı deneyler Excel'de olmayan parçacıklara referans veriyor olabilir.
     Bu durumda o deneyler eşleştirilemez ve training data'ya dahil edilmez.
""")
