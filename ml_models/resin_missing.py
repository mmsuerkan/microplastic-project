"""RESIN eksik parçacıkları bul"""
import pandas as pd
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Excel'deki RESIN parçacıkları
excel_path = r'c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS-updated.xlsx'
xl = pd.ExcelFile(excel_path)

excel_resin = {
    'SPHERE': {'r=3': set(), 'r=4.5': set()},
    'CUBE': {'a=6': set(), 'a=9': set()},
    'CYLINDER': {'r=3': set(), 'r=4.5': set()},
    'EC': {'r=3': set(), 'r=4.5': set()},
}

for sheet in ['RESIN SPHERE', 'RESIN CYLINDER', 'RESIN CUBE', 'RESIN EC']:
    df = pd.read_excel(xl, sheet_name=sheet, header=None)
    shape_type = sheet.replace('RESIN ', '')

    for idx in range(2, len(df)):
        row = df.iloc[idx]
        type_val = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ''
        code = str(row.iloc[1]).strip().upper() if pd.notna(row.iloc[1]) else ''

        if not code:
            continue

        # Normalize code
        code = code.replace('CUBE-', 'CUBE-').replace('SP-', 'SP-').replace('C-', 'C-').replace('D-', 'D-')

        if 'r=3' in type_val or 'a=6' in type_val:
            key = 'r=3' if 'r=3' in type_val else 'a=6'
        else:
            key = 'r=4.5' if 'r=4.5' in type_val else 'a=9'

        excel_resin[shape_type][key].add(code)

print("=" * 60)
print("EXCEL'DEKİ RESIN PARÇACIKLARI")
print("=" * 60)
for shape, sizes in excel_resin.items():
    print(f"\nRESIN {shape}:")
    for size, codes in sizes.items():
        print(f"  {size}: {sorted(codes)}")

# Experiments'daki RESIN kodları
with open(r'C:\Users\mmert\PycharmProjects\ObjectTrackingProject\processed_results\experiments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

exp_resin = {
    'a=6 r=3': {'SP': set(), 'CUBE': set(), 'C': set(), 'D': set()},
    'a=9 r=4.5': {'SP': set(), 'CUBE': set(), 'C': set(), 'D': set()},
}

for exp in data['experiments']:
    if 'RESIN' not in exp['category'] or exp['status'] != 'success':
        continue

    cat = exp['category']
    code = exp['code'].upper()

    size = 'a=6 r=3' if 'a=6' in cat else 'a=9 r=4.5'

    if code.startswith('SP-'):
        exp_resin[size]['SP'].add(code)
    elif code.startswith('CUBE-'):
        exp_resin[size]['CUBE'].add(code)
    elif code.startswith('C-'):
        exp_resin[size]['C'].add(code)
    elif code.startswith('D-'):
        exp_resin[size]['D'].add(code)

print("\n" + "=" * 60)
print("EXPERIMENTS'DAKİ RESIN KODLARI")
print("=" * 60)
for size, shapes in exp_resin.items():
    print(f"\nRESIN ({size}):")
    for shape, codes in shapes.items():
        print(f"  {shape}: {sorted(codes)}")

# Eksikleri bul
print("\n" + "=" * 60)
print("EKSİK PARÇACIKLAR (Experiments'da var, Excel'de yok)")
print("=" * 60)

missing = []

# a=6 r=3 için
for shape_prefix, exp_codes in exp_resin['a=6 r=3'].items():
    if shape_prefix == 'SP':
        excel_codes = excel_resin['SPHERE']['r=3']
    elif shape_prefix == 'CUBE':
        excel_codes = excel_resin['CUBE']['a=6']
    elif shape_prefix == 'C':
        excel_codes = excel_resin['CYLINDER']['r=3']
    elif shape_prefix == 'D':
        excel_codes = excel_resin['EC']['r=3']

    diff = exp_codes - excel_codes
    if diff:
        missing.append(f"RESIN (a=6 r=3) {shape_prefix}: {sorted(diff)}")

# a=9 r=4.5 için
for shape_prefix, exp_codes in exp_resin['a=9 r=4.5'].items():
    if shape_prefix == 'SP':
        excel_codes = excel_resin['SPHERE']['r=4.5']
    elif shape_prefix == 'CUBE':
        excel_codes = excel_resin['CUBE']['a=9']
    elif shape_prefix == 'C':
        excel_codes = excel_resin['CYLINDER']['r=4.5']
    elif shape_prefix == 'D':
        excel_codes = excel_resin['EC']['r=4.5']

    diff = exp_codes - excel_codes
    if diff:
        missing.append(f"RESIN (a=9 r=4.5) {shape_prefix}: {sorted(diff)}")

if missing:
    for m in missing:
        print(f"\n  {m}")
else:
    print("\n  Eksik parçacık yok!")
