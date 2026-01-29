"""
RESIN SPHERE Sheet Detaylı Kontrol
"""
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

excel_path = r'c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS-updated.xlsx'

print("=" * 70)
print("RESIN SPHERE - Detaylı Kontrol")
print("=" * 70)

df = pd.read_excel(excel_path, sheet_name='RESIN SPHERE', header=None)

# Tüm satırları göster
print("\nTüm parçacıklar:")
print("-" * 70)

row1 = [str(c).strip() if pd.notna(c) else '' for c in df.iloc[1]]
print(f"Kolonlar: {row1}")

for idx in range(2, len(df)):
    row = df.iloc[idx]
    type_val = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ''
    shape = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ''
    a = row.iloc[2] if pd.notna(row.iloc[2]) else 0
    b = row.iloc[3] if pd.notna(row.iloc[3]) else 0
    c = row.iloc[4] if pd.notna(row.iloc[4]) else 0
    density = row.iloc[8] if pd.notna(row.iloc[8]) else 0

    if shape and shape != 'nan':
        print(f"  {type_val:15s} | {shape:8s} | a={a:5.2f}, b={b}, c={c}, density={density:.0f}")

# Experiments.json'dan RESIN deneylerini kontrol et
print("\n" + "=" * 70)
print("RESIN Deneyleri (experiments.json)")
print("=" * 70)

import json
with open(r'C:\Users\mmert\PycharmProjects\ObjectTrackingProject\processed_results\experiments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for exp in data['experiments']:
    if 'RESIN' in exp.get('category', '') and exp['status'] == 'success':
        hiz = exp.get('metrics', {}).get('Hiz', '')
        print(f"  {exp['category']:20s} | {exp['code']:10s} | {hiz}")
