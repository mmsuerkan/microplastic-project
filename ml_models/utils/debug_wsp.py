import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

xl = pd.ExcelFile('c:/Users/mmert/Downloads/ALL PARTICLES MEASUREMENTS-updated.xlsx')
df = pd.read_excel(xl, sheet_name='PMMA Wedge-Shaped', header=None)

print("=== PMMA Wedge-Shaped SHEET YAPISI ===")
print(f"Toplam satır: {len(df)}")
print(f"\nİlk 3 satır (headers):")
for i in range(3):
    row = [str(c)[:20] if pd.notna(c) else '' for c in df.iloc[i]]
    print(f"  Row {i}: {row[:15]}")

row1 = [str(c).strip() if pd.notna(c) else '' for c in df.iloc[1]]
print(f"\nKolon isimleri (row1): {row1}")

# Density kolonunu bul
density_idx = None
for i, c in enumerate(row1):
    if 'Density' in c and 'kg' in c:
        density_idx = i
        print(f"\nDensity kolonu: {i} -> {c}")
        break

# Tüm satırları kontrol et
valid_count = 0
for idx in range(2, len(df)):
    row = df.iloc[idx]
    shape = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ''
    density = row.iloc[density_idx] if density_idx and pd.notna(row.iloc[density_idx]) else None

    if shape and shape.lower() != 'nan':
        if density and density > 0:
            valid_count += 1
        else:
            print(f"  Geçersiz density: {shape} -> density={density}")

print(f"\nGeçerli parçacık sayısı: {valid_count}")
