"""
Veri Doğrulama - Training data vs Original Excel
"""
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

excel_path = r'c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS-updated.xlsx'

print("=" * 70)
print("VERİ DOĞRULAMA - Original Excel Kontrolü")
print("=" * 70)

xl = pd.ExcelFile(excel_path)
print(f"\nSheet'ler: {xl.sheet_names}")

# Birkaç kritik sheet'i kontrol edelim
sheets_to_check = ['ABS CYLINDER', 'PLA CUBE ', 'RESIN SPHERE', 'PMMA BSP']

for sheet in sheets_to_check:
    if sheet not in xl.sheet_names:
        print(f"\n{sheet} bulunamadı!")
        continue

    print(f"\n{'='*70}")
    print(f"SHEET: {sheet}")
    print("="*70)

    df = pd.read_excel(xl, sheet_name=sheet, header=None)

    # İlk 3 satırı göster (header)
    print("\nHeader satırları:")
    for i in range(min(3, len(df))):
        row = [str(c)[:15] if pd.notna(c) else '' for c in df.iloc[i]]
        print(f"  Row {i}: {row[:15]}")

    # Kolon isimlerini bul (row 1)
    row1 = [str(c).strip() if pd.notna(c) else '' for c in df.iloc[1]]

    # Shape, a, b, c, density kolonlarını bul
    shape_idx = None
    density_idx = None

    for i, c in enumerate(row1):
        if c == 'Shape':
            shape_idx = i
        if 'Density' in c and 'kg' in c:
            density_idx = i

    print(f"\nShape kolonu: {shape_idx}")
    print(f"Density kolonu: {density_idx}")

    # İlk 5 veri satırını göster
    print("\nİlk 5 parçacık:")
    for idx in range(2, min(7, len(df))):
        row = df.iloc[idx]
        shape = str(row.iloc[shape_idx]).strip() if shape_idx and pd.notna(row.iloc[shape_idx]) else 'N/A'
        density = row.iloc[density_idx] if density_idx and pd.notna(row.iloc[density_idx]) else 'N/A'

        # Row'un tamamını göster
        values = [f"{row.iloc[i]:.2f}" if isinstance(row.iloc[i], float) else str(row.iloc[i])[:10]
                  for i in range(min(15, len(row))) if pd.notna(row.iloc[i])]
        print(f"  {shape}: {values[:10]}")

print("\n" + "="*70)
print("Training Data'dan örnekler:")
print("="*70)

# Training data'yı yükle
train_df = pd.read_csv(r'C:\Users\mmert\PycharmProjects\ObjectTrackingProject\data\training_data_v2.csv')

# Her shape'den bir örnek
for shape in train_df['shape_name'].unique():
    sample = train_df[train_df['shape_name'] == shape].iloc[0]
    print(f"\n{shape} ({sample['category']}, {sample['code']}):")
    print(f"  a={sample['a']:.2f}, b={sample['b']:.2f}, c={sample['c']:.2f}, density={sample['density']:.0f}")
