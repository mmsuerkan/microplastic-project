"""
Tüm RESIN Sheet'lerini Kontrol Et
"""
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

excel_path = r'c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS-updated.xlsx'

print("=" * 70)
print("TÜM RESIN SHEET'LERI")
print("=" * 70)

xl = pd.ExcelFile(excel_path)

resin_sheets = [s for s in xl.sheet_names if 'RESIN' in s]
print(f"\nRESIN sheet'leri: {resin_sheets}")

for sheet in resin_sheets:
    print(f"\n{'='*70}")
    print(f"SHEET: {sheet}")
    print("="*70)

    df = pd.read_excel(xl, sheet_name=sheet, header=None)

    # Kolon isimleri
    row1 = [str(c).strip() if pd.notna(c) else '' for c in df.iloc[1]]
    print(f"Kolonlar: {row1[:10]}")

    # Tüm parçacıkları listele
    print("\nParçacıklar:")
    for idx in range(2, min(12, len(df))):
        row = df.iloc[idx]
        type_val = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ''
        shape = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ''

        if shape and shape != 'nan':
            # Density kolonunu bul
            density_col = None
            for i, c in enumerate(row1):
                if 'Density' in c and 'kg' in c:
                    density_col = i
                    break

            density = row.iloc[density_col] if density_col and pd.notna(row.iloc[density_col]) else 'N/A'
            print(f"  {type_val:15s} | {shape:15s} | density={density}")
