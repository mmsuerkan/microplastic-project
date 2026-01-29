"""
Debug: RESIN sheet key'lerini kontrol et
"""
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

excel_path = r'c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS-updated.xlsx'

def normalize_code(code):
    code = str(code).strip().upper()
    code = code.replace('BOX-SHAPED PRISM-', 'BSP-')
    code = code.replace('WEDGE-SHAPED-', 'WSP-')
    code = code.replace('ELLIPTICAL HALF CYLINDER-', 'HC-')
    code = code.replace('CYLINDER-', 'C-')
    code = code.replace('HALF  CYLINDER-', 'HC-')
    code = code.replace('HALF CYLINDER-', 'HC-')
    return code

xl = pd.ExcelFile(excel_path)

for sheet in ['RESIN SPHERE', 'RESIN CYLINDER', 'RESIN CUBE', 'RESIN EC']:
    print(f"\n{'='*60}")
    print(f"SHEET: {sheet}")
    print("="*60)

    df = pd.read_excel(xl, sheet_name=sheet, header=None)
    row1 = [str(c).strip() if pd.notna(c) else '' for c in df.iloc[1]]

    print(f"Kolonlar: {row1[:10]}")

    # Type ve Shape indekslerini bul
    type_idx = None
    shape_idx = None
    for i, c in enumerate(row1):
        if c == 'Type':
            type_idx = i
        if c == 'Shape':
            shape_idx = i

    print(f"Type idx: {type_idx}, Shape idx: {shape_idx}")

    # Tüm parçacıkları listele
    print("\nKey'ler:")
    for idx in range(2, min(6, len(df))):  # Sadece ilk 4 satır
        row = df.iloc[idx]
        raw_type = row.iloc[type_idx] if type_idx else None
        type_val = str(raw_type).strip() if type_idx and pd.notna(raw_type) else ''
        shape = str(row.iloc[shape_idx]).strip() if shape_idx and pd.notna(row.iloc[shape_idx]) else ''
        code = normalize_code(shape)

        print(f"  raw_type={repr(raw_type)}, type_val={repr(type_val)}, code={code}")

        if code and code != 'NAN':
            key = f"{type_val}|{code}" if type_val else code
            print(f"    -> key={key}")
