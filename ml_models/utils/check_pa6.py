import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')
xl = pd.ExcelFile('c:/Users/mmert/Downloads/ALL PARTICLES MEASUREMENTS-updated.xlsx')
df = pd.read_excel(xl, sheet_name='PA6 BSP ', header=None)
print('=== PA6 BSP DENSITY DEĞERLERİ ===')
row1 = [str(c).strip() if pd.notna(c) else '' for c in df.iloc[1]]
density_idx = None
for i, c in enumerate(row1):
    if 'Density' in c and 'kg' in c:
        density_idx = i
        break
print(f'Density kolonu: {density_idx}')
for idx in range(2, len(df)):
    row = df.iloc[idx]
    shape = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ''
    density = row.iloc[density_idx] if density_idx and pd.notna(row.iloc[density_idx]) else None
    if shape:
        print(f'  {shape}: {density:.2f} kg/m3' if density else f'  {shape}: YOK')
