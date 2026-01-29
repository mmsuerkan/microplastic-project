"""Eksik PMMA parçacıklarını Excel'e çıkar - Lab için liste"""
import pandas as pd
import json
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ================================================================
# 1. MEVCUT VELOCITY ÖLÇÜMLERİNİ BUL
# ================================================================
with open('processed_results/experiments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# PMMA kategorileri ve kolon bilgileri
pmma_config = {
    'BSP': {
        'sheet': 'PMMA BSP',
        'shape': 'Box Shape Prism',
        'a_col': 'a (mm).3',
        'b_col': 'b (mm).3',
        'c_col': 'c (mm).3',
    },
    'C': {
        'sheet': 'PMMA Cylinder',
        'shape': 'Cylinder',
        'a_col': 'Diameter (mm)',      # a = diameter
        'b_col': 'Diameter (mm)',      # b = diameter (cylinder)
        'c_col': 'Height(mm)',         # c = height
    },
    'HC': {
        'sheet': 'PMMA Half Cylinder',
        'shape': 'Half Cylinder',
        'a_col': 'Diameter (mm).1',    # Average diameter
        'b_col': 'Diameter (mm).1',    # b = diameter (half cylinder)
        'c_col': 'Height (mm).1',      # Average height
    },
    'WSP': {
        'sheet': 'PMMA Wedge-Shaped',
        'shape': 'Wedge Shape Prism',
        'a_col': 'a (mm).3',
        'b_col': 'b (mm).3',
        'c_col': 'c (mm).3',
    },
}

# Mevcut velocity ölçümleri (numara bazlı)
existing = {}
for cat in pmma_config.keys():
    existing[cat] = set()
    for exp in data['experiments']:
        if exp['status'] == 'success' and exp['category'] == cat:
            match = re.search(r'(\d+)', exp['code'])
            if match:
                existing[cat].add(int(match.group(1)))

print("Mevcut velocity ölçümleri:")
for cat, nums in existing.items():
    print(f"  {cat}: {len(nums)} parçacık")

# ================================================================
# 2. EXCEL'DEN BOYUT VE DENSITY BİLGİLERİNİ AL
# ================================================================
excel_path = r'c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS (2).xlsx'
xl = pd.ExcelFile(excel_path)

missing_list = []

for cat, config in pmma_config.items():
    sheet_name = config['sheet']
    print(f"\n{sheet_name} işleniyor...")

    # Header satırı 1 (0-indexed)
    df = pd.read_excel(xl, sheet_name=sheet_name, header=1)
    cols = df.columns.tolist()

    print(f"  Kolon sayısı: {len(cols)}")

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

    print(f"  a_idx={a_idx}, b_idx={b_idx}, c_idx={c_idx}, density_idx={density_idx}")

    for idx, row in df.iterrows():
        code = row.iloc[1]  # İkinci kolon: Shape/Code
        if pd.notna(code) and str(code).strip():
            code_str = str(code).strip()

            # Numarayı çıkar
            match = re.search(r'(\d+)', code_str)
            if match:
                num = int(match.group(1))

                # Bu numara eksik mi?
                if num not in existing[cat]:
                    # Boyut ve density bilgilerini al
                    a = row.iloc[a_idx] if a_idx is not None and pd.notna(row.iloc[a_idx]) else None
                    b = row.iloc[b_idx] if b_idx is not None and pd.notna(row.iloc[b_idx]) else None
                    c = row.iloc[c_idx] if c_idx is not None and pd.notna(row.iloc[c_idx]) else None
                    density = row.iloc[density_idx] if density_idx is not None and pd.notna(row.iloc[density_idx]) else None

                    missing_list.append({
                        'Kategori': cat,
                        'Shape': config['shape'],
                        'Numara': num,
                        'Excel_Kod': code_str,
                        'Experiments_Kod': f"{cat}-{num}",
                        'a_mm': round(float(a), 2) if a is not None else None,
                        'b_mm': round(float(b), 2) if b is not None else None,
                        'c_mm': round(float(c), 2) if c is not None else None,
                        'Density_kg_m3': round(float(density), 1) if density is not None else None,
                        'Velocity_Status': 'EKSIK - Ölçüm gerekli'
                    })

# DataFrame oluştur
df_missing = pd.DataFrame(missing_list)

# Kategori ve numaraya göre sırala
df_missing = df_missing.sort_values(['Kategori', 'Numara'])

print(f"\n{'='*60}")
print(f"TOPLAM EKSİK PMMA: {len(df_missing)} parçacık")
print(f"{'='*60}")

# Kategori bazlı özet
print("\nKategori bazlı:")
for cat in pmma_config.keys():
    count = len(df_missing[df_missing['Kategori'] == cat])
    print(f"  {cat}: {count} eksik")

# Excel'e kaydet
output_path = 'data/eksik_pmma_velocity_listesi.xlsx'
df_missing.to_excel(output_path, index=False, sheet_name='Eksik PMMA')

print(f"\n✓ Kaydedildi: {output_path}")

# Ayrıca CSV olarak da kaydet
csv_path = 'data/eksik_pmma_velocity_listesi.csv'
df_missing.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✓ Kaydedildi: {csv_path}")

# Ekrana örnek göster
print(f"\n--- Örnek Veriler (Her kategoriden 3'er) ---")
for cat in pmma_config.keys():
    cat_data = df_missing[df_missing['Kategori'] == cat].head(3)
    if len(cat_data) > 0:
        print(f"\n{cat} ({pmma_config[cat]['shape']}):")
        print(cat_data[['Numara', 'a_mm', 'b_mm', 'c_mm', 'Density_kg_m3']].to_string(index=False))

# Density istatistikleri
print(f"\n--- Boyut ve Density İstatistikleri ---")
stats = df_missing.groupby('Kategori').agg({
    'a_mm': ['mean', 'min', 'max'],
    'b_mm': ['mean', 'min', 'max'],
    'c_mm': ['mean', 'min', 'max'],
    'Density_kg_m3': ['mean', 'min', 'max']
}).round(2)
print(stats.to_string())
