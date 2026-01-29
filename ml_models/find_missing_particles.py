"""Excel ve Experiments.json arasındaki farkı bul"""
import pandas as pd
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ================================================================
# 1. EXCEL'DEKİ PARÇACIKLARI LİSTELE
# ================================================================
print("="*70)
print("1. EXCEL'DEKİ PARÇACIKLAR")
print("="*70)

excel_path = r'c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS (2).xlsx'
xl = pd.ExcelFile(excel_path)

excel_particles = {}  # {sheet: [codes]}

for sheet in xl.sheet_names:
    df = pd.read_excel(xl, sheet_name=sheet, header=None)

    # Parçacık kodlarını bul (genelde 2. kolon, 3. satırdan itibaren)
    codes = []
    for idx in range(2, len(df)):
        code = df.iloc[idx, 1]  # Shape/Code kolonu
        if pd.notna(code) and str(code).strip():
            codes.append(str(code).strip().upper())

    if codes:
        excel_particles[sheet] = codes
        print(f"\n{sheet}: {len(codes)} parçacık")
        print(f"  Kodlar: {codes[:10]}{'...' if len(codes) > 10 else ''}")

# ================================================================
# 2. EXPERIMENTS.JSON'DAKİ PARÇACIKLARI LİSTELE
# ================================================================
print("\n" + "="*70)
print("2. EXPERIMENTS.JSON'DAKİ PARÇACIKLAR")
print("="*70)

with open('processed_results/experiments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

exp_particles = {}  # {category: set(codes)}

for exp in data['experiments']:
    if exp['status'] != 'success':
        continue

    cat = exp['category']
    code = exp['code'].upper()

    if cat not in exp_particles:
        exp_particles[cat] = set()
    exp_particles[cat].add(code)

for cat, codes in sorted(exp_particles.items()):
    print(f"\n{cat}: {len(codes)} unique parçacık")
    print(f"  Kodlar: {sorted(codes)[:10]}{'...' if len(codes) > 10 else ''}")

# ================================================================
# 3. EŞLEŞTİRME VE FARK BULMA
# ================================================================
print("\n" + "="*70)
print("3. EŞLEŞTİRME ANALİZİ")
print("="*70)

# Manuel eşleştirme (sheet name -> category)
sheet_to_category = {
    'ABS CYLINDER': 'ABS C',
    'ABS HC': 'ABS HC',
    'PLA CYLINDER ': 'PLA C',
    'PLA CUBE ': 'PLA CUBE',
    'PLA HC ': 'PLA HC',
    'PS EC ': None,  # Experiments'da yok olabilir
    'RESIN (a=9 mm)': 'RESIN (a=9 r=4.5)',
    'RESIN (a=6 mm) ': 'RESIN (a=6 r=3)',
    'P6 BSP ': 'PA 6',
    'P6 HC ': 'PA 6',
    'P6 CYLINDER ': 'PA 6',
    'PMMA BSP': 'BSP',
    'PMMA Cylinder': 'C',
    'PMMA Wedge-Shaped': 'WSP',
    'PMMA Half Cylinder': 'HC',
}

missing_in_exp = []  # Excel'de var, experiments'da yok
extra_in_exp = []    # Experiments'da var, Excel'de yok

for sheet, excel_codes in excel_particles.items():
    cat = sheet_to_category.get(sheet.strip())

    if cat is None:
        print(f"\n⚠️  {sheet}: Experiments'da karşılığı bulunamadı")
        missing_in_exp.extend([(sheet, code) for code in excel_codes])
        continue

    if cat not in exp_particles:
        print(f"\n⚠️  {sheet} → {cat}: Experiments'da kategori yok")
        missing_in_exp.extend([(sheet, code) for code in excel_codes])
        continue

    exp_codes = exp_particles[cat]

    # Excel'de olup experiments'da olmayanlar
    excel_set = set(excel_codes)
    missing = excel_set - exp_codes

    if missing:
        print(f"\n{sheet} → {cat}:")
        print(f"  Excel'de var, Exp'da YOK: {sorted(missing)}")
        missing_in_exp.extend([(sheet, code) for code in missing])

# ================================================================
# 4. ÖZET
# ================================================================
print("\n" + "="*70)
print("4. ÖZET")
print("="*70)

total_excel = sum(len(codes) for codes in excel_particles.values())
total_exp = sum(len(codes) for codes in exp_particles.values())

print(f"""
Excel'deki toplam parçacık:      {total_excel}
Experiments'daki toplam:         {total_exp}
Excel'de olup Exp'da olmayan:    {len(missing_in_exp)}
""")

if missing_in_exp:
    print("--- EKSİK PARÇACIKLAR (Velocity ölçümü gerekli) ---")
    for sheet, code in missing_in_exp[:30]:
        print(f"  {sheet:<25} {code}")
    if len(missing_in_exp) > 30:
        print(f"  ... ve {len(missing_in_exp) - 30} tane daha")

# ================================================================
# 5. PS EC KONTROLÜ (Yeni kategori olabilir)
# ================================================================
print("\n" + "="*70)
print("5. PS EC KONTROLÜ")
print("="*70)

# PS EC experiments'da var mı?
ps_categories = [cat for cat in exp_particles.keys() if 'PS' in cat.upper() or 'EC' in cat.upper()]
print(f"Experiments'da PS/EC içeren kategoriler: {ps_categories}")

# RESIN EC var mı?
if 'RESIN (a=6 r=3)' in exp_particles:
    resin_ec = [c for c in exp_particles['RESIN (a=6 r=3)'] if 'D-' in c or 'EC' in c.upper()]
    print(f"RESIN (a=6 r=3) içinde EC/D kodları: {resin_ec}")
