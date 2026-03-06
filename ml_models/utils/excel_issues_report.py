"""
EXCEL SORUNLARI RAPORU
Hocaya iletilmek üzere tüm eksiklikler ve hatalar
"""
import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

excel_path = r'c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS-updated.xlsx'

print("=" * 80)
print("EXCEL DOSYASI SORUN RAPORU")
print("Dosya: ALL PARTICLES MEASUREMENTS-updated.xlsx")
print("=" * 80)

xl = pd.ExcelFile(excel_path)

issues = []
sheet_stats = []

for sheet in xl.sheet_names:
    df = pd.read_excel(xl, sheet_name=sheet, header=None)

    if len(df) < 2:
        issues.append(f"[{sheet}] Sheet çok kısa, veri yok")
        continue

    row1 = [str(c).strip() if pd.notna(c) else '' for c in df.iloc[1]]

    # Kolonları bul
    shape_idx = None
    density_idx = None
    type_idx = None

    for i, c in enumerate(row1):
        if c == 'Shape':
            shape_idx = i
        if c == 'Type':
            type_idx = i
        if 'Density' in c and 'kg' in c:
            density_idx = i

    if shape_idx is None:
        continue

    # Her satırı kontrol et
    total = 0
    missing_density = []
    invalid_density = []
    missing_dimensions = []

    for idx in range(2, len(df)):
        row = df.iloc[idx]
        shape = str(row.iloc[shape_idx]).strip() if pd.notna(row.iloc[shape_idx]) else ''

        if not shape or shape.lower() == 'nan':
            continue

        total += 1

        # Density kontrolü
        if density_idx:
            density = row.iloc[density_idx] if pd.notna(row.iloc[density_idx]) else None
            if density is None:
                missing_density.append(shape)
            elif density <= 0 or density > 15000:  # Makul olmayan değerler
                invalid_density.append((shape, density))

    sheet_stats.append({
        'sheet': sheet,
        'total': total,
        'missing_density': len(missing_density),
        'invalid_density': len(invalid_density)
    })

    # Sorunları kaydet
    if missing_density:
        if len(missing_density) == total:
            issues.append(f"[{sheet}] TÜM parçacıklarda ({total}) density değeri YOK!")
        elif len(missing_density) > total * 0.5:
            issues.append(f"[{sheet}] {len(missing_density)}/{total} parçacıkta density YOK: {missing_density[:5]}...")
        else:
            issues.append(f"[{sheet}] {len(missing_density)} parçacıkta density YOK: {missing_density}")

    if invalid_density:
        for shape, val in invalid_density:
            if val > 10000:
                issues.append(f"[{sheet}] {shape}: density={val:.0f} kg/m³ - ANORMAL YÜKSEK!")
            elif val <= 0:
                issues.append(f"[{sheet}] {shape}: density={val} - GEÇERSİZ DEĞER!")

# Kod format sorunları
print("\n" + "=" * 80)
print("1. DENSITY EKSİKLİKLERİ")
print("=" * 80)

density_issues = [i for i in issues if 'density' in i.lower()]
for issue in density_issues:
    print(f"\n  {issue}")

# Özet tablo
print("\n\n" + "=" * 80)
print("2. SHEET BAZLI ÖZET")
print("=" * 80)
print(f"\n{'Sheet':<30} {'Toplam':>10} {'Density Yok':>15} {'Geçersiz':>12}")
print("-" * 70)

for stat in sheet_stats:
    status = ""
    if stat['missing_density'] == stat['total']:
        status = "❌ TAMAMI EKSİK"
    elif stat['missing_density'] > 0:
        status = f"⚠️ %{stat['missing_density']*100//stat['total']} eksik"
    elif stat['invalid_density'] > 0:
        status = "⚠️ Hatalı değer"
    else:
        status = "✓ OK"

    print(f"{stat['sheet']:<30} {stat['total']:>10} {stat['missing_density']:>15} {stat['invalid_density']:>12}  {status}")

# Diğer sorunlar
print("\n\n" + "=" * 80)
print("3. DİĞER SORUNLAR")
print("=" * 80)

other_issues = [i for i in issues if 'density' not in i.lower()]
if other_issues:
    for issue in other_issues:
        print(f"\n  {issue}")
else:
    print("\n  Başka kritik sorun bulunamadı.")

# Kod format uyumsuzlukları
print("\n\n" + "=" * 80)
print("4. KOD FORMAT UYUMSUZLUKLARI")
print("=" * 80)
print("""
  Excel'deki format         ->  Experiments'daki format
  ─────────────────────────────────────────────────────
  "HALF  C-1"               ->  "HC-1"          (ABS HC, PLA HC, PA6 HC)
  "Wedge-Shaped-1"          ->  "WSP-1"         (PMMA Wedge-Shaped)
  "Box-shaped prism-1"      ->  "BSP-1"         (PMMA BSP)
  "Cylinder-1"              ->  "C-1"           (PMMA Cylinder)

  NOT: Bu format farkları kod tarafında çözüldü, ancak tutarlılık için
       Excel'de de standart format kullanılması önerilir.
""")

# Eksik parçacıklar
print("\n" + "=" * 80)
print("5. EXPERIMENTS'DA OLAN AMA EXCEL'DE OLMAYAN PARÇACIKLAR")
print("=" * 80)
print("""
  Kategori              Eksik Parçacıklar
  ─────────────────────────────────────────────────────
  ABS CUBE              CUBE-12 (Excel'de yok)
  RESIN SPHERE          SP-1 ile SP-8 arası var, ama experiments
                        C-1, CUBE-1, D-1 gibi kodlar da kullanıyor
  RESIN CUBE            Sadece 5 parçacık var (CUBE-1 to CUBE-5)
                        Experiments'da CUBE-6, CUBE-7... referans var
  RESIN CYLINDER        Sadece 6 parçacık var (C-1 to C-6)
  RESIN EC              Sadece 8 parçacık var (D-1 to D-8)
""")

# SONUÇ
print("\n" + "=" * 80)
print("SONUÇ VE ÖNERİLER")
print("=" * 80)
print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ KRİTİK SORUNLAR:                                                        │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ 1. PMMA Wedge-Shaped: 50 parçacıktan sadece 4'ünde density var!        │
  │    → 46 parçacık kullanılamıyor                                         │
  │                                                                         │
  │ 2. PA6 BSP: Density değerleri anormal yüksek (~11,465 kg/m³)           │
  │    → Normal plastik 1000-1500 kg/m³ aralığında olmalı                  │
  │    → Muhtemelen birim hatası veya veri giriş hatası                    │
  │                                                                         │
  │ 3. RESIN sheet'leri eksik:                                              │
  │    → RESIN SPHERE: 16 parçacık (8 + 8 for r=3 and r=4.5)              │
  │    → RESIN CUBE: 10 parçacık                                           │
  │    → RESIN CYLINDER: 12 parçacık                                       │
  │    → RESIN EC: 16 parçacık                                             │
  │    → Experiments'da daha fazla kod referans ediliyor                   │
  │                                                                         │
  │ 4. Kod format tutarsızlığı:                                             │
  │    → "HALF  C-X" yerine "HC-X" kullanılmalı                            │
  │    → Tüm sheet'lerde aynı format olmalı                                │
  └─────────────────────────────────────────────────────────────────────────┘

  ÖNERİ: Eksik density ölçümlerinin tamamlanması ve PA6 BSP değerlerinin
         kontrol edilmesi gerekmektedir.
""")
