"""
Training Data Hazırlama - VERSIYON V4
Excel (parçacık ölçümleri) + JSON (velocity) → Training CSV
Yeni Excel dosyası: ALL PARTICLES MEASUREMENTS (1).xlsx
"""
import pandas as pd
import numpy as np
import json
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

# Şekil kodları
SHAPE_NAMES = {
    0: 'Cylinder',
    1: 'Half Cylinder',
    2: 'Cube',
    3: 'Wedge Shape Prism',
    4: 'Box Shape Prism',
    5: 'Sphere',
    6: 'Elliptic Cylinder'
}

# Experiments category -> Shape
CATEGORY_TO_SHAPE = {
    'ABS C': 0, 'PLA C': 0, 'C': 0,
    'ABS HC': 1, 'PLA HC': 1, 'HC': 1,
    'PLA CUBE': 2, 'ABS CUBE': 2,
    'WSP': 3,
    'BSP': 4, 'PA 6': 4,
    'PS': 6, 'ABS EC': 6,
}

# Category -> Excel sheet
CATEGORY_TO_SHEET = {
    'ABS C': 'ABS CYLINDER',
    'PLA C': 'PLA CYLINDER ',
    'C': 'PMMA Cylinder',
    'ABS HC': 'ABS HC',
    'PLA HC': 'PLA HC ',
    'HC': 'PMMA Half Cylinder',
    'PLA CUBE': 'PLA CUBE ',
    'ABS CUBE': 'PLA CUBE ',
    'WSP': 'PMMA Wedge-Shaped',
    'BSP': 'PMMA BSP',
}

# RESIN kod prefixi -> sheet ve shape
RESIN_CODE_MAP = {
    'C': ('RESIN CYLINDER', 0),    # Cylinder
    'SP': ('RESIN SPHERE', 5),     # Sphere
    'CUBE': ('RESIN CUBE', 2),     # Cube
    'D': ('RESIN EC', 6),          # Elliptic Cylinder
}

# PA6 kod prefixi -> sheet ve shape
PA6_CODE_MAP = {
    'C': ('PA6 CYLINDER ', 0),     # Cylinder
    'HC': ('PA6 HC ', 1),          # Half Cylinder
    'BSP': ('PA6 BSP ', 4),        # Box Shape Prism
}

# RESIN category -> Excel type
RESIN_TYPE_MAP = {
    'RESIN (a=6 r=3)': {
        'RESIN SPHERE': 'RESIN (r=3)',
        'RESIN CYLINDER': 'RESIN (r=3)',
        'RESIN EC': 'RESIN (r=3)',
        'RESIN CUBE': 'RESIN (a=6)',
    },
    'RESIN (a=9 r=4.5)': {
        'RESIN SPHERE': 'RESIN (r=4.5)',
        'RESIN CYLINDER': 'RESIN (r=4.5)',
        'RESIN EC': 'RESIN (r=4.5)',
        'RESIN CUBE': 'RESIN (a=9)',
    },
}


def parse_velocity(hiz_str):
    if not hiz_str:
        return None
    match = re.search(r'([\d.]+)\s*cm/s', str(hiz_str))
    return float(match.group(1)) / 100 if match else None


def normalize_code(code):
    code = str(code).strip().upper()
    code = code.replace('BOX-SHAPED PRISM-', 'BSP-')
    code = code.replace('WEDGE-SHAPED-', 'WSP-')
    code = code.replace('ELLIPTICAL HALF CYLINDER-', 'HC-')
    code = code.replace('CYLINDER-', 'C-')
    code = code.replace('HALF  CYLINDER-', 'HC-')
    code = code.replace('HALF CYLINDER-', 'HC-')
    # Excel'deki "HALF  C-X" formatını "HC-X" formatına çevir
    code = code.replace('HALF  C-', 'HC-')
    code = code.replace('HALF C-', 'HC-')
    return code


def load_excel_data(excel_path):
    """Excel'den tüm parçacık verilerini yükle - DÜZELTILMIŞ V3"""
    xl = pd.ExcelFile(excel_path)
    all_data = {}

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
        particles = {}

        row0 = [str(c).strip() if pd.notna(c) else '' for c in df.iloc[0]]
        row1 = [str(c).strip() if pd.notna(c) else '' for c in df.iloc[1]]

        # Type kolonu (RESIN sheet'leri için)
        type_idx = None
        for i, c in enumerate(row1):
            if c == 'Type':
                type_idx = i
                break

        # Shape kolonu
        shape_idx = None
        for i, c in enumerate(row1):
            if c == 'Shape':
                shape_idx = i
                break

        if shape_idx is None:
            continue

        # Density kolonu (kg/m^3)
        density_idx = None
        for i, c in enumerate(row1):
            if 'Density' in c and 'kg' in c:
                density_idx = i
                break

        # c (mm) kolonunu bul - bazen en sonda
        c_explicit_idx = None
        for i, c in enumerate(row1):
            if c == 'c (mm)':
                c_explicit_idx = i

        # Sheet tipine göre a, b, c kolonlarını belirle
        # CYLINDER sheet'leri: Average Dimension altında a, b var, c ayrı kolonda
        # 3D shape sheet'leri: Average Dimension altında a, b, c ardışık

        avg_start = None
        for i, c in enumerate(row0):
            if 'Average' in c and 'Dimension' in c:
                avg_start = i
                break

        std_start = None
        for i, c in enumerate(row0):
            if 'Standard' in c and 'deviation' in c:
                std_start = i
                break

        # Cylinder tipi sheet mi kontrol et (Height kolonu var mı?)
        is_cylinder_sheet = any('Height' in str(c) for c in row1)

        if is_cylinder_sheet and avg_start is not None:
            # Cylinder sheet: a=çap, b=yükseklik, c=0 veya ayrı kolonda
            a_idx = avg_start
            b_idx = avg_start + 1
            # c için ayrı kolon ara (std_start'tan sonra değil!)
            if c_explicit_idx and c_explicit_idx > std_start if std_start else True:
                c_idx = c_explicit_idx
            else:
                c_idx = None  # Cylinder için c=0
        elif avg_start is not None:
            # 3D shape sheet: a, b, c ardışık
            a_idx = avg_start
            b_idx = avg_start + 1
            c_idx = avg_start + 2
        else:
            # RESIN gibi basit sheet'ler
            a_idx = b_idx = c_idx = None
            found_a = found_b = found_c = False
            for i, c in enumerate(row1):
                if c == 'a (mm)' and not found_a:
                    a_idx = i
                    found_a = True
                elif c == 'b (mm)' and not found_b:
                    b_idx = i
                    found_b = True
                elif c == 'c (mm)' and not found_c:
                    c_idx = i
                    found_c = True

        if a_idx is None or density_idx is None:
            continue

        # Verileri oku
        for idx in range(2, len(df)):
            try:
                row = df.iloc[idx]
                shape_val = str(row.iloc[shape_idx]).strip()
                code = normalize_code(shape_val)

                if not code or code == 'NAN' or code == '':
                    continue

                # Type değeri (RESIN için)
                type_val = str(row.iloc[type_idx]).strip() if type_idx is not None and pd.notna(row.iloc[type_idx]) else ''

                a = float(row.iloc[a_idx]) if pd.notna(row.iloc[a_idx]) else 0
                b = float(row.iloc[b_idx]) if b_idx and pd.notna(row.iloc[b_idx]) else 0

                if c_idx is not None:
                    c_val = row.iloc[c_idx]
                    c = float(c_val) if pd.notna(c_val) else 0
                else:
                    c = 0  # Cylinder için c=0

                density = float(row.iloc[density_idx]) if pd.notna(row.iloc[density_idx]) else None

                if density is None or density <= 0:
                    continue
                if a <= 0:
                    continue

                # RESIN sheet'leri için type+code kombinasyonu kullan
                if 'RESIN' in sheet_name and type_val:
                    key = f"{type_val}|{code}"
                else:
                    key = code

                particles[key] = {'a': a, 'b': b, 'c': c, 'density': density, 'type': type_val}
            except Exception as e:
                continue

        all_data[sheet_name] = particles

    return all_data


def load_experiments(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for exp in data['experiments']:
        if exp['status'] != 'success':
            continue
        velocity = parse_velocity(exp.get('metrics', {}).get('Hiz', ''))
        if velocity and velocity > 0:
            results.append({
                'category': exp['category'],
                'code': exp['code'],
                'date': exp['date'],
                'view': exp['view'],
                'repeat': exp['repeat'],
                'velocity_ms': velocity
            })
    return results


def main():
    print("=" * 70)
    print("TRAINING DATA HAZIRLAMA - DUZELTILMIS")
    print("=" * 70)

    excel_path = r'c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS-updated (1).xlsx'
    wsp_excel_path = r'c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS-updated (3).xlsx'
    json_path = r'C:\Users\mmert\PycharmProjects\ObjectTrackingProject\processed_results\experiments.json'
    output_path = r'C:\Users\mmert\PycharmProjects\ObjectTrackingProject\data\training_data_v2.csv'

    # 1. Excel yükle
    print("\n1. Excel yukleniyor...")
    excel_data = load_excel_data(excel_path)

    # WSP density'lerini ayrı dosyadan al
    print("   WSP density'leri yukleniyor...")
    wsp_data = load_excel_data(wsp_excel_path)
    if 'PMMA Wedge-Shaped' in wsp_data:
        excel_data['PMMA Wedge-Shaped'] = wsp_data['PMMA Wedge-Shaped']
        print(f"   WSP guncellendi: {len(wsp_data['PMMA Wedge-Shaped'])} parcacik")
    total = sum(len(v) for v in excel_data.values())
    print(f"   Toplam parcacik: {total}")
    for sheet, particles in excel_data.items():
        if particles:
            print(f"     {sheet}: {len(particles)}")
            # İlk parçacığı göster
            first_code = list(particles.keys())[0]
            first_p = particles[first_code]
            print(f"       Ornek: {first_code} -> a={first_p['a']:.2f}, b={first_p['b']:.2f}, c={first_p['c']:.2f}, d={first_p['density']:.0f}")

    # 2. Experiments yükle
    print("\n2. Experiments yukleniyor...")
    experiments = load_experiments(json_path)
    print(f"   Basarili deney: {len(experiments)}")

    # 3. Eşleştir
    print("\n3. Eslestiriliyor...")
    matched = []
    unmatched_cats = set()
    unmatched_codes = []
    shape_counts = {i: 0 for i in range(7)}

    for exp in experiments:
        category = exp['category']
        code = normalize_code(exp['code'])

        # PA 6 kategorisi için özel işlem
        if category == 'PA 6':
            # Kod prefixini belirle
            code_prefix = None
            for prefix in ['BSP', 'HC', 'C']:
                if code.startswith(prefix + '-'):
                    code_prefix = prefix
                    break

            if not code_prefix:
                unmatched_codes.append(f"{category}:{code}")
                continue

            sheet_name, shape_enc = PA6_CODE_MAP.get(code_prefix, (None, None))
            if not sheet_name or sheet_name not in excel_data:
                unmatched_codes.append(f"{category}:{code} -> {sheet_name}")
                continue

            particles = excel_data[sheet_name]
            particle = particles.get(code)

            # Bulunamadıysa code numarasıyla ara
            if not particle:
                code_num = re.search(r'(\d+)', code)
                if code_num:
                    for pkey, pdata in particles.items():
                        if code_num.group(1) in pkey:
                            particle = pdata
                            break

            if not particle:
                unmatched_codes.append(f"{category}:{code} -> {sheet_name}")
                continue

        # RESIN kategorileri için özel işlem
        elif 'RESIN' in category:
            # Kod prefixini belirle
            code_prefix = None
            for prefix in ['SP', 'CUBE', 'C', 'D']:
                if code.startswith(prefix + '-'):
                    code_prefix = prefix
                    break

            if not code_prefix:
                unmatched_codes.append(f"{category}:{code}")
                continue

            sheet_name, shape_enc = RESIN_CODE_MAP.get(code_prefix, (None, None))
            if not sheet_name:
                unmatched_codes.append(f"{category}:{code} (prefix)")
                continue

            if sheet_name not in excel_data:
                unmatched_cats.add(f"{category} -> {sheet_name}")
                continue

            # RESIN type belirleme (sheet'e göre farklı format)
            type_map = RESIN_TYPE_MAP.get(category, {})
            excel_type = type_map.get(sheet_name)
            if not excel_type:
                unmatched_cats.add(f"{category} -> {sheet_name} (type)")
                continue

            particles = excel_data[sheet_name]

            # type|code kombinasyonu ile ara
            search_key = f"{excel_type}|{code}"
            particle = particles.get(search_key)

            # Bulunamadıysa code numarasıyla ara
            if not particle:
                code_num = re.search(r'(\d+)', code)
                if code_num:
                    for pkey, pdata in particles.items():
                        if excel_type in pkey and code_num.group(1) in pkey:
                            particle = pdata
                            break

            if not particle:
                unmatched_codes.append(f"{category}:{code} -> {sheet_name}")
                continue

        else:
            # Normal kategoriler
            shape_enc = CATEGORY_TO_SHAPE.get(category)
            if shape_enc is None:
                unmatched_cats.add(category)
                continue

            sheet_name = CATEGORY_TO_SHEET.get(category)
            if not sheet_name or sheet_name not in excel_data:
                unmatched_cats.add(f"{category} (sheet)")
                continue

            particles = excel_data[sheet_name]
            particle = particles.get(code)

            # Code eşleşmediyse sayısal ara
            if not particle:
                num = re.search(r'(\d+)', code)
                if num:
                    for pcode, pdata in particles.items():
                        if code.split('-')[0] in pcode and num.group(1) in pcode:
                            particle = pdata
                            break

            if not particle:
                unmatched_codes.append(f"{category}:{code}")
                continue

        # Şekle göre a, b, c dönüşümü
        a, b, c = particle['a'], particle['b'], particle['c']

        if shape_enc == 5:  # Sphere
            a = max(a, b, c) if max(a, b, c) > 0 else a
            b, c = 0.0, 0.0
        elif shape_enc in [0, 6]:  # Cylinder, Elliptic Cylinder
            c = 0.0

        shape_counts[shape_enc] += 1
        matched.append({
            'category': category,
            'code': exp['code'],
            'shape_enc': shape_enc,
            'shape_name': SHAPE_NAMES[shape_enc],
            'a': a,
            'b': b,
            'c': c,
            'density': particle['density'],
            'velocity_ms': exp['velocity_ms'],
            'velocity_cms': exp['velocity_ms'] * 100,
            'date': exp['date'],
            'view': exp['view'],
            'repeat': exp['repeat']
        })

    print(f"   Eslesen: {len(matched)}")
    if unmatched_cats:
        print(f"   Eslesmeyen kategoriler: {unmatched_cats}")
    if unmatched_codes:
        print(f"   Eslesmeyen kod sayisi: {len(unmatched_codes)}")
        # İlk 10 örneği göster
        for uc in unmatched_codes[:10]:
            print(f"     - {uc}")

    # 4. Feature'ları hesapla ve kaydet
    df = pd.DataFrame(matched)

    # Volume hesaplama
    def calc_volume(row):
        shape = row['shape_enc']
        a, b, c = row['a'], row['b'], row['c']
        if shape == 0:  # Cylinder: a=çap, b=yükseklik
            return np.pi * (a/2)**2 * b if b > 0 else np.pi * (a/2)**2 * a
        elif shape == 1:  # Half Cylinder
            return 0.5 * np.pi * (a/2)**2 * b if b > 0 else 0.5 * np.pi * (a/2)**2 * a
        elif shape == 2:  # Cube
            return a * b * c if b > 0 and c > 0 else a**3
        elif shape == 3:  # Wedge
            return 0.5 * a * b * c if b > 0 and c > 0 else 0.5 * a**3
        elif shape == 4:  # Box
            return a * b * c if b > 0 and c > 0 else a**3
        elif shape == 5:  # Sphere
            return (4/3) * np.pi * (a/2)**3
        elif shape == 6:  # Elliptic Cylinder
            return np.pi * (a/2) * (b/2) * c if b > 0 and c > 0 else np.pi * (a/2)**2 * a
        return a * b * c if b > 0 and c > 0 else a**3

    def calc_surface_area(row):
        shape = row['shape_enc']
        a, b, c = row['a'], row['b'], row['c']
        if shape == 5:  # Sphere
            return 4 * np.pi * (a/2)**2
        elif shape == 0:  # Cylinder: a=çap, b=yükseklik
            r = a/2
            h = b if b > 0 else a
            return 2 * np.pi * r * (r + h)
        elif shape == 1:  # Half Cylinder
            r = a/2
            h = b if b > 0 else a
            return np.pi * r * (r + h) + 2 * r * h
        return 2 * (a*b + b*c + a*c) if b > 0 and c > 0 else 6 * a**2

    df['volume'] = df.apply(calc_volume, axis=1)
    df['surface_area'] = df.apply(calc_surface_area, axis=1)
    df['aspect_ratio'] = df['a'] / np.where(df['b'] == 0, df['a'], df['b'])
    df['vol_surf_ratio'] = df['volume'] / np.where(df['surface_area'] == 0, 1, df['surface_area'])

    df.to_csv(output_path, index=False)
    print(f"\n4. Kaydedildi: {output_path}")

    # Özet
    print("\n" + "=" * 70)
    print("OZET")
    print("=" * 70)
    print(f"Toplam: {len(matched)} satir\n")
    print("Sekil dagilimi:")
    for i, count in shape_counts.items():
        if count > 0:
            print(f"  {i}: {SHAPE_NAMES[i]} - {count}")

    print(f"\nVelocity: {df['velocity_cms'].min():.2f} - {df['velocity_cms'].max():.2f} cm/s")
    print(f"\nOrnek (her sekilden):")
    for shape in df['shape_name'].unique():
        sample = df[df['shape_name'] == shape].iloc[0]
        print(f"  {shape}: a={sample['a']:.2f}, b={sample['b']:.2f}, c={sample['c']:.2f}, d={sample['density']:.0f}, v={sample['velocity_cms']:.2f}")


if __name__ == "__main__":
    main()
