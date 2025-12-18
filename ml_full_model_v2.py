"""
ML Model v2 - ALL PARTICLES MEASUREMENTS.xlsx'ten boyut ve yoğunluk
Tüm malzemeler dahil
"""
import pandas as pd
import numpy as np
import os
import glob
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import sys

sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = 'processed_results'

def parse_summary(filepath):
    """summary.csv dosyasından metrikleri oku"""
    metrics = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    metrics[parts[0].strip()] = parts[1].strip()
    except:
        pass
    return metrics

def load_particle_data():
    """ALL PARTICLES MEASUREMENTS.xlsx'ten boyut ve yoğunluk verilerini yükle"""
    xlsx = pd.ExcelFile('ALL PARTICLES MEASUREMENTS.xlsx')

    # Kategori eşleştirme (tüm key'ler strip edilmiş)
    sheet_to_category = {
        'ABS CYLINDER': 'ABS C',
        'ABS HC': 'ABS HC',
        'PLA CYLINDER': 'PLA C',
        'PLA CUBE': 'PLA CUBE',
        'PLA HC': 'PLA HC',
        'PS EC': 'PS',
        'RESIN (a=9 mm)': 'RESIN (a=9 r=4.5)',
        'RESIN (a=6 mm)': 'RESIN (a=6 r=3)',
        'P6 BSP': 'PA 6',
        'P6 HC': 'PA 6',
        'P6 CYLINDER': 'PA 6',
        'PMMA BSP': 'BSP',
        'PMMA Cylinder': 'C',
        'PMMA Wedge-Shaped': 'WSP',
        'PMMA Half Cylinder': 'HC'
    }

    # Kod öneki - local dosya yapısına göre
    shape_to_prefix = {
        'ABS CYLINDER': 'C',
        'ABS HC': 'HC',
        'PLA CYLINDER': 'C',
        'PLA CUBE': 'CUBE',
        'PLA HC': 'HC',
        'PS EC': 'EC',
        'RESIN (a=9 mm)': 'C',
        'RESIN (a=6 mm)': 'C',
        'P6 BSP': 'BSP',
        'P6 HC': 'HC',
        'P6 CYLINDER': 'C',
        'PMMA BSP': 'BSP',
        'PMMA Cylinder': 'C',
        'PMMA Wedge-Shaped': 'WSP',
        'PMMA Half Cylinder': 'HC'
    }

    # Kolon yapıları
    sheet_config = {
        'ABS CYLINDER': {'type': 'cylinder', 'avg_cols': [9, 10]},
        'ABS HC': {'type': 'prism3d', 'avg_cols': [12, 13, 14]},
        'PLA CYLINDER': {'type': 'cylinder', 'avg_cols': [9, 10]},
        'PLA CUBE': {'type': 'prism3d', 'avg_cols': [12, 13, 14]},
        'PLA HC': {'type': 'prism3d', 'avg_cols': [12, 13, 14]},
        'PS EC': {'type': 'prism3d', 'avg_cols': [12, 13, 14]},
        'RESIN (a=9 mm)': {'type': 'resin_cube', 'vol_col': 3, 'density_col': 5},
        'RESIN (a=6 mm)': {'type': 'resin_cube', 'vol_col': 3, 'density_col': 5},
        'P6 BSP': {'type': 'prism3d', 'avg_cols': [12, 13, 14]},
        'P6 HC': {'type': 'prism3d', 'avg_cols': [12, 13, 14]},
        'P6 CYLINDER': {'type': 'cylinder', 'avg_cols': [9, 10]},
        'PMMA BSP': {'type': 'prism3d', 'avg_cols': [12, 13, 14]},
        'PMMA Cylinder': {'type': 'cylinder', 'avg_cols': [9, 10]},
        'PMMA Wedge-Shaped': {'type': 'prism3d', 'avg_cols': [12, 13, 14]},
        'PMMA Half Cylinder': {'type': 'prism3d', 'avg_cols': [12, 13, 14]},
    }

    all_particles = []

    for sheet in xlsx.sheet_names:
        sheet_key = sheet.strip() if sheet.strip() in sheet_config else sheet
        if sheet_key not in sheet_config:
            continue

        config = sheet_config[sheet_key]
        category = sheet_to_category.get(sheet_key) or sheet_to_category.get(sheet)
        prefix = shape_to_prefix.get(sheet_key) or shape_to_prefix.get(sheet)

        df = pd.read_excel(xlsx, sheet, header=None)

        for i in range(2, len(df)):
            row = df.iloc[i]
            material = str(row[0]) if pd.notna(row[0]) else ''
            shape = str(row[1]) if pd.notna(row[1]) else ''

            # Header satırlarını atla (tam eşleşme)
            if not shape or shape == 'Shape' or material == 'Type':
                continue

            match = re.search(r'(\d+)', shape)
            if not match:
                continue
            code_num = match.group(1)
            code = f'{prefix}-{code_num}'

            # Boyutları al
            a, b, c, density = None, None, None, None

            if config['type'] == 'cylinder':
                # Diameter, Height → a=diameter, b=height, c=diameter
                cols = config['avg_cols']
                d = row[cols[0]] if pd.notna(row[cols[0]]) else None
                h = row[cols[1]] if pd.notna(row[cols[1]]) else None
                if d and h:
                    a, b, c = float(d), float(h), float(d)

            elif config['type'] == 'prism3d':
                # a, b, c doğrudan
                cols = config['avg_cols']
                a_val = row[cols[0]] if pd.notna(row[cols[0]]) else None
                b_val = row[cols[1]] if pd.notna(row[cols[1]]) else None
                c_val = row[cols[2]] if pd.notna(row[cols[2]]) else None
                if a_val and b_val and c_val:
                    a, b, c = float(a_val), float(b_val), float(c_val)

            elif config['type'] == 'resin_cube':
                # Volume'dan küp kenarı hesapla
                vol = row[config['vol_col']] if pd.notna(row[config['vol_col']]) else None
                if vol:
                    edge = float(vol) ** (1/3)
                    a, b, c = edge, edge, edge
                # Yoğunluk direkt kolonda
                density = row[config['density_col']] if pd.notna(row[config['density_col']]) else None

            # Yoğunluk bul (RESIN dışındakiler için)
            if density is None:
                for j in range(len(row)-1, -1, -1):
                    val = row[j]
                    if pd.notna(val) and isinstance(val, (int, float)) and 900 < val < 1500:
                        density = val
                        break

            if a and b and c and density:
                all_particles.append({
                    'category': category,
                    'code': code.upper(),
                    'material': material,
                    'a': a,
                    'b': b,
                    'c': c,
                    'density': float(density)
                })

    return pd.DataFrame(all_particles)

def main():
    # 1. ALL PARTICLES MEASUREMENTS'tan boyut ve yoğunluk yükle
    print("1. ALL PARTICLES MEASUREMENTS.xlsx'ten veriler yükleniyor...")
    df_particles = load_particle_data()
    print(f"   {len(df_particles)} parçacık verisi yüklendi")
    print(f"   Kategoriler: {df_particles['category'].nunique()}")
    print(f"   Malzemeler: {df_particles['material'].unique().tolist()}")

    # 2. Local summary dosyalarından hızları oku
    print("\n2. Local summary.csv dosyalarından hızlar okunuyor...")
    experiments = []
    pattern = os.path.join(RESULTS_DIR, 'success', '**', 'summary.csv')

    for summary_path in glob.glob(pattern, recursive=True):
        parts = os.path.normpath(summary_path).split(os.sep)
        # [0]processed_results/[1]success/[2]date/[3]view/[4]repeat/[5]category/[6]code/[7]summary.csv
        if len(parts) >= 8:
            metrics = parse_summary(summary_path)
            hiz_str = metrics.get('Hiz', '0')
            try:
                hiz_cms = float(hiz_str)
                hiz_ms = hiz_cms / 100
            except:
                continue

            experiments.append({
                'date': parts[2],
                'view': parts[3].upper(),
                'repeat': parts[4].upper(),
                'category': parts[5].upper(),
                'code': parts[6].upper(),
                'velocity_ms': hiz_ms
            })

    df_velocity = pd.DataFrame(experiments)
    print(f"   Local summary: {len(df_velocity)} deney")
    print(f"   Hız aralığı: {df_velocity['velocity_ms'].min():.4f} - {df_velocity['velocity_ms'].max():.4f} m/s")

    # 3. Birleştir (category + code üzerinden)
    print("\n3. Veriler birleştiriliyor (category + code)...")

    # Kategori normalizasyonu
    df_velocity['category_norm'] = df_velocity['category'].str.upper().str.strip()
    df_particles['category_norm'] = df_particles['category'].str.upper().str.strip()
    df_particles['code'] = df_particles['code'].str.upper().str.strip()

    merged = pd.merge(
        df_velocity,
        df_particles[['category_norm', 'code', 'material', 'a', 'b', 'c', 'density']],
        left_on=['category_norm', 'code'],
        right_on=['category_norm', 'code'],
        how='inner'
    )
    print(f"   Eşleşen: {len(merged)} satır")

    # Eşleşme oranı
    matched_cats = merged['category_norm'].unique()
    print(f"   Eşleşen kategoriler: {len(matched_cats)}")

    if len(merged) == 0:
        print("\n   HATA: Hiç eşleşme yok! Kategori/kod formatlarını kontrol et.")
        print("\n   Velocity kategorileri:", df_velocity['category_norm'].unique()[:10])
        print("   Particles kategorileri:", df_particles['category_norm'].unique()[:10])
        print("\n   Velocity kodları:", df_velocity['code'].unique()[:10])
        print("   Particles kodları:", df_particles['code'].unique()[:10])
        return

    # 4. Veri temizleme
    print("\n4. Veri temizleme...")
    merged = merged[(merged['velocity_ms'] > 0) & (merged['velocity_ms'] < 0.5)]
    merged = merged.dropna(subset=['a', 'b', 'c', 'density', 'velocity_ms'])
    print(f"   Temiz veri: {len(merged)} satır")

    # 5. Feature engineering
    print("\n5. Feature engineering...")

    # Türetilen özellikler
    merged['d_eq'] = (merged['a'] * merged['b'] * merged['c']) ** (1/3)
    merged['volume'] = merged['a'] * merged['b'] * merged['c']
    merged['surface_area'] = 2 * (merged['a']*merged['b'] + merged['b']*merged['c'] + merged['a']*merged['c'])
    merged['sphericity'] = (np.pi ** (1/3)) * ((6 * merged['volume']) ** (2/3)) / merged['surface_area']

    # Boyut sıralaması (L > I > S)
    dims = merged[['a', 'b', 'c']].values
    dims_sorted = np.sort(dims, axis=1)[:, ::-1]
    merged['L'] = dims_sorted[:, 0]
    merged['I'] = dims_sorted[:, 1]
    merged['S'] = dims_sorted[:, 2]
    merged['aspect_ratio'] = merged['L'] / merged['S']
    merged['CSF'] = merged['S'] / np.sqrt(merged['L'] * merged['I'])

    # Yoğunluk farkı (batma kuvveti)
    merged['delta_rho'] = merged['density'] - 1000

    # Encoding
    merged['cat_enc'] = LabelEncoder().fit_transform(merged['category_norm'].astype(str))
    merged['mat_enc'] = LabelEncoder().fit_transform(merged['material'].astype(str))
    merged['view_enc'] = LabelEncoder().fit_transform(merged['view'])

    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()

    # İstatistikler
    print(f"\n   Malzeme dağılımı:")
    for mat, count in merged['material'].value_counts().items():
        print(f"     {mat}: {count}")

    print(f"\n   Kategori dağılımı:")
    for cat, count in merged['category_norm'].value_counts().head(10).items():
        print(f"     {cat}: {count}")

    # 6. ML
    print("\n6. Model eğitimi...")
    features = ['a', 'b', 'c', 'd_eq', 'volume', 'surface_area', 'sphericity',
                'L', 'I', 'S', 'aspect_ratio', 'CSF', 'density', 'delta_rho',
                'cat_enc', 'mat_enc', 'view_enc']

    X = merged[features]
    y = merged['velocity_ms']

    print(f"   ML veri: {len(X)} satır, {len(features)} feature")
    print(f"   Hız: {y.mean():.4f} ± {y.std():.4f} m/s")
    print(f"   Yoğunluk: {merged['density'].mean():.0f} ± {merged['density'].std():.0f} kg/m³")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Random Forest
    print()
    print('=' * 65)
    print('RANDOM FOREST (Tüm Malzemeler - ALL PARTICLES verisi)')
    print('=' * 65)
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)
    print(f'R²:   {r2_score(y_test, y_pred):.4f}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.6f} m/s ({np.sqrt(mean_squared_error(y_test, y_pred))*100:.2f} cm/s)')
    print(f'MAE:  {mean_absolute_error(y_test, y_pred):.6f} m/s')
    cv = cross_val_score(rf, X_train_s, y_train, cv=5, scoring='r2')
    print(f'CV R²: {cv.mean():.4f} ± {cv.std():.4f}')

    # XGBoost
    print()
    print('=' * 65)
    print('XGBOOST (Tüm Malzemeler - ALL PARTICLES verisi)')
    print('=' * 65)
    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    xgb.fit(X_train_s, y_train)
    y_pred_xgb = xgb.predict(X_test_s)
    print(f'R²:   {r2_score(y_test, y_pred_xgb):.4f}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.6f} m/s ({np.sqrt(mean_squared_error(y_test, y_pred_xgb))*100:.2f} cm/s)')
    print(f'MAE:  {mean_absolute_error(y_test, y_pred_xgb):.6f} m/s')
    cv_xgb = cross_val_score(xgb, X_train_s, y_train, cv=5, scoring='r2')
    print(f'CV R²: {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}')

    # Feature Importance
    print()
    print('Feature Importance (RF):')
    imp = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    for _, row in imp.head(12).iterrows():
        print(f"  {row['feature']:15s}: {row['importance']:.3f}")

    # Veriyi kaydet
    merged.to_csv('ml_data_full_v2.csv', index=False, encoding='utf-8')
    print(f"\nVeri kaydedildi: ml_data_full_v2.csv")

if __name__ == "__main__":
    main()
