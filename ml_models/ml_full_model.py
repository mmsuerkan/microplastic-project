"""
ML Model - Tüm malzemeler ve ölçülmüş yoğunluklarla
ALL PARTICLES MEASUREMENTS.xlsx'ten gerçek yoğunluklar kullanılıyor
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

def load_density_data():
    """ALL PARTICLES MEASUREMENTS.xlsx'ten yoğunluk verilerini yükle"""
    xlsx = pd.ExcelFile('ALL PARTICLES MEASUREMENTS.xlsx')

    sheet_to_category = {
        'ABS CYLINDER': 'ABS C',
        'ABS HC': 'ABS HC',
        'PLA CYLINDER ': 'PLA C',
        'PLA CUBE ': 'PLA CUBE',
        'PLA HC ': 'PLA HC',
        'PS EC ': 'PS',
        'RESIN (a=9 mm)': 'RESIN (a=9 r=4.5)',
        'RESIN (a=6 mm) ': 'RESIN (a=6 r=3)',
        'P6 BSP ': 'PA 6',
        'P6 HC ': 'PA 6',
        'P6 CYLINDER ': 'PA 6',
        'PMMA BSP': 'BSP',
        'PMMA Cylinder': 'C',
        'PMMA Wedge-Shaped': 'WSP',
        'PMMA Half Cylinder': 'HC'
    }

    shape_to_prefix = {
        'ABS CYLINDER': 'C',
        'ABS HC': 'HC',
        'PLA CYLINDER ': 'C',
        'PLA CUBE ': 'CUBE',
        'PLA HC ': 'HC',
        'PS EC ': 'EC',
        'RESIN (a=9 mm)': 'C',
        'RESIN (a=6 mm) ': 'C',
        'P6 BSP ': 'BSP',
        'P6 HC ': 'HC',
        'P6 CYLINDER ': 'C',
        'PMMA BSP': 'BSP',
        'PMMA Cylinder': 'C',
        'PMMA Wedge-Shaped': 'WSP',
        'PMMA Half Cylinder': 'HC'
    }

    density_data = {}

    for sheet in xlsx.sheet_names:
        df = pd.read_excel(xlsx, sheet, header=None)
        category = sheet_to_category.get(sheet.strip(), None)
        prefix = shape_to_prefix.get(sheet.strip(), None)

        if not category:
            continue

        for i in range(2, len(df)):
            row = df.iloc[i]
            shape = str(row[1]) if pd.notna(row[1]) else ''

            match = re.search(r'(\d+)', shape)
            if not match:
                continue
            code_num = match.group(1)

            # Yoğunluk bul (kg/m³ değeri 900-1500 aralığında)
            density = None
            for j in range(len(row)-1, -1, -1):
                val = row[j]
                if pd.notna(val) and isinstance(val, (int, float)) and 900 < val < 1500:
                    density = val
                    break

            if density:
                code = f'{prefix}-{code_num}'
                key = (category, code)
                density_data[key] = density

    return density_data

def main():
    # 1. Yoğunluk verilerini yükle
    print("1. Yoğunluk verileri yükleniyor (ALL PARTICLES MEASUREMENTS.xlsx)...")
    density_data = load_density_data()
    print(f"   {len(density_data)} parçacık için yoğunluk verisi")

    # 2. Local summary dosyalarından hızları oku
    print("\n2. Local summary.csv dosyalarından hızlar okunuyor...")
    experiments = []
    pattern = os.path.join(RESULTS_DIR, 'success', '**', 'summary.csv')

    for summary_path in glob.glob(pattern, recursive=True):
        rel_path = os.path.relpath(summary_path, RESULTS_DIR)
        parts = rel_path.replace('\\', '/').split('/')

        if len(parts) >= 6:
            metrics = parse_summary(summary_path)
            hiz_str = metrics.get('Hiz', '0')
            try:
                hiz_cms = float(hiz_str)
                hiz_ms = hiz_cms / 100
            except:
                continue

            experiments.append({
                'date': parts[1],
                'view': parts[2],
                'repeat': parts[3],
                'category': parts[4],
                'code': parts[5],
                'velocity_ms': hiz_ms
            })

    df_velocity = pd.DataFrame(experiments)
    print(f"   Local summary: {len(df_velocity)} deney")

    # 3. Excel'den boyutları oku
    print("\n3. Excel'den boyutlar okunuyor...")
    xlsx = pd.ExcelFile('Video_Boyut_Eslestirme_FINAL.xlsx')
    df_mak = pd.read_excel(xlsx, 'MAK')
    df_ang = pd.read_excel(xlsx, 'ANG')
    df_mak['view'] = 'MAK'
    df_ang['view'] = 'ANG'
    df_dims = pd.concat([df_mak, df_ang], ignore_index=True)
    df_dims = df_dims.rename(columns={
        'Kod': 'code',
        'Deney': 'repeat',
        'Boyut 1': 'a',
        'Boyut 2': 'b',
        'Boyut 3': 'c',
        'Plastik Tipi': 'material',
        'Kategori': 'cat_excel'
    })
    print(f"   Excel boyut: {len(df_dims)} satır")

    # 4. Birleştir
    print("\n4. Veriler birleştiriliyor...")
    df_velocity['code'] = df_velocity['code'].str.upper().str.strip()
    df_dims['code'] = df_dims['code'].str.upper().str.strip()
    df_velocity['view'] = df_velocity['view'].str.upper().str.strip()
    df_dims['view'] = df_dims['view'].str.upper().str.strip()
    df_velocity['repeat'] = df_velocity['repeat'].str.upper().str.strip()
    df_dims['repeat'] = df_dims['repeat'].str.upper().str.strip()
    df_velocity['category'] = df_velocity['category'].str.upper().str.strip()

    merged = pd.merge(
        df_velocity,
        df_dims[['code', 'view', 'repeat', 'cat_excel', 'a', 'b', 'c', 'material']],
        on=['code', 'view', 'repeat'],
        how='inner'
    )
    print(f"   Eşleşen: {len(merged)} satır")

    # 5. Yoğunluk ekle
    print("\n5. Ölçülmüş yoğunluklar ekleniyor...")

    def get_measured_density(row):
        cat = str(row['category']).upper().strip()
        code = str(row['code']).upper().strip()
        key = (cat, code)

        # Direkt eşleşme
        if key in density_data:
            return density_data[key]

        # Kategori normalizations dene
        cat_mappings = {
            'RESIN (A=9 R=4.5)': 'RESIN (a=9 r=4.5)',
            'RESIN (A=6 R=3)': 'RESIN (a=6 r=3)',
        }

        normalized_cat = cat_mappings.get(cat, cat)
        key = (normalized_cat, code)
        if key in density_data:
            return density_data[key]

        return None

    merged['measured_density'] = merged.apply(get_measured_density, axis=1)

    # Ölçülmüş yoğunluk olmayanlar için tahmin kullan
    density_map = {'ABS': 1135, 'PLA': 1200, 'PS': 1047, 'PLEXIGLASS': 1180, 'RESIN': 1150, 'PMMA': 1180}

    def get_fallback_density(row):
        if pd.notna(row['measured_density']):
            return row['measured_density']

        m_upper = str(row['material']).upper()
        for k, v in density_map.items():
            if k in m_upper:
                return v
        return 1150  # default

    merged['density'] = merged.apply(get_fallback_density, axis=1)

    measured_count = merged['measured_density'].notna().sum()
    print(f"   Ölçülmüş yoğunluk: {measured_count}/{len(merged)} ({100*measured_count/len(merged):.1f}%)")

    # 6. Feature engineering
    print("\n6. Feature engineering...")
    merged['a'] = pd.to_numeric(merged['a'], errors='coerce')
    merged['b'] = pd.to_numeric(merged['b'], errors='coerce')
    merged['c'] = pd.to_numeric(merged['c'], errors='coerce')
    merged = merged.dropna(subset=['a', 'b', 'c', 'velocity_ms'])
    merged = merged[(merged['velocity_ms'] > 0) & (merged['velocity_ms'] < 0.5)]

    print(f"   Temiz veri: {len(merged)} satır")

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

    # Yoğunluk farkı (batma kuvveti ile ilgili)
    merged['delta_rho'] = merged['density'] - 1000  # su yoğunluğu

    # Encoding
    merged['cat_enc'] = LabelEncoder().fit_transform(merged['category'].astype(str))
    merged['mat_enc'] = LabelEncoder().fit_transform(merged['material'].astype(str))
    merged['view_enc'] = LabelEncoder().fit_transform(merged['view'])

    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()

    # Malzeme dağılımı
    print("\n   Malzeme dağılımı:")
    for mat, count in merged['material'].value_counts().items():
        print(f"     {mat}: {count}")

    # 7. ML
    print("\n7. Model eğitimi...")
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
    print('=' * 60)
    print('RANDOM FOREST (Tüm Malzemeler + Ölçülmüş Yoğunluklar)')
    print('=' * 60)
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
    print('=' * 60)
    print('XGBOOST (Tüm Malzemeler + Ölçülmüş Yoğunluklar)')
    print('=' * 60)
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
    for _, row in imp.head(10).iterrows():
        print(f"  {row['feature']:15s}: {row['importance']:.3f}")

    # Veriyi kaydet
    merged.to_csv('ml_data_full.csv', index=False, encoding='utf-8')
    print(f"\nVeri kaydedildi: ml_data_full.csv")

if __name__ == "__main__":
    main()
