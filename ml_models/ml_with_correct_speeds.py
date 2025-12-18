"""
ML Model - Doğru hızlarla (local summary.csv'lerden)
"""
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import sys

sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = 'processed_results'

def parse_summary(filepath):
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

def main():
    # 1. Local summary dosyalarından hızları oku
    print("1. Local summary.csv dosyalarından hızlar okunuyor...")
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
    print(f"   Hız aralığı: {df_velocity['velocity_ms'].min():.4f} - {df_velocity['velocity_ms'].max():.4f} m/s")
    print(f"   Ortalama hız: {df_velocity['velocity_ms'].mean():.4f} m/s")

    # 2. Excel'den boyutları oku
    print("\n2. Excel'den boyutlar okunuyor...")
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

    # 3. Birleştir
    print("\n3. Veriler birleştiriliyor...")
    df_velocity['code'] = df_velocity['code'].str.upper().str.strip()
    df_dims['code'] = df_dims['code'].str.upper().str.strip()
    df_velocity['view'] = df_velocity['view'].str.upper().str.strip()
    df_dims['view'] = df_dims['view'].str.upper().str.strip()
    df_velocity['repeat'] = df_velocity['repeat'].str.upper().str.strip()
    df_dims['repeat'] = df_dims['repeat'].str.upper().str.strip()

    merged = pd.merge(
        df_velocity,
        df_dims[['code', 'view', 'repeat', 'cat_excel', 'a', 'b', 'c', 'material']],
        on=['code', 'view', 'repeat'],
        how='inner'
    )
    print(f"   Eşleşen: {len(merged)} satır")

    # 4. Feature engineering
    print("\n4. Feature engineering...")
    merged['a'] = pd.to_numeric(merged['a'], errors='coerce')
    merged['b'] = pd.to_numeric(merged['b'], errors='coerce')
    merged['c'] = pd.to_numeric(merged['c'], errors='coerce')
    merged = merged.dropna(subset=['a', 'b', 'c', 'velocity_ms'])
    merged = merged[(merged['velocity_ms'] > 0) & (merged['velocity_ms'] < 0.5)]

    print(f"   Temiz veri: {len(merged)} satır")

    # Yoğunluk
    density_map = {'ABS': 1135, 'PLA': 1200, 'PS': 1047, 'Plexiglass': 1200, 'RESIN': 1150}
    def get_density(m):
        m_upper = str(m).upper()
        for k, v in density_map.items():
            if k in m_upper:
                return v
        return 1150
    merged['density'] = merged['material'].apply(get_density)

    # Türetilen özellikler
    merged['d_eq'] = (merged['a'] * merged['b'] * merged['c']) ** (1/3)
    merged['volume'] = merged['a'] * merged['b'] * merged['c']
    merged['surface_area'] = 2 * (merged['a']*merged['b'] + merged['b']*merged['c'] + merged['a']*merged['c'])
    merged['sphericity'] = (np.pi ** (1/3)) * ((6 * merged['volume']) ** (2/3)) / merged['surface_area']

    dims = merged[['a', 'b', 'c']].values
    dims_sorted = np.sort(dims, axis=1)[:, ::-1]
    merged['L'] = dims_sorted[:, 0]
    merged['I'] = dims_sorted[:, 1]
    merged['S'] = dims_sorted[:, 2]
    merged['aspect_ratio'] = merged['L'] / merged['S']
    merged['CSF'] = merged['S'] / np.sqrt(merged['L'] * merged['I'])

    # Encoding
    merged['cat_enc'] = LabelEncoder().fit_transform(merged['category'].astype(str))
    merged['mat_enc'] = LabelEncoder().fit_transform(merged['material'].astype(str))
    merged['view_enc'] = LabelEncoder().fit_transform(merged['view'])

    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()

    # 5. ML
    print("\n5. Model eğitimi...")
    features = ['a', 'b', 'c', 'd_eq', 'volume', 'surface_area', 'sphericity',
                'L', 'I', 'S', 'aspect_ratio', 'CSF', 'density',
                'cat_enc', 'mat_enc', 'view_enc']

    X = merged[features]
    y = merged['velocity_ms']

    print(f"   ML veri: {len(X)} satır, {len(features)} feature")
    print(f"   Hız: {y.mean():.4f} ± {y.std():.4f} m/s")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Random Forest
    print()
    print('=' * 55)
    print('RANDOM FOREST (Güncel Local Hızlar)')
    print('=' * 55)
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
    print('=' * 55)
    print('XGBOOST (Güncel Local Hızlar)')
    print('=' * 55)
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
    for _, row in imp.head(8).iterrows():
        print(f"  {row['feature']:15s}: {row['importance']:.3f}")

    # Veriyi kaydet
    merged.to_csv('ml_data_correct_speeds.csv', index=False, encoding='utf-8')
    print(f"\nVeri kaydedildi: ml_data_correct_speeds.csv")

if __name__ == "__main__":
    main()
