"""
Mikroplastik Çökelme Hızı Tahmin Modeli
TÜBİTAK 1001 - İP4

Girdi: Parçacık özellikleri (boyut, şekil, yoğunluk)
Çıktı: Çökelme hızı (m/s)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
import sys

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# XGBoost opsiyonel
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost yüklü değil, RandomForest kullanılacak")


def load_particle_measurements(filepath='ALL PARTICLES MEASUREMENTS.xlsx'):
    """Parçacık ölçüm verilerini yükle ve birleştir"""

    xlsx = pd.ExcelFile(filepath)
    all_data = []

    for sheet in xlsx.sheet_names:
        df = pd.read_excel(xlsx, sheet, header=1)

        # İlk satır header olabilir, temizle
        df = df.dropna(how='all')

        # Sheet isminden malzeme ve şekil bilgisi çıkar
        sheet_lower = sheet.lower()

        # Malzeme belirleme
        if 'abs' in sheet_lower:
            material = 'ABS'
        elif 'pla' in sheet_lower:
            material = 'PLA'
        elif 'ps' in sheet_lower:
            material = 'PS'
        elif 'pmma' in sheet_lower or 'plexiglass' in sheet_lower:
            material = 'PMMA'
        elif 'resin' in sheet_lower:
            material = 'RESIN'
        elif 'p6' in sheet_lower or 'pa 6' in sheet_lower:
            material = 'PA6'
        else:
            material = 'UNKNOWN'

        # Şekil belirleme
        if 'bsp' in sheet_lower or 'box' in sheet_lower:
            shape = 'BSP'
        elif 'hc' in sheet_lower or 'half' in sheet_lower:
            shape = 'HC'
        elif 'cube' in sheet_lower:
            shape = 'CUBE'
        elif 'ec' in sheet_lower or 'ellip' in sheet_lower:
            shape = 'EC'
        elif 'wedge' in sheet_lower:
            shape = 'WSP'
        elif 'cylinder' in sheet_lower or 'cyl' in sheet_lower:
            shape = 'C'
        else:
            shape = 'OTHER'

        # Sütun isimlerini normalize et
        df.columns = df.columns.astype(str)

        for idx, row in df.iterrows():
            # Shape/kod bilgisini al
            shape_col = None
            for col in df.columns:
                if 'shape' in col.lower() or 'unnamed: 1' in col.lower():
                    shape_col = col
                    break

            particle_code = str(row.get(shape_col, f'{shape}-{idx}')) if shape_col else f'{shape}-{idx}'

            # Boyutları al (farklı formatlar için)
            a, b, c = None, None, None

            # a, b, c formatı
            for col in df.columns:
                col_lower = col.lower()
                if 'average' in col_lower or 'unnamed: 12' in col_lower or 'unnamed: 13' in col_lower:
                    if 'a (mm)' in col_lower or 'a' == col_lower.strip():
                        a = row.get(col)
                    elif 'b (mm)' in col_lower or 'b' == col_lower.strip():
                        b = row.get(col)
                    elif 'c (mm)' in col_lower or 'c' == col_lower.strip():
                        c = row.get(col)

            # Diameter/Height formatı (silindir)
            diameter, height = None, None
            for col in df.columns:
                col_lower = col.lower()
                if 'diameter' in col_lower and 'average' in col_lower:
                    diameter = row.get(col)
                elif 'height' in col_lower and ('average' in col_lower or 'unnamed: 10' in col_lower):
                    height = row.get(col)

            # Boyutları standartlaştır
            if diameter is not None and height is not None:
                a = diameter
                b = diameter
                c = height

            # Hacim
            volume = None
            for col in df.columns:
                if 'volume-average' in col.lower() or 'volume' in col.lower():
                    vol_val = row.get(col)
                    if pd.notna(vol_val) and isinstance(vol_val, (int, float)):
                        volume = vol_val
                        break

            # Yoğunluk (kg/m³)
            density = None
            for col in df.columns:
                if 'density (kg/m' in col.lower() or 'kg/m^3' in col.lower() or 'kg/m³' in col.lower():
                    dens_val = row.get(col)
                    if pd.notna(dens_val) and isinstance(dens_val, (int, float)):
                        density = dens_val
                        break

            # Yüzey alanı
            surface_area = None
            for col in df.columns:
                if 'surface area' in col.lower():
                    sa_val = row.get(col)
                    if pd.notna(sa_val) and isinstance(sa_val, (int, float)):
                        surface_area = sa_val
                        break

            # Ağırlık
            weight = None
            for col in df.columns:
                if 'weight' in col.lower():
                    w_val = row.get(col)
                    if pd.notna(w_val) and isinstance(w_val, (int, float)):
                        weight = w_val
                        break

            # Veriyi ekle
            if any([a, b, c, volume, density]):
                all_data.append({
                    'particle_code': particle_code,
                    'material': material,
                    'shape': shape,
                    'a_mm': a,
                    'b_mm': b,
                    'c_mm': c,
                    'volume_mm3': volume,
                    'surface_area_mm2': surface_area,
                    'density_kg_m3': density,
                    'weight_g': weight,
                    'source_sheet': sheet
                })

    return pd.DataFrame(all_data)


def load_velocity_data(filepath='Video_Boyut_Eslestirme_FINAL.xlsx'):
    """Hız verilerini yükle"""

    xlsx = pd.ExcelFile(filepath)
    df_mak = pd.read_excel(xlsx, 'MAK')
    df_ang = pd.read_excel(xlsx, 'ANG')

    df_mak['view'] = 'MAK'
    df_ang['view'] = 'ANG'

    df = pd.concat([df_mak, df_ang], ignore_index=True)

    # Sütun isimlerini düzenle
    df = df.rename(columns={
        'Kod': 'particle_code',
        'Kategori': 'category',
        'Plastik Tipi': 'material_v',
        'Hiz (m/s)': 'velocity_ms',
        'Deney': 'experiment',
        'Boyut 1': 'a_v',
        'Boyut 2': 'b_v',
        'Boyut 3': 'c_v'
    })

    return df


def merge_data(particles_df, velocity_df):
    """Parçacık ve hız verilerini birleştir"""

    # Parçacık kodunu temizle
    particles_df['particle_code_clean'] = particles_df['particle_code'].str.strip().str.upper()
    particles_df['particle_code_clean'] = particles_df['particle_code_clean'].str.replace(r'^(CYLINDER|CUBE|BSP|HC|EC|HALF\s*CYLINDER|WEDGE-SHAPED|BOX-SHAPED PRISM|ELLIPTICAL\s*CYLINDER)-?', '', regex=True)
    particles_df['particle_code_clean'] = particles_df['particle_code_clean'].str.strip()

    velocity_df['particle_code_clean'] = velocity_df['particle_code'].str.strip().str.upper()

    # Shape bilgisini de kullanarak eşleştir
    merged_list = []

    for _, vel_row in velocity_df.iterrows():
        code = vel_row['particle_code_clean']
        category = str(vel_row.get('category', '')).upper()

        # Kategoriyi shape'e çevir
        shape_map = {
            'BSP': 'BSP', 'WSP': 'WSP', 'HC': 'HC', 'C': 'C',
            'CUBE': 'CUBE', 'EC': 'EC',
            'ABS C': 'C', 'ABS CUBE': 'CUBE', 'ABS HC': 'HC', 'ABS EC': 'EC',
            'PLA C': 'C', 'PLA CUBE': 'CUBE', 'PLA HC': 'HC',
            'RESIN': 'CUBE', 'PS': 'EC', 'PA 6': 'C'
        }

        shape = None
        for key, val in shape_map.items():
            if key in category:
                shape = val
                break

        # Parçacık verilerinde eşleşme ara
        match = particles_df[particles_df['particle_code_clean'] == code]

        if len(match) > 0:
            # Shape eşleşmesi de kontrol et
            if shape:
                shape_match = match[match['shape'] == shape]
                if len(shape_match) > 0:
                        match = shape_match.iloc[0]
                else:
                    match = match.iloc[0]
            else:
                match = match.iloc[0]

            merged_row = vel_row.to_dict()
            merged_row.update({
                'a_mm': match['a_mm'],
                'b_mm': match['b_mm'],
                'c_mm': match['c_mm'],
                'volume_mm3': match['volume_mm3'],
                'surface_area_mm2': match['surface_area_mm2'],
                'density_kg_m3': match['density_kg_m3'],
                'weight_g': match['weight_g'],
                'matched': True
            })
            merged_list.append(merged_row)
        else:
            # Eşleşme bulunamadı - velocity dosyasındaki boyutları kullan
            merged_row = vel_row.to_dict()
            merged_row.update({
                'a_mm': vel_row.get('a_v'),
                'b_mm': vel_row.get('b_v'),
                'c_mm': vel_row.get('c_v'),
                'volume_mm3': None,
                'surface_area_mm2': None,
                'density_kg_m3': None,
                'weight_g': None,
                'matched': False
            })
            merged_list.append(merged_row)

    return pd.DataFrame(merged_list)


def calculate_features(df):
    """Feature engineering - shape factor'ları hesapla"""

    # Eksik boyutları doldur
    df['a_mm'] = pd.to_numeric(df['a_mm'], errors='coerce')
    df['b_mm'] = pd.to_numeric(df['b_mm'], errors='coerce')
    df['c_mm'] = pd.to_numeric(df['c_mm'], errors='coerce')

    # c eksikse a veya b ile doldur (küp/küre varsayımı)
    df['c_mm'] = df['c_mm'].fillna(df[['a_mm', 'b_mm']].mean(axis=1))

    # Boyutları sırala (a >= b >= c)
    dims = df[['a_mm', 'b_mm', 'c_mm']].values
    dims_sorted = np.sort(dims, axis=1)[:, ::-1]  # Büyükten küçüğe
    df['L'] = dims_sorted[:, 0]  # En uzun
    df['I'] = dims_sorted[:, 1]  # Orta
    df['S'] = dims_sorted[:, 2]  # En kısa

    # Eşdeğer çap (mm)
    df['d_eq'] = (df['a_mm'] * df['b_mm'] * df['c_mm']) ** (1/3)

    # Hacim (mm³) - yoksa hesapla (dikdörtgen prizma varsayımı)
    df['volume_mm3'] = df['volume_mm3'].fillna(df['a_mm'] * df['b_mm'] * df['c_mm'])

    # Yüzey alanı (mm²) - yoksa hesapla
    df['surface_area_mm2'] = df['surface_area_mm2'].fillna(
        2 * (df['a_mm']*df['b_mm'] + df['b_mm']*df['c_mm'] + df['a_mm']*df['c_mm'])
    )

    # Sphericity (küresellik)
    # Ψ = π^(1/3) * (6V)^(2/3) / A
    df['sphericity'] = (np.pi ** (1/3)) * ((6 * df['volume_mm3']) ** (2/3)) / df['surface_area_mm2']

    # Aspect Ratio (en/boy oranı)
    df['aspect_ratio'] = df['L'] / df['S']

    # Corey Shape Factor: CSF = S / sqrt(L * I)
    df['CSF'] = df['S'] / np.sqrt(df['L'] * df['I'])

    # Flatness: S / I
    df['flatness'] = df['S'] / df['I']

    # Elongation: I / L
    df['elongation'] = df['I'] / df['L']

    # Yoğunluk farkı (parçacık - su)
    df['density_kg_m3'] = pd.to_numeric(df['density_kg_m3'], errors='coerce')

    # Eksik yoğunlukları malzemeye göre doldur
    density_defaults = {
        'ABS': 1135,
        'PLA': 1200,
        'PS': 1047,
        'PMMA': 1200,
        'Plexiglass': 1200,
        'RESIN': 1150,
        'PA6': 1200
    }

    for mat, dens in density_defaults.items():
        mask = (df['density_kg_m3'].isna()) & (df['material_v'].str.contains(mat, case=False, na=False))
        df.loc[mask, 'density_kg_m3'] = dens

    # Kalan eksikleri ortalama ile doldur
    df['density_kg_m3'] = df['density_kg_m3'].fillna(1150)

    # Yoğunluk oranı ve farkı
    df['density_ratio'] = df['density_kg_m3'] / 1000  # ρ_p / ρ_water
    df['density_diff'] = df['density_kg_m3'] - 1000   # ρ_p - ρ_water

    return df


def prepare_ml_data(df):
    """ML için veri hazırla"""

    # Outlier temizle (hız > 0.5 m/s)
    df = df[df['velocity_ms'] <= 0.5].copy()
    df = df[df['velocity_ms'] > 0].copy()

    # Feature listesi
    numeric_features = [
        'a_mm', 'b_mm', 'c_mm',
        'd_eq', 'volume_mm3', 'surface_area_mm2',
        'sphericity', 'aspect_ratio', 'CSF',
        'flatness', 'elongation',
        'density_kg_m3', 'density_ratio', 'density_diff'
    ]

    # Kategorik özellikler
    categorical_features = ['category', 'view']

    # Eksik değerleri temizle
    df = df.dropna(subset=['velocity_ms'])

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Kategorik encoding
    le_dict = {}
    for cat in categorical_features:
        if cat in df.columns:
            le = LabelEncoder()
            df[f'{cat}_encoded'] = le.fit_transform(df[cat].astype(str))
            le_dict[cat] = le
            numeric_features.append(f'{cat}_encoded')

    # NaN değerleri doldur
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

    # Inf değerleri temizle
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=numeric_features)

    X = df[numeric_features]
    y = df['velocity_ms']

    return X, y, numeric_features, df


def train_model(X, y, feature_names):
    """Model eğit ve değerlendir"""

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalizasyon
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # 1. Random Forest (baseline)
    print("\n" + "="*50)
    print("Random Forest Regressor")
    print("="*50)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    y_pred_rf = rf.predict(X_test_scaled)

    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"RMSE: {rmse_rf:.6f} m/s")
    print(f"MAE:  {mae_rf:.6f} m/s")
    print(f"R²:   {r2_rf:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    results['RandomForest'] = {
        'model': rf,
        'rmse': rmse_rf,
        'mae': mae_rf,
        'r2': r2_rf,
        'cv_r2': cv_scores.mean()
    }

    # Feature importance
    print("\nFeature Importance (Top 10):")
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in importance.head(10).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")

    # 2. XGBoost (eğer yüklüyse)
    if HAS_XGBOOST:
        print("\n" + "="*50)
        print("XGBoost Regressor")
        print("="*50)

        xgb = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        xgb.fit(X_train_scaled, y_train)

        y_pred_xgb = xgb.predict(X_test_scaled)

        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        r2_xgb = r2_score(y_test, y_pred_xgb)

        print(f"RMSE: {rmse_xgb:.6f} m/s")
        print(f"MAE:  {mae_xgb:.6f} m/s")
        print(f"R²:   {r2_xgb:.4f}")

        cv_scores_xgb = cross_val_score(xgb, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"CV R² (5-fold): {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")

        results['XGBoost'] = {
            'model': xgb,
            'rmse': rmse_xgb,
            'mae': mae_xgb,
            'r2': r2_xgb,
            'cv_r2': cv_scores_xgb.mean()
        }

    return results, scaler, X_test, y_test


def main():
    print("="*60)
    print("MİKROPLASTİK ÇÖKELME HIZI TAHMİN MODELİ")
    print("="*60)

    # 1. Veri yükleme
    print("\n[1/4] Veriler yükleniyor...")

    particles_df = load_particle_measurements('ALL PARTICLES MEASUREMENTS.xlsx')
    print(f"  Parçacık ölçümleri: {len(particles_df)} kayıt")

    velocity_df = load_velocity_data('Video_Boyut_Eslestirme_FINAL.xlsx')
    print(f"  Hız verileri: {len(velocity_df)} kayıt")

    # 2. Veri birleştirme
    print("\n[2/4] Veriler birleştiriliyor...")
    merged_df = merge_data(particles_df, velocity_df)
    matched = merged_df['matched'].sum()
    print(f"  Eşleşen: {matched} / {len(merged_df)}")

    # 3. Feature engineering
    print("\n[3/4] Feature engineering...")
    merged_df = calculate_features(merged_df)

    # 4. ML hazırlık
    print("\n[4/4] Model eğitimi...")
    X, y, feature_names, clean_df = prepare_ml_data(merged_df)
    print(f"  Kullanılabilir veri: {len(X)} satır")
    print(f"  Feature sayısı: {len(feature_names)}")

    # Model eğitimi
    results, scaler, X_test, y_test = train_model(X, y, feature_names)

    # Özet
    print("\n" + "="*60)
    print("ÖZET")
    print("="*60)

    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"\nEn iyi model: {best_model[0]}")
    print(f"  R²:   {best_model[1]['r2']:.4f}")
    print(f"  RMSE: {best_model[1]['rmse']:.6f} m/s")
    print(f"  MAE:  {best_model[1]['mae']:.6f} m/s")

    # Veriyi kaydet
    clean_df.to_csv('ml_prepared_data.csv', index=False, encoding='utf-8')
    print(f"\nHazırlanan veri kaydedildi: ml_prepared_data.csv")

    return results, clean_df


if __name__ == "__main__":
    results, data = main()
