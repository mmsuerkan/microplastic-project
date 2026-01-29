"""Deney 5: Parçacık bazlı ortalama velocity ile model eğitimi"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

# Veri yükle (feature'lar dahil)
df = pd.read_csv('data/training_data_v2_features.csv')
print(f"Orijinal veri: {len(df)} satır")

# Feature'ları hesapla (eğer yoksa)
if 'volume' not in df.columns:
    def calc_volume(row):
        shape = row['shape_enc']
        a, b, c = row['a'], row['b'], row['c']
        if shape == 0:  # Cylinder
            return np.pi * (b/2)**2 * a if b > 0 else np.pi * (a/2)**2 * a
        elif shape == 1:  # Half Cylinder
            return 0.5 * np.pi * (b/2)**2 * a if b > 0 else 0.5 * np.pi * (a/2)**2 * a
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
        elif shape == 0:  # Cylinder
            r = b/2 if b > 0 else a/2
            return 2 * np.pi * r * (r + a)
        return 2 * (a*b + b*c + a*c) if b > 0 and c > 0 else 6 * a**2

    df['volume'] = df.apply(calc_volume, axis=1)
    df['surface_area'] = df.apply(calc_surface_area, axis=1)
    df['aspect_ratio'] = df['a'] / np.where(df['b'] == 0, df['a'], df['b'])
    df['vol_surf_ratio'] = df['volume'] / np.where(df['surface_area'] == 0, 1, df['surface_area'])

# Unique parçacık sayısı
unique_particles = df.groupby(['category', 'code']).size()
print(f"Unique parçacık: {len(unique_particles)}")

# Parçacık bazlı ortalama
particle_avg = df.groupby(['category', 'code']).agg({
    'shape_enc': 'first',
    'shape_name': 'first',
    'a': 'first',
    'b': 'first',
    'c': 'first',
    'density': 'first',
    'velocity_cms': ['mean', 'std', 'count'],
    'volume': 'first',
    'surface_area': 'first',
    'aspect_ratio': 'first',
    'vol_surf_ratio': 'first'
}).reset_index()

# Flatten column names
particle_avg.columns = ['category', 'code', 'shape_enc', 'shape_name', 'a', 'b', 'c',
                        'density', 'velocity_mean', 'velocity_std', 'measurement_count',
                        'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio']

# NaN std'leri 0 yap (tek ölçüm olanlar)
particle_avg['velocity_std'] = particle_avg['velocity_std'].fillna(0)

print(f"\nParçacık bazlı veri: {len(particle_avg)} satır")
print(f"Ortalama ölçüm sayısı: {particle_avg['measurement_count'].mean():.1f}")

# İstatistikler
print("\n" + "="*60)
print("PARÇACIK BAZLI VERİ İSTATİSTİKLERİ")
print("="*60)
print(f"\nŞekil dağılımı:")
for shape in particle_avg['shape_name'].unique():
    count = len(particle_avg[particle_avg['shape_name'] == shape])
    mean_vel = particle_avg[particle_avg['shape_name'] == shape]['velocity_mean'].mean()
    print(f"  {shape}: {count} parçacık, ortalama {mean_vel:.2f} cm/s")

# CSV kaydet
particle_avg.to_csv('data/training_data_particle_avg.csv', index=False)
print(f"\nKaydedildi: data/training_data_particle_avg.csv")

# Model eğitimi
print("\n" + "="*60)
print("MODEL EĞİTİMİ (PARÇACIK BAZLI ORTALAMA)")
print("="*60)

# Features
feature_cols = ['a', 'b', 'c', 'density', 'shape_enc',
                'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio']

X = particle_avg[feature_cols].values
y = particle_avg['velocity_mean'].values

print(f"\nFeatures: {feature_cols}")
print(f"Veri boyutu: {X.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Manuel 5-Fold CV (tüm modeller için aynı fold'lar)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf_scores = []
nn_scores = []
ensemble_scores = []

print("\n--- 5-Fold Cross Validation ---")
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Scaler
    scaler_temp = StandardScaler()
    X_tr_s = scaler_temp.fit_transform(X_tr)
    X_val_s = scaler_temp.transform(X_val)

    # Random Forest
    rf_temp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_temp.fit(X_tr, y_tr)
    rf_pred = rf_temp.predict(X_val)
    rf_r2 = 1 - np.sum((y_val - rf_pred)**2) / np.sum((y_val - y_val.mean())**2)
    rf_scores.append(rf_r2)

    # Neural Network
    nn_temp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True)
    nn_temp.fit(X_tr_s, y_tr)
    nn_pred = nn_temp.predict(X_val_s)
    nn_r2 = 1 - np.sum((y_val - nn_pred)**2) / np.sum((y_val - y_val.mean())**2)
    nn_scores.append(nn_r2)

    # Ensemble
    ens_pred = 0.6 * rf_pred + 0.4 * nn_pred
    ens_r2 = 1 - np.sum((y_val - ens_pred)**2) / np.sum((y_val - y_val.mean())**2)
    ensemble_scores.append(ens_r2)

    print(f"  Fold {fold}: RF={rf_r2:.3f}, NN={nn_r2:.3f}, Ensemble={ens_r2:.3f}")

print(f"\n--- Random Forest ---")
print(f"CV R²: {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")

print(f"\n--- Neural Network ---")
print(f"CV R²: {np.mean(nn_scores):.4f} ± {np.std(nn_scores):.4f}")

print(f"\n--- Ensemble (RF 60% + NN 40%) ---")
print(f"CV R²: {np.mean(ensemble_scores):.4f} ± {np.std(ensemble_scores):.4f}")

# Final model (tüm veri ile)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Feature importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE (RF)")
print("="*60)
importances = rf.feature_importances_
for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")

# Karşılaştırma
print("\n" + "="*60)
print("KARŞILAŞTIRMA: Tüm Ölçümler vs Parçacık Ortalaması")
print("="*60)
print(f"\nTüm ölçümler (827 satır):")
print(f"  RF CV R²: 0.81 ± 0.05")
print(f"\nParçacık ortalaması ({len(particle_avg)} satır):")
print(f"  RF CV R²: {np.mean(rf_scores):.2f} ± {np.std(rf_scores):.2f}")
