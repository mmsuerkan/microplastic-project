"""Deney 7: Stokes-inspired feature ile model eğitimi"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

# Veri yükle
df = pd.read_csv('data/training_data_particle_avg.csv')
print(f"Veri: {len(df)} parçacık\n")

# ================================================================
# STOKES-INSPIRED FEATURE'LAR EKLE
# ================================================================
print("="*70)
print("YENİ FİZİKSEL FEATURE'LAR")
print("="*70)

# Suyun density'si (yaklaşık 1000 kg/m³ @ 20°C)
rho_fluid = 1000

# 1. Stokes Factor: (ρp - ρf) * a²
# Stokes Law: v ∝ (ρp - ρf) * r²
df['stokes_factor'] = (df['density'] - rho_fluid) * (df['a'] ** 2)

# 2. Equivalent diameter (tüm boyutların geometrik ortalaması)
# Sphere için a, diğerleri için (a*b*c)^(1/3) veya (a*b)^(1/2)
df['equiv_diameter'] = np.where(
    df['shape_name'] == 'Sphere',
    df['a'],  # Sphere için zaten çap
    np.where(
        df['c'] > 0,
        (df['a'] * df['b'] * df['c']) ** (1/3),  # 3D shapes
        (df['a'] * df['b']) ** (1/2)  # 2D shapes (cylinder etc)
    )
)

# 3. Buoyancy factor: (ρp - ρf) / ρf
df['buoyancy_factor'] = (df['density'] - rho_fluid) / rho_fluid

# 4. Size squared (Stokes'ta r² var)
df['size_squared'] = df['a'] ** 2

# 5. Stokes velocity estimate (simplified)
# v_stokes ∝ (ρp - ρf) * d² / μ, μ sabit kabul edersek:
df['stokes_velocity_est'] = df['stokes_factor'] / 1000  # Normalize

print("\nYeni feature'lar:")
print(f"  stokes_factor:       (density - 1000) × a²")
print(f"  equiv_diameter:      Eşdeğer çap")
print(f"  buoyancy_factor:     (density - 1000) / 1000")
print(f"  size_squared:        a²")
print(f"  stokes_velocity_est: Stokes tahmin (normalize)")

# Yeni feature korelasyonları
print("\n--- Yeni Feature Korelasyonları (velocity ile) ---")
new_features = ['stokes_factor', 'equiv_diameter', 'buoyancy_factor', 'size_squared', 'stokes_velocity_est']
for feat in new_features:
    corr = df[feat].corr(df['velocity_mean'])
    bar = '█' * int(abs(corr) * 30)
    print(f"  {feat:<22} {corr:+.3f} {bar}")

# ================================================================
# MODEL EĞİTİMİ
# ================================================================
print("\n" + "="*70)
print("MODEL KARŞILAŞTIRMASI")
print("="*70)

# Feature setleri
feature_sets = {
    'Baseline (9 feat)': ['a', 'b', 'c', 'density', 'shape_enc',
                          'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio'],

    'Stokes Added (11 feat)': ['a', 'b', 'c', 'density', 'shape_enc',
                                'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio',
                                'stokes_factor', 'buoyancy_factor'],

    'Stokes Only (6 feat)': ['stokes_factor', 'equiv_diameter', 'buoyancy_factor',
                             'size_squared', 'shape_enc', 'density'],

    'Best Physics (8 feat)': ['stokes_factor', 'equiv_diameter', 'aspect_ratio',
                              'shape_enc', 'density', 'volume', 'a', 'b'],
}

y = df['velocity_mean'].values
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for set_name, features in feature_sets.items():
    X = df[features].values

    rf_scores = []
    nn_scores = []
    ens_scores = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Scaler
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # RF
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        rf_pred = rf.predict(X_val)
        rf_r2 = 1 - np.sum((y_val - rf_pred)**2) / np.sum((y_val - y_val.mean())**2)
        rf_scores.append(rf_r2)

        # NN
        nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True)
        nn.fit(X_tr_s, y_tr)
        nn_pred = nn.predict(X_val_s)
        nn_r2 = 1 - np.sum((y_val - nn_pred)**2) / np.sum((y_val - y_val.mean())**2)
        nn_scores.append(nn_r2)

        # Ensemble
        ens_pred = 0.6 * rf_pred + 0.4 * nn_pred
        ens_r2 = 1 - np.sum((y_val - ens_pred)**2) / np.sum((y_val - y_val.mean())**2)
        ens_scores.append(ens_r2)

    results[set_name] = {
        'RF': (np.mean(rf_scores), np.std(rf_scores)),
        'NN': (np.mean(nn_scores), np.std(nn_scores)),
        'Ensemble': (np.mean(ens_scores), np.std(ens_scores)),
    }

    print(f"\n{set_name}:")
    print(f"  RF:       {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
    print(f"  NN:       {np.mean(nn_scores):.4f} ± {np.std(nn_scores):.4f}")
    print(f"  Ensemble: {np.mean(ens_scores):.4f} ± {np.std(ens_scores):.4f}")

# ================================================================
# EN İYİ MODEL ANALİZİ
# ================================================================
print("\n" + "="*70)
print("SONUÇ KARŞILAŞTIRMASI")
print("="*70)

print("\n--- Ensemble Skorları ---")
print(f"{'Feature Set':<25} {'CV R²':<15} {'Değişim':<10}")
print("-" * 50)
baseline_score = results['Baseline (9 feat)']['Ensemble'][0]
for set_name, scores in results.items():
    ens_mean, ens_std = scores['Ensemble']
    change = ens_mean - baseline_score
    change_str = f"{change:+.4f}" if set_name != 'Baseline (9 feat)' else "-"
    print(f"{set_name:<25} {ens_mean:.4f} ± {ens_std:.4f}  {change_str}")

# En iyi set
best_set = max(results.keys(), key=lambda k: results[k]['Ensemble'][0])
best_score = results[best_set]['Ensemble'][0]
print(f"\n✓ En iyi feature set: {best_set}")
print(f"  CV R²: {best_score:.4f}")

# Feature importance for best set
if 'Stokes' in best_set or best_set == 'Best Physics (8 feat)':
    print("\n--- Feature Importance (En İyi Set - RF) ---")
    best_features = feature_sets[best_set]
    X_best = df[best_features].values
    rf_final = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_final.fit(X_best, y)

    for feat, imp in sorted(zip(best_features, rf_final.feature_importances_), key=lambda x: -x[1]):
        bar = '█' * int(imp * 40)
        print(f"  {feat:<20} {imp:.3f} {bar}")

# Veriyi kaydet
df.to_csv('data/training_data_stokes.csv', index=False)
print(f"\nVeri kaydedildi: data/training_data_stokes.csv")

# ================================================================
# HATA ANALİZİ (En iyi model ile)
# ================================================================
print("\n" + "="*70)
print("HATA ANALİZİ (En İyi Model)")
print("="*70)

from sklearn.model_selection import cross_val_predict

best_features = feature_sets[best_set]
X_best = df[best_features].values
rf_final = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
y_pred_cv = cross_val_predict(rf_final, X_best, y, cv=5)

df['predicted_new'] = y_pred_cv
df['error_new'] = np.abs(df['velocity_mean'] - df['predicted_new'])

# RESIN vs diğerleri hata karşılaştırması
resin_error_old = 2.82  # Önceki analiz
resin_error_new = df[df['category'].str.contains('RESIN')]['error_new'].mean()
other_error_new = df[~df['category'].str.contains('RESIN')]['error_new'].mean()

print(f"\n--- RESIN Hata Karşılaştırması ---")
print(f"  Önceki model MAE: 2.82 cm/s")
print(f"  Yeni model MAE:   {resin_error_new:.2f} cm/s")
print(f"  İyileşme:         {2.82 - resin_error_new:.2f} cm/s ({(2.82 - resin_error_new)/2.82*100:.0f}%)")

print(f"\n--- Şekil Bazlı Hata ---")
shape_errors = df.groupby('shape_name')['error_new'].mean().sort_values(ascending=False)
for shape, error in shape_errors.items():
    print(f"  {shape:<20} {error:.2f} cm/s")
