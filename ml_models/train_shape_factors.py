"""Deney 10: Corey Shape Factor ve Sphericity ile model eğitimi"""
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
# SHAPE FACTOR HESAPLAMALARI
# ================================================================
print("="*70)
print("SHAPE FACTOR HESAPLAMALARI")
print("="*70)

print("""
Literatürdeki önemli shape parametreleri:

1. Corey Shape Factor (CSF): c / sqrt(a * b)
   - Sphere = 1, flat = 0'a yakın

2. Sphericity (Ψ): (π^(1/3) * (6V)^(2/3)) / A
   - Sphere = 1, irregular < 1

3. Flatness: c / b (en kısa / orta)

4. Elongation: b / a (orta / en uzun)
""")

# Boyutları sırala (a >= b >= c olacak şekilde)
def sort_dimensions(row):
    dims = [row['a'], row['b'], row['c']]
    # 0 olmayan değerleri al
    dims = [d for d in dims if d > 0]
    if len(dims) == 0:
        return row['a'], row['a'], row['a']  # fallback
    elif len(dims) == 1:
        return dims[0], dims[0], dims[0]  # sphere
    elif len(dims) == 2:
        dims.sort(reverse=True)
        return dims[0], dims[1], dims[1]  # cylinder (c = b varsay)
    else:
        dims.sort(reverse=True)
        return dims[0], dims[1], dims[2]  # 3D shape

# Sıralı boyutlar
sorted_dims = df.apply(sort_dimensions, axis=1)
df['dim_a'] = [d[0] for d in sorted_dims]  # en uzun
df['dim_b'] = [d[1] for d in sorted_dims]  # orta
df['dim_c'] = [d[2] for d in sorted_dims]  # en kısa

# 1. Corey Shape Factor (CSF)
# CSF = c / sqrt(a * b)
df['corey_sf'] = df['dim_c'] / np.sqrt(df['dim_a'] * df['dim_b'])

# 2. Sphericity (Wadell's sphericity)
# Ψ = (π^(1/3) * (6V)^(2/3)) / A
# Veya basitleştirilmiş: Ψ = (36 * π * V²)^(1/3) / A
df['sphericity'] = (np.pi ** (1/3)) * ((6 * df['volume']) ** (2/3)) / df['surface_area']

# 3. Flatness
# F = c / b
df['flatness'] = df['dim_c'] / df['dim_b']

# 4. Elongation
# E = b / a
df['elongation'] = df['dim_b'] / df['dim_a']

# 5. Powers roundness (basitleştirilmiş)
# PR = (a * b * c)^(1/3) / a
df['powers_roundness'] = (df['dim_a'] * df['dim_b'] * df['dim_c']) ** (1/3) / df['dim_a']

print("\n--- Yeni Feature İstatistikleri ---")
new_features = ['corey_sf', 'sphericity', 'flatness', 'elongation', 'powers_roundness']
for feat in new_features:
    print(f"  {feat:<18}: min={df[feat].min():.3f}, max={df[feat].max():.3f}, mean={df[feat].mean():.3f}")

# Korelasyonlar
print("\n--- Yeni Feature Korelasyonları (velocity ile) ---")
for feat in new_features:
    corr = df[feat].corr(df['velocity_mean'])
    bar = '█' * int(abs(corr) * 30)
    sign = '+' if corr > 0 else '-'
    print(f"  {feat:<18} {sign}{abs(corr):.3f} {bar}")

# Şekil bazlı shape factor
print("\n--- Şekil Bazlı Ortalama Shape Factors ---")
shape_sf = df.groupby('shape_name')[['corey_sf', 'sphericity', 'flatness', 'elongation']].mean().round(3)
print(shape_sf.to_string())

# ================================================================
# MODEL EĞİTİMİ
# ================================================================
print("\n" + "="*70)
print("MODEL KARŞILAŞTIRMASI")
print("="*70)

y = df['velocity_mean'].values
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Feature setleri
feature_sets = {
    'Baseline (9 feat)': ['a', 'b', 'c', 'density', 'shape_enc',
                          'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio'],

    'Shape Factors Added (14 feat)': ['a', 'b', 'c', 'density', 'shape_enc',
                                       'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio',
                                       'corey_sf', 'sphericity', 'flatness', 'elongation', 'powers_roundness'],

    'Literatür Style (10 feat)': ['dim_a', 'dim_b', 'dim_c', 'density',
                                   'volume', 'corey_sf', 'sphericity',
                                   'flatness', 'elongation', 'shape_enc'],

    'Best Combo (11 feat)': ['a', 'b', 'c', 'density', 'shape_enc',
                              'volume', 'aspect_ratio',
                              'corey_sf', 'sphericity', 'flatness', 'elongation'],
}

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

        # RF (tuned params)
        rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=10,
                                   min_samples_leaf=2, max_features='log2',
                                   random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        rf_pred = rf.predict(X_val)
        rf_r2 = 1 - np.sum((y_val - rf_pred)**2) / np.sum((y_val - y_val.mean())**2)
        rf_scores.append(rf_r2)

        # NN (tuned params)
        nn = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.001,
                          learning_rate_init=0.001, activation='relu',
                          max_iter=1000, random_state=42, early_stopping=True)
        nn.fit(X_tr_s, y_tr)
        nn_pred = nn.predict(X_val_s)
        nn_r2 = 1 - np.sum((y_val - nn_pred)**2) / np.sum((y_val - y_val.mean())**2)
        nn_scores.append(nn_r2)

        # Ensemble (50-50)
        ens_pred = 0.5 * rf_pred + 0.5 * nn_pred
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
# SONUÇ
# ================================================================
print("\n" + "="*70)
print("SONUÇ KARŞILAŞTIRMASI")
print("="*70)

print(f"\n{'Feature Set':<30} {'Ensemble R²':<20} {'Değişim':<10}")
print("-" * 60)
baseline_score = results['Baseline (9 feat)']['Ensemble'][0]
for set_name, scores in results.items():
    ens_mean, ens_std = scores['Ensemble']
    change = ens_mean - baseline_score
    change_str = f"{change:+.4f}" if set_name != 'Baseline (9 feat)' else "-"
    print(f"{set_name:<30} {ens_mean:.4f} ± {ens_std:.4f}  {change_str}")

# En iyi set
best_set = max(results.keys(), key=lambda k: results[k]['Ensemble'][0])
best_score = results[best_set]['Ensemble'][0]
print(f"\n✓ En iyi: {best_set}")
print(f"  Ensemble R²: {best_score:.4f}")

# Feature importance
print("\n--- Feature Importance (En İyi Set - RF) ---")
best_features = feature_sets[best_set]
X_best = df[best_features].values
rf_final = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=10,
                                  min_samples_leaf=2, max_features='log2',
                                  random_state=42, n_jobs=-1)
rf_final.fit(X_best, y)

for feat, imp in sorted(zip(best_features, rf_final.feature_importances_), key=lambda x: -x[1])[:10]:
    bar = '█' * int(imp * 40)
    print(f"  {feat:<18} {imp:.3f} {bar}")

# Karşılaştırma
print("\n" + "="*70)
print("LİTERATÜR KARŞILAŞTIRMASI")
print("="*70)
print(f"""
Literatür (Leng et al. 2024):  R² = 0.93-0.95 (2110 veri)
Önceki modelimiz:              R² = 0.84 (228 veri)
Shape factor eklenmiş:         R² = {best_score:.2f} (228 veri)
""")

# Veriyi kaydet
df.to_csv('data/training_data_shape_factors.csv', index=False)
print("Kaydedildi: data/training_data_shape_factors.csv")
