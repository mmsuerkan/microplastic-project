"""Yeni veri ile model eğitimi - 265 parçacık"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import json
import sys
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

# Veri yükle
df = pd.read_csv('data/training_data_with_fails.csv')
print(f"Toplam veri: {len(df)} parçacık")
print(f"  - Original: {len(df[df['source'] == 'original'])}")
print(f"  - Fail recovery: {len(df[df['source'] == 'fail_recovery'])}")

# Features
feature_cols = ['a', 'b', 'c', 'density', 'shape_enc',
                'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio']

X = df[feature_cols].values
y = df['velocity_mean'].values

print(f"\nFeature sayısı: {len(feature_cols)}")
print(f"Velocity range: {y.min():.2f} - {y.max():.2f} cm/s")

# En iyi parametreleri yükle
with open('ml_models/best_hyperparameters.json', 'r') as f:
    best_params = json.load(f)

rf_params = best_params['rf']
nn_params = best_params['nn']

print(f"\n--- Model Parametreleri ---")
print(f"RF: {rf_params}")
print(f"NN: {nn_params}")

# ================================================================
# 5-FOLD CV
# ================================================================
print("\n" + "="*70)
print("5-FOLD CROSS VALIDATION")
print("="*70)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf_scores = []
nn_scores = []
ens_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Scaler
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # RF
    rf = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_val)
    rf_r2 = 1 - np.sum((y_val - rf_pred)**2) / np.sum((y_val - y_val.mean())**2)
    rf_scores.append(rf_r2)

    # NN
    nn = MLPRegressor(**nn_params, max_iter=1000, random_state=42, early_stopping=True)
    nn.fit(X_tr_s, y_tr)
    nn_pred = nn.predict(X_val_s)
    nn_r2 = 1 - np.sum((y_val - nn_pred)**2) / np.sum((y_val - y_val.mean())**2)
    nn_scores.append(nn_r2)

    # Ensemble (50-50)
    ens_pred = 0.5 * rf_pred + 0.5 * nn_pred
    ens_r2 = 1 - np.sum((y_val - ens_pred)**2) / np.sum((y_val - y_val.mean())**2)
    ens_scores.append(ens_r2)

    print(f"Fold {fold+1}: RF={rf_r2:.4f}, NN={nn_r2:.4f}, Ensemble={ens_r2:.4f}")

print(f"\n--- ORTALAMA SONUÇLAR ---")
print(f"RF:       {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
print(f"NN:       {np.mean(nn_scores):.4f} ± {np.std(nn_scores):.4f}")
print(f"Ensemble: {np.mean(ens_scores):.4f} ± {np.std(ens_scores):.4f}")

# ================================================================
# ÖNCEKİ MODEL İLE KARŞILAŞTIRMA
# ================================================================
print("\n" + "="*70)
print("ÖNCEKİ MODEL İLE KARŞILAŞTIRMA")
print("="*70)

previous_score = 0.84  # Önceki en iyi
current_score = np.mean(ens_scores)
change = current_score - previous_score

print(f"""
{'Model':<25} {'Parçacık':<12} {'Ensemble R²':<15} {'Değişim':<10}
{'-'*60}
{'Önceki (228 parçacık)':<25} {'228':<12} {'0.8400':<15} {'-':<10}
{'Yeni (265 parçacık)':<25} {len(df):<12} {current_score:.4f}{'':<11} {change:+.4f}
""")

if change > 0:
    print(f"✓ Model İYİLEŞTİ: +{change:.2%}")
else:
    print(f"⚠ Model değişmedi veya hafif düştü: {change:+.2%}")

# ================================================================
# PMMA PERFORMANSI
# ================================================================
print("\n" + "="*70)
print("PMMA KATEGORİSİ PERFORMANSI")
print("="*70)

# Tüm veri üzerinde eğit
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf_final = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)
rf_final.fit(X, y)
rf_pred_all = rf_final.predict(X)

nn_final = MLPRegressor(**nn_params, max_iter=1000, random_state=42, early_stopping=True)
nn_final.fit(X_scaled, y)
nn_pred_all = nn_final.predict(X_scaled)

ens_pred_all = 0.5 * rf_pred_all + 0.5 * nn_pred_all

# Kategori bazlı MAE
df['pred'] = ens_pred_all
df['error'] = np.abs(df['velocity_mean'] - df['pred'])

print(f"\n{'Kategori':<15} {'MAE (cm/s)':<12} {'Parçacık':<10}")
print("-"*40)
for cat in df['category'].unique():
    cat_df = df[df['category'] == cat]
    mae = cat_df['error'].mean()
    count = len(cat_df)
    print(f"{cat:<15} {mae:.2f}{'':<8} {count:<10}")

# PMMA toplam
pmma_cats = ['BSP', 'C', 'HC', 'WSP']
pmma_df = df[df['category'].isin(pmma_cats)]
pmma_mae = pmma_df['error'].mean()
print("-"*40)
print(f"{'PMMA Toplam':<15} {pmma_mae:.2f}{'':<8} {len(pmma_df):<10}")

# ================================================================
# FEATURE IMPORTANCE
# ================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)

for feat, imp in sorted(zip(feature_cols, rf_final.feature_importances_), key=lambda x: -x[1]):
    bar = '█' * int(imp * 40)
    print(f"  {feat:<18} {imp:.3f} {bar}")

# ================================================================
# SONUÇ
# ================================================================
print("\n" + "="*70)
print("SONUÇ")
print("="*70)
print(f"""
Veri artışı:     228 → 265 parçacık (+{len(df)-228} / +{(len(df)-228)/228*100:.1f}%)
Ensemble R²:     0.84 → {current_score:.4f} ({change:+.4f})
PMMA parçacık:   73 → {len(pmma_df)} (+{len(pmma_df)-73})
""")
