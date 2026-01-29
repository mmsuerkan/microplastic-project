"""Deney 8: Sample Weighting - RESIN'e daha fazla ağırlık"""
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
print(f"Veri: {len(df)} parçacık")
print(f"RESIN: {df['category'].str.contains('RESIN').sum()} parçacık")
print(f"Diğer: {(~df['category'].str.contains('RESIN')).sum()} parçacık")

# Features
feature_cols = ['a', 'b', 'c', 'density', 'shape_enc',
                'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio']

X = df[feature_cols].values
y = df['velocity_mean'].values
is_resin = df['category'].str.contains('RESIN').values

# ================================================================
# FARKLI AĞIRLIKLARLA DENEME
# ================================================================
print("\n" + "="*70)
print("SAMPLE WEIGHTING DENEMELERİ")
print("="*70)

weight_configs = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

for resin_weight in weight_configs:
    # Sample weights oluştur
    sample_weights = np.where(is_resin, resin_weight, 1.0)

    rf_scores = []
    rf_resin_mae = []
    rf_other_mae = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = sample_weights[train_idx]
        is_resin_val = is_resin[val_idx]

        # RF with sample weights
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr, sample_weight=w_tr)
        y_pred = rf.predict(X_val)

        # Overall R²
        r2 = 1 - np.sum((y_val - y_pred)**2) / np.sum((y_val - y_val.mean())**2)
        rf_scores.append(r2)

        # RESIN MAE
        if is_resin_val.sum() > 0:
            resin_mae = np.abs(y_val[is_resin_val] - y_pred[is_resin_val]).mean()
            rf_resin_mae.append(resin_mae)

        # Other MAE
        if (~is_resin_val).sum() > 0:
            other_mae = np.abs(y_val[~is_resin_val] - y_pred[~is_resin_val]).mean()
            rf_other_mae.append(other_mae)

    results.append({
        'weight': resin_weight,
        'cv_r2': np.mean(rf_scores),
        'cv_r2_std': np.std(rf_scores),
        'resin_mae': np.mean(rf_resin_mae),
        'other_mae': np.mean(rf_other_mae),
    })

    print(f"\nRESIN weight = {resin_weight}x:")
    print(f"  CV R²:      {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
    print(f"  RESIN MAE:  {np.mean(rf_resin_mae):.2f} cm/s")
    print(f"  Diğer MAE:  {np.mean(rf_other_mae):.2f} cm/s")

# ================================================================
# SONUÇ TABLOSU
# ================================================================
print("\n" + "="*70)
print("SONUÇ TABLOSU")
print("="*70)

print(f"\n{'Weight':<10} {'CV R²':<15} {'RESIN MAE':<12} {'Diğer MAE':<12} {'Trade-off':<10}")
print("-" * 60)

baseline_r2 = results[0]['cv_r2']
baseline_resin = results[0]['resin_mae']
baseline_other = results[0]['other_mae']

for r in results:
    r2_change = r['cv_r2'] - baseline_r2
    resin_change = baseline_resin - r['resin_mae']  # Pozitif = iyileşme
    other_change = baseline_other - r['other_mae']

    trade_off = "✓" if r['cv_r2'] >= baseline_r2 - 0.01 and r['resin_mae'] < baseline_resin else ""

    print(f"{r['weight']:<10} {r['cv_r2']:.4f} ({r2_change:+.3f})  "
          f"{r['resin_mae']:.2f} ({resin_change:+.2f})  "
          f"{r['other_mae']:.2f} ({other_change:+.2f})  {trade_off}")

# En iyi weight bul (R² düşmeden RESIN MAE en az)
best_weight = None
best_resin_mae = float('inf')
for r in results:
    if r['cv_r2'] >= baseline_r2 - 0.02 and r['resin_mae'] < best_resin_mae:
        best_weight = r['weight']
        best_resin_mae = r['resin_mae']

print(f"\n✓ Önerilen weight: {best_weight}x")
print(f"  (R² çok düşmeden RESIN MAE en az)")

# ================================================================
# EN İYİ WEIGHT İLE ENSEMBLE
# ================================================================
print("\n" + "="*70)
print(f"EN İYİ WEIGHT ({best_weight}x) İLE ENSEMBLE")
print("="*70)

sample_weights = np.where(is_resin, best_weight, 1.0)

rf_scores = []
nn_scores = []
ens_scores = []
resin_maes = []
other_maes = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    w_tr = sample_weights[train_idx]
    is_resin_val = is_resin[val_idx]

    # Scaler
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # RF
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr, sample_weight=w_tr)
    rf_pred = rf.predict(X_val)
    rf_r2 = 1 - np.sum((y_val - rf_pred)**2) / np.sum((y_val - y_val.mean())**2)
    rf_scores.append(rf_r2)

    # NN (sample weight desteklemiyor, normal eğit)
    nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True)
    nn.fit(X_tr_s, y_tr)
    nn_pred = nn.predict(X_val_s)
    nn_r2 = 1 - np.sum((y_val - nn_pred)**2) / np.sum((y_val - y_val.mean())**2)
    nn_scores.append(nn_r2)

    # Ensemble
    ens_pred = 0.6 * rf_pred + 0.4 * nn_pred
    ens_r2 = 1 - np.sum((y_val - ens_pred)**2) / np.sum((y_val - y_val.mean())**2)
    ens_scores.append(ens_r2)

    # MAE
    if is_resin_val.sum() > 0:
        resin_maes.append(np.abs(y_val[is_resin_val] - ens_pred[is_resin_val]).mean())
    if (~is_resin_val).sum() > 0:
        other_maes.append(np.abs(y_val[~is_resin_val] - ens_pred[~is_resin_val]).mean())

print(f"\nRF:       {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
print(f"NN:       {np.mean(nn_scores):.4f} ± {np.std(nn_scores):.4f}")
print(f"Ensemble: {np.mean(ens_scores):.4f} ± {np.std(ens_scores):.4f}")
print(f"\nRESIN MAE:  {np.mean(resin_maes):.2f} cm/s")
print(f"Diğer MAE:  {np.mean(other_maes):.2f} cm/s")

# ================================================================
# KARŞILAŞTIRMA
# ================================================================
print("\n" + "="*70)
print("KARŞILAŞTIRMA: Baseline vs Sample Weight")
print("="*70)

print(f"\n{'Metrik':<20} {'Baseline (w=1)':<18} {'Weight={:.1f}':<18} {'Değişim':<15}".format(best_weight))
print("-" * 70)
print(f"{'Ensemble R²':<20} {'0.83':<18} {np.mean(ens_scores):.2f}{'':<14} {np.mean(ens_scores)-0.83:+.2f}")
print(f"{'RESIN MAE':<20} {'2.82 cm/s':<18} {np.mean(resin_maes):.2f} cm/s{'':<10} {np.mean(resin_maes)-2.82:+.2f} cm/s")
print(f"{'Diğer MAE':<20} {'1.50 cm/s':<18} {np.mean(other_maes):.2f} cm/s{'':<10} {np.mean(other_maes)-1.50:+.2f} cm/s")
