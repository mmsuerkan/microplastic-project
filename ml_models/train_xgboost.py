"""Deney 6: XGBoost ve LightGBM ile model eğitimi"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import sys
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

# Parçacık bazlı ortalama veri yükle
df = pd.read_csv('data/training_data_particle_avg.csv')
print(f"Veri: {len(df)} parçacık")

# Features
feature_cols = ['a', 'b', 'c', 'density', 'shape_enc',
                'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio']

X = df[feature_cols].values
y = df['velocity_mean'].values

print(f"Features: {len(feature_cols)}")
print(f"Target: velocity_mean")

# 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Modeller
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                            random_state=42, verbosity=0, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                              random_state=42, verbosity=-1, n_jobs=-1),
}

print("\n" + "="*60)
print("MODEL KARŞILAŞTIRMASI (5-Fold CV)")
print("="*60)

results = {}

for name, model in models.items():
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_tr, y_tr)
        y_pred = model_copy.predict(X_val)

        r2 = 1 - np.sum((y_val - y_pred)**2) / np.sum((y_val - y_val.mean())**2)
        scores.append(r2)

    results[name] = scores
    print(f"\n{name}:")
    print(f"  CV R²: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"  Fold skorları: {[f'{s:.3f}' for s in scores]}")

# En iyi model
print("\n" + "="*60)
print("SONUÇ")
print("="*60)

best_model = max(results.keys(), key=lambda k: np.mean(results[k]))
print(f"\nEn iyi model: {best_model}")
print(f"CV R²: {np.mean(results[best_model]):.4f} ± {np.std(results[best_model]):.4f}")

# XGBoost hiperparametre optimizasyonu
print("\n" + "="*60)
print("XGBOOST HİPERPARAMETRE DENEMELERİ")
print("="*60)

xgb_configs = [
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
    {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
    {'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.1},
    {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
    {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1},
    {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.2},
]

best_xgb_score = 0
best_xgb_config = None

for config in xgb_configs:
    model = XGBRegressor(**config, random_state=42, verbosity=0, n_jobs=-1)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model_copy = XGBRegressor(**config, random_state=42, verbosity=0, n_jobs=-1)
        model_copy.fit(X_tr, y_tr)
        y_pred = model_copy.predict(X_val)
        r2 = 1 - np.sum((y_val - y_pred)**2) / np.sum((y_val - y_val.mean())**2)
        scores.append(r2)

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"  n={config['n_estimators']}, d={config['max_depth']}, lr={config['learning_rate']}: "
          f"R²={mean_score:.4f} ± {std_score:.4f}")

    if mean_score > best_xgb_score:
        best_xgb_score = mean_score
        best_xgb_config = config.copy()
        best_xgb_std = std_score

print(f"\nEn iyi XGBoost config: {best_xgb_config}")
print(f"CV R²: {best_xgb_score:.4f} ± {best_xgb_std:.4f}")

# Feature Importance (XGBoost)
print("\n" + "="*60)
print("FEATURE IMPORTANCE (XGBoost)")
print("="*60)

xgb_best = XGBRegressor(**best_xgb_config, random_state=42, verbosity=0, n_jobs=-1)
xgb_best.fit(X, y)

importances = xgb_best.feature_importances_
for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")

# Karşılaştırma özeti
print("\n" + "="*60)
print("TÜM MODELLER KARŞILAŞTIRMASI")
print("="*60)
print(f"\nDeney 5 (RF+NN Ensemble): 0.83 ± 0.06")
print(f"Random Forest:            {np.mean(results['Random Forest']):.2f} ± {np.std(results['Random Forest']):.2f}")
print(f"XGBoost (default):        {np.mean(results['XGBoost']):.2f} ± {np.std(results['XGBoost']):.2f}")
print(f"XGBoost (tuned):          {best_xgb_score:.2f} ± {best_xgb_std:.2f}")
print(f"LightGBM:                 {np.mean(results['LightGBM']):.2f} ± {np.std(results['LightGBM']):.2f}")
