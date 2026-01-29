"""En iyi model parametrelerini ve modeli kaydet"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import joblib
import json
import sys
import os
warnings_module = __import__('warnings')
warnings_module.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

# Veri
df = pd.read_csv('data/training_data_particle_avg.csv')
print(f"Veri: {len(df)} parcacik\n")

feature_cols = ['a', 'b', 'c', 'density', 'shape_enc',
                'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio']

X = df[feature_cols].values
y = df['velocity_mean'].values

# En iyi parametreler
best_rf_params = {
    'n_estimators': 300,
    'max_depth': 25,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

best_nn_params = {
    'hidden_layer_sizes': (128, 64, 32),
    'alpha': 0.001,
    'random_state': 42,
    'max_iter': 2000,
    'early_stopping': True
}

ensemble_weights = {
    'rf_weight': 0.7,
    'nn_weight': 0.3
}

# Final CV skorunu hesapla
print("="*60)
print("FINAL MODEL CV SKORU")
print("="*60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # RF
    rf = RandomForestRegressor(**best_rf_params)
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_val)

    # NN
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    nn = MLPRegressor(**best_nn_params)
    nn.fit(X_tr_s, y_tr)
    nn_pred = nn.predict(X_val_s)

    # Ensemble
    ens_pred = 0.7 * rf_pred + 0.3 * nn_pred
    r2 = 1 - np.sum((y_val - ens_pred)**2) / np.sum((y_val - y_val.mean())**2)
    scores.append(r2)

print(f"Ensemble CV R2: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

# Final model egitimi (tum veri ile)
print("\n" + "="*60)
print("FINAL MODEL EGITIMI")
print("="*60)

# Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# RF
print("Random Forest egitiliyor...")
rf_final = RandomForestRegressor(**best_rf_params)
rf_final.fit(X, y)
print(f"  RF egitildi. n_estimators={rf_final.n_estimators}")

# NN
print("Neural Network egitiliyor...")
nn_final = MLPRegressor(**best_nn_params)
nn_final.fit(X_scaled, y)
print(f"  NN egitildi. layers={nn_final.hidden_layer_sizes}")

# Kayit klasoru
save_dir = 'ml_models/best_model'
os.makedirs(save_dir, exist_ok=True)

# Modelleri kaydet
print("\n" + "="*60)
print("MODELLER KAYDEDILIYOR")
print("="*60)

joblib.dump(rf_final, f'{save_dir}/rf_model.joblib')
print(f"  {save_dir}/rf_model.joblib")

joblib.dump(nn_final, f'{save_dir}/nn_model.joblib')
print(f"  {save_dir}/nn_model.joblib")

joblib.dump(scaler, f'{save_dir}/scaler.joblib')
print(f"  {save_dir}/scaler.joblib")

# Parametreleri kaydet
params = {
    'model_info': {
        'name': 'Ensemble (RF + NN)',
        'cv_r2': float(np.mean(scores)),
        'cv_r2_std': float(np.std(scores)),
        'n_samples': len(df),
        'n_particles': len(df),
        'n_features': len(feature_cols)
    },
    'feature_columns': feature_cols,
    'rf_params': {k: v for k, v in best_rf_params.items() if k != 'n_jobs'},
    'nn_params': {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_nn_params.items()},
    'ensemble_weights': ensemble_weights
}

with open(f'{save_dir}/model_params.json', 'w', encoding='utf-8') as f:
    json.dump(params, f, indent=2, ensure_ascii=False)
print(f"  {save_dir}/model_params.json")

# Feature importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

importances = rf_final.feature_importances_
for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")

# Kullanim ornegi
print("\n" + "="*60)
print("KULLANIM ORNEGI")
print("="*60)
print("""
# Model yukleme ve tahmin
import joblib
import numpy as np

rf = joblib.load('ml_models/best_model/rf_model.joblib')
nn = joblib.load('ml_models/best_model/nn_model.joblib')
scaler = joblib.load('ml_models/best_model/scaler.joblib')

# Ornek veri: [a, b, c, density, shape_enc, volume, surface_area, aspect_ratio, vol_surf_ratio]
X_new = np.array([[5.0, 3.0, 0, 1150, 0, 35.3, 70.7, 1.67, 0.5]])

# Tahmin
rf_pred = rf.predict(X_new)
nn_pred = nn.predict(scaler.transform(X_new))
ensemble_pred = 0.7 * rf_pred + 0.3 * nn_pred

print(f"Tahmin edilen hiz: {ensemble_pred[0]:.2f} cm/s")
""")

print("\n" + "="*60)
print("TAMAMLANDI")
print("="*60)
print(f"\nModel klasoru: {save_dir}/")
print(f"CV R2: {np.mean(scores):.4f}")
