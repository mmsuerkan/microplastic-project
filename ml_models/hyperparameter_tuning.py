"""Deney 9: Hyperparameter Tuning (RF + NN)"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
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

# Features
feature_cols = ['a', 'b', 'c', 'density', 'shape_enc',
                'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio']

X = df[feature_cols].values
y = df['velocity_mean'].values

# Scaler for NN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================================================
# 1. RANDOM FOREST TUNING
# ================================================================
print("="*70)
print("1. RANDOM FOREST HYPERPARAMETER TUNING")
print("="*70)

rf_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}

print(f"\nAranacak parametre kombinasyonu: {np.prod([len(v) for v in rf_param_grid.values()])}")
print("RandomizedSearchCV ile 50 kombinasyon denenecek...")

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_search = RandomizedSearchCV(
    rf, rf_param_grid, n_iter=50, cv=5, scoring='r2',
    random_state=42, n_jobs=-1, verbose=0
)
rf_search.fit(X, y)

print(f"\nEn iyi RF parametreleri:")
for param, value in rf_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nEn iyi RF CV R²: {rf_search.best_score_:.4f}")

# Top 5 sonuç
print("\n--- Top 5 RF Kombinasyonu ---")
results_df = pd.DataFrame(rf_search.cv_results_)
top5 = results_df.nsmallest(5, 'rank_test_score')[['params', 'mean_test_score', 'std_test_score']]
for i, row in top5.iterrows():
    print(f"  R²={row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")

# ================================================================
# 2. NEURAL NETWORK TUNING
# ================================================================
print("\n" + "="*70)
print("2. NEURAL NETWORK HYPERPARAMETER TUNING")
print("="*70)

nn_param_grid = {
    'hidden_layer_sizes': [(32,), (64,), (128,), (64, 32), (128, 64), (64, 32, 16)],
    'learning_rate_init': [0.001, 0.01, 0.05],
    'alpha': [0.0001, 0.001, 0.01],
    'activation': ['relu', 'tanh'],
}

print(f"\nAranacak parametre kombinasyonu: {np.prod([len(v) for v in nn_param_grid.values()])}")
print("GridSearchCV ile tüm kombinasyonlar denenecek...")

nn = MLPRegressor(max_iter=1000, random_state=42, early_stopping=True)
nn_search = GridSearchCV(
    nn, nn_param_grid, cv=5, scoring='r2',
    n_jobs=-1, verbose=0
)
nn_search.fit(X_scaled, y)

print(f"\nEn iyi NN parametreleri:")
for param, value in nn_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nEn iyi NN CV R²: {nn_search.best_score_:.4f}")

# Top 5 sonuç
print("\n--- Top 5 NN Kombinasyonu ---")
results_df = pd.DataFrame(nn_search.cv_results_)
top5 = results_df.nsmallest(5, 'rank_test_score')[['params', 'mean_test_score', 'std_test_score']]
for i, row in top5.iterrows():
    print(f"  R²={row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")

# ================================================================
# 3. TUNED ENSEMBLE
# ================================================================
print("\n" + "="*70)
print("3. TUNED ENSEMBLE (En iyi RF + En iyi NN)")
print("="*70)

# En iyi modelleri al
best_rf = rf_search.best_estimator_
best_nn = nn_search.best_estimator_

# Farklı ensemble ağırlıkları dene
kf = KFold(n_splits=5, shuffle=True, random_state=42)
weight_options = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]

print("\n--- Ensemble Ağırlık Denemeleri ---")
best_ensemble_score = 0
best_weights = None

for rf_w, nn_w in weight_options:
    ens_scores = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # Clone and fit
        rf_temp = RandomForestRegressor(**best_rf.get_params())
        rf_temp.fit(X_tr, y_tr)
        rf_pred = rf_temp.predict(X_val)

        nn_temp = MLPRegressor(**best_nn.get_params())
        nn_temp.fit(X_tr_s, y_tr)
        nn_pred = nn_temp.predict(X_val_s)

        # Ensemble
        ens_pred = rf_w * rf_pred + nn_w * nn_pred
        ens_r2 = 1 - np.sum((y_val - ens_pred)**2) / np.sum((y_val - y_val.mean())**2)
        ens_scores.append(ens_r2)

    mean_score = np.mean(ens_scores)
    print(f"  RF {rf_w:.0%} + NN {nn_w:.0%}: R² = {mean_score:.4f} ± {np.std(ens_scores):.4f}")

    if mean_score > best_ensemble_score:
        best_ensemble_score = mean_score
        best_weights = (rf_w, nn_w)

print(f"\n✓ En iyi ensemble: RF {best_weights[0]:.0%} + NN {best_weights[1]:.0%}")
print(f"  CV R²: {best_ensemble_score:.4f}")

# ================================================================
# 4. SONUÇ KARŞILAŞTIRMASI
# ================================================================
print("\n" + "="*70)
print("4. SONUÇ KARŞILAŞTIRMASI")
print("="*70)

print(f"\n{'Model':<30} {'CV R²':<20}")
print("-" * 50)
print(f"{'Baseline RF':<30} {'0.8072 ± 0.0894':<20}")
print(f"{'Baseline NN':<30} {'0.8304 ± 0.0490':<20}")
print(f"{'Baseline Ensemble':<30} {'0.8336 ± 0.0609':<20}")
print("-" * 50)
print(f"{'Tuned RF':<30} {rf_search.best_score_:.4f}")
print(f"{'Tuned NN':<30} {nn_search.best_score_:.4f}")
print(f"{'Tuned Ensemble':<30} {best_ensemble_score:.4f}")

improvement = best_ensemble_score - 0.8336
print(f"\n{'İyileşme:':<30} {improvement:+.4f} ({improvement/0.8336*100:+.1f}%)")

# ================================================================
# 5. EN İYİ PARAMETRELERİ KAYDET
# ================================================================
print("\n" + "="*70)
print("5. EN İYİ PARAMETRELER")
print("="*70)

print("\n--- Random Forest ---")
print(rf_search.best_params_)

print("\n--- Neural Network ---")
print(nn_search.best_params_)

print(f"\n--- Ensemble Weights ---")
print(f"RF: {best_weights[0]}, NN: {best_weights[1]}")

# Parametreleri dosyaya kaydet
import json
best_params = {
    'rf': rf_search.best_params_,
    'nn': nn_search.best_params_,
    'ensemble_weights': {'rf': best_weights[0], 'nn': best_weights[1]},
    'scores': {
        'rf_cv_r2': float(rf_search.best_score_),
        'nn_cv_r2': float(nn_search.best_score_),
        'ensemble_cv_r2': float(best_ensemble_score),
    }
}

with open('ml_models/best_hyperparameters.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print("\nKaydedildi: ml_models/best_hyperparameters.json")
