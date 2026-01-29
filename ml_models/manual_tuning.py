"""Manuel Hyperparameter Tuning - Daha kontrollü arama"""
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

# Veri
df = pd.read_csv('data/training_data_particle_avg.csv')
print(f"Veri: {len(df)} parcacik\n")

feature_cols = ['a', 'b', 'c', 'density', 'shape_enc',
                'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio']

X = df[feature_cols].values
y = df['velocity_mean'].values

# KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_rf(params):
    """RF parametrelerini degerlendir"""
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        rf = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        pred = rf.predict(X_val)
        r2 = 1 - np.sum((y_val - pred)**2) / np.sum((y_val - y_val.mean())**2)
        scores.append(r2)
    return np.mean(scores), np.std(scores)

def evaluate_nn(params):
    """NN parametrelerini degerlendir"""
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        nn = MLPRegressor(**params, random_state=42, max_iter=2000, early_stopping=True)
        nn.fit(X_tr_s, y_tr)
        pred = nn.predict(X_val_s)
        r2 = 1 - np.sum((y_val - pred)**2) / np.sum((y_val - y_val.mean())**2)
        scores.append(r2)
    return np.mean(scores), np.std(scores)

# ================================================================
print("="*60)
print("RANDOM FOREST TUNING")
print("="*60)

# RF param grid
rf_configs = [
    {'n_estimators': 100, 'max_depth': None},
    {'n_estimators': 200, 'max_depth': None},
    {'n_estimators': 200, 'max_depth': 20},
    {'n_estimators': 200, 'max_depth': 15},
    {'n_estimators': 300, 'max_depth': 20},
    {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5},
    {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 10},
    {'n_estimators': 200, 'max_depth': 20, 'min_samples_leaf': 2},
    {'n_estimators': 200, 'max_depth': 20, 'max_features': 'sqrt'},
    {'n_estimators': 200, 'max_depth': 20, 'max_features': 'log2'},
    {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 5},
    {'n_estimators': 300, 'max_depth': 25, 'min_samples_leaf': 2},
]

best_rf = None
best_rf_score = -999

for params in rf_configs:
    mean, std = evaluate_rf(params)
    marker = ""
    if mean > best_rf_score:
        best_rf_score = mean
        best_rf = params
        marker = " <-- best"
    print(f"  {params}: R2={mean:.4f} +/- {std:.4f}{marker}")

print(f"\nEn iyi RF: {best_rf}")
print(f"En iyi RF R2: {best_rf_score:.4f}")

# ================================================================
print("\n" + "="*60)
print("NEURAL NETWORK TUNING")
print("="*60)

nn_configs = [
    {'hidden_layer_sizes': (64, 32), 'alpha': 0.001},
    {'hidden_layer_sizes': (64, 32), 'alpha': 0.01},
    {'hidden_layer_sizes': (64, 32), 'alpha': 0.0001},
    {'hidden_layer_sizes': (128, 64), 'alpha': 0.001},
    {'hidden_layer_sizes': (128, 64, 32), 'alpha': 0.001},
    {'hidden_layer_sizes': (100,), 'alpha': 0.001},
    {'hidden_layer_sizes': (64, 32), 'alpha': 0.001, 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (64, 32), 'alpha': 0.001, 'learning_rate_init': 0.0001},
    {'hidden_layer_sizes': (64, 32), 'alpha': 0.001, 'activation': 'tanh'},
    {'hidden_layer_sizes': (32, 16), 'alpha': 0.01},
]

best_nn = None
best_nn_score = -999

for params in nn_configs:
    mean, std = evaluate_nn(params)
    marker = ""
    if mean > best_nn_score:
        best_nn_score = mean
        best_nn = params
        marker = " <-- best"
    print(f"  {params}: R2={mean:.4f} +/- {std:.4f}{marker}")

print(f"\nEn iyi NN: {best_nn}")
print(f"En iyi NN R2: {best_nn_score:.4f}")

# ================================================================
print("\n" + "="*60)
print("ENSEMBLE TUNING")
print("="*60)

# En iyi RF ve NN ile ensemble dene
for rf_w in [0.4, 0.5, 0.6, 0.7, 0.8]:
    nn_w = 1 - rf_w
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # RF
        rf = RandomForestRegressor(**best_rf, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        rf_pred = rf.predict(X_val)

        # NN
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        nn = MLPRegressor(**best_nn, random_state=42, max_iter=2000, early_stopping=True)
        nn.fit(X_tr_s, y_tr)
        nn_pred = nn.predict(X_val_s)

        # Ensemble
        ens_pred = rf_w * rf_pred + nn_w * nn_pred
        r2 = 1 - np.sum((y_val - ens_pred)**2) / np.sum((y_val - y_val.mean())**2)
        scores.append(r2)

    print(f"  RF {int(rf_w*100)}% + NN {int(nn_w*100)}%: R2={np.mean(scores):.4f} +/- {np.std(scores):.4f}")

# ================================================================
print("\n" + "="*60)
print("SONUC")
print("="*60)
print(f"En iyi RF: {best_rf_score:.4f}")
print(f"En iyi NN: {best_nn_score:.4f}")
