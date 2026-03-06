"""
ML Model Comparison: Baseline vs 3D-Enhanced Features

Compares 3 model configurations using 5-fold CV:
  1. Baseline RF: 9 original features, RF only
  2. Baseline Ensemble: 9 features, RF(70%) + NN(30%)
  3. 3D Enhanced Ensemble: 9 + 3 new 3D trajectory features, RF(70%) + NN(30%)

3D features: drift_normalized, tortuosity_pct, drift_gradient
"""

import sys
import os
import warnings

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# --- Paths ---
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DATA = os.path.join(PROJECT_DIR, 'data', 'training_data_particle_avg.csv')
RESULTS_3D = os.path.join(PROJECT_DIR, 'analysis_3d', 'results_3d.csv')
OUTPUT_CSV = os.path.join(PROJECT_DIR, 'analysis_3d', 'ml_comparison.csv')

# --- Model Parameters (from best_model/model_params.json) ---
RF_PARAMS = dict(n_estimators=100, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
NN_PARAMS = dict(hidden_layer_sizes=(128,), alpha=0.01, max_iter=2000,
                 early_stopping=True, random_state=42)
RF_WEIGHT, NN_WEIGHT = 0.7, 0.3

# --- Feature columns ---
BASELINE_FEATURES = ['a', 'b', 'c', 'density', 'shape_enc',
                     'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio']
NEW_3D_FEATURES = ['drift_normalized', 'tortuosity_pct', 'drift_gradient']
ENHANCED_FEATURES = BASELINE_FEATURES + NEW_3D_FEATURES


def load_and_merge():
    """Load baseline training data and 3D results, merge on category+code."""
    df_train = pd.read_csv(TRAINING_DATA)
    df_3d = pd.read_csv(RESULTS_3D)

    print(f"Baseline training data: {len(df_train)} particles")
    print(f"3D results: {len(df_3d)} records ({df_3d['code'].nunique()} unique codes)")

    # Average 3D metrics per particle (across repeats)
    avg_3d = df_3d.groupby(['category', 'code']).agg({
        'drift_normalized': 'mean',
        'tortuosity_pct': 'mean',
        'drift_gradient': 'mean'
    }).reset_index()
    print(f"3D averaged per particle: {len(avg_3d)} unique (category, code) pairs")

    # Inner join: only particles with both training data and 3D results
    merged = pd.merge(df_train, avg_3d, on=['category', 'code'], how='inner')
    print(f"Merged (inner join): {len(merged)} particles")

    return df_train, merged


def run_cv(X, y, feature_names, model_type='rf_only'):
    """
    Run 5-fold CV for a given model configuration.

    model_type: 'rf_only' | 'ensemble'
    Returns: dict with cv_r2, cv_std, train_r2
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_scores = []
    fold_train_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # Random Forest (uses unscaled data)
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_tr, y_tr)
        rf_pred_val = rf.predict(X_val)
        rf_pred_tr = rf.predict(X_tr)

        if model_type == 'rf_only':
            val_pred = rf_pred_val
            tr_pred = rf_pred_tr
        else:
            # Neural Network (uses scaled data)
            nn = MLPRegressor(**NN_PARAMS)
            nn.fit(X_tr_s, y_tr)
            nn_pred_val = nn.predict(X_val_s)
            nn_pred_tr = nn.predict(X_tr_s)

            # Ensemble
            val_pred = RF_WEIGHT * rf_pred_val + NN_WEIGHT * nn_pred_val
            tr_pred = RF_WEIGHT * rf_pred_tr + NN_WEIGHT * nn_pred_tr

        fold_r2 = r2_score(y_val, val_pred)
        fold_train_r2 = r2_score(y_tr, tr_pred)
        fold_scores.append(fold_r2)
        fold_train_scores.append(fold_train_r2)

    return {
        'cv_r2': np.mean(fold_scores),
        'cv_std': np.std(fold_scores),
        'train_r2': np.mean(fold_train_scores),
        'fold_scores': fold_scores
    }


def get_feature_importance(X, y, feature_names):
    """Train RF on full data and return feature importances."""
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X, y)
    importances = rf.feature_importances_
    return sorted(zip(feature_names, importances), key=lambda x: -x[1])


def main():
    print("=" * 70)
    print("ML MODEL COMPARISON: Baseline vs 3D-Enhanced Features")
    print("=" * 70)
    print()

    # --- Load data ---
    df_full, df_merged = load_and_merge()
    print()

    # --- Prepare datasets ---
    # For Baseline models: use merged subset (fair comparison on same particles)
    X_baseline = df_merged[BASELINE_FEATURES].values
    y_baseline = df_merged['velocity_mean'].values

    # For 3D Enhanced: same particles, extra features
    X_enhanced = df_merged[ENHANCED_FEATURES].values
    y_enhanced = df_merged['velocity_mean'].values

    # Also run baseline on FULL dataset (240) for reference
    X_full = df_full[BASELINE_FEATURES].values
    y_full = df_full['velocity_mean'].values

    n_merged = len(df_merged)
    n_full = len(df_full)

    print(f"Full dataset: {n_full} particles, {len(BASELINE_FEATURES)} features")
    print(f"Merged subset: {n_merged} particles, {len(BASELINE_FEATURES)} baseline / {len(ENHANCED_FEATURES)} enhanced features")
    print()

    # --- Run Cross-Validation ---
    print("-" * 70)
    print("Running 5-fold CV...")
    print("-" * 70)
    print()

    # Config 1: Baseline RF only (on merged subset for fair comparison)
    print("[1/4] Baseline RF (merged subset, n={})...".format(n_merged))
    res_baseline_rf = run_cv(X_baseline, y_baseline, BASELINE_FEATURES, model_type='rf_only')

    # Config 2: Baseline Ensemble (on merged subset)
    print("[2/4] Baseline Ensemble (merged subset, n={})...".format(n_merged))
    res_baseline_ens = run_cv(X_baseline, y_baseline, BASELINE_FEATURES, model_type='ensemble')

    # Config 3: 3D Enhanced Ensemble
    print("[3/4] 3D Enhanced Ensemble (merged subset, n={})...".format(n_merged))
    res_enhanced = run_cv(X_enhanced, y_enhanced, ENHANCED_FEATURES, model_type='ensemble')

    # Reference: Baseline Ensemble on full 240 particles
    print("[4/4] Baseline Ensemble (full dataset, n={})...".format(n_full))
    res_full = run_cv(X_full, y_full, BASELINE_FEATURES, model_type='ensemble')

    print()

    # --- Comparison Table ---
    print("=" * 70)
    print("RESULTS: 5-Fold Cross-Validation Comparison")
    print("=" * 70)
    print()

    header = f"{'Model':<40} {'N':>4} {'CV R2':>8} {'Std':>8} {'Train R2':>9} {'Gap':>6}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Baseline RF (merged)", n_merged, res_baseline_rf),
        ("Baseline Ensemble (merged)", n_merged, res_baseline_ens),
        ("3D Enhanced Ensemble (merged)", n_merged, res_enhanced),
        ("Baseline Ensemble (full 240)", n_full, res_full),
    ]

    for name, n, res in rows:
        gap = res['train_r2'] - res['cv_r2']
        print(f"{name:<40} {n:>4} {res['cv_r2']:>8.4f} {res['cv_std']:>8.4f} {res['train_r2']:>9.4f} {gap:>6.3f}")

    print()

    # Delta analysis
    delta_rf_to_ens = res_baseline_ens['cv_r2'] - res_baseline_rf['cv_r2']
    delta_ens_to_3d = res_enhanced['cv_r2'] - res_baseline_ens['cv_r2']
    delta_total = res_enhanced['cv_r2'] - res_baseline_rf['cv_r2']

    print("Delta Analysis:")
    print(f"  RF -> Ensemble:        {delta_rf_to_ens:+.4f}")
    print(f"  Ensemble -> 3D:        {delta_ens_to_3d:+.4f}")
    print(f"  RF -> 3D Enhanced:     {delta_total:+.4f}")
    print()

    # --- Per-fold details ---
    print("-" * 70)
    print("Per-Fold CV R2 Scores")
    print("-" * 70)
    print(f"{'Fold':<6} {'Baseline RF':>12} {'Baseline Ens':>13} {'3D Enhanced':>13}")
    for i in range(5):
        print(f"  {i+1:<4} {res_baseline_rf['fold_scores'][i]:>12.4f} "
              f"{res_baseline_ens['fold_scores'][i]:>13.4f} "
              f"{res_enhanced['fold_scores'][i]:>13.4f}")
    print()

    # --- Feature Importance for 3D Enhanced ---
    print("=" * 70)
    print("FEATURE IMPORTANCE: 3D Enhanced Model (RF)")
    print("=" * 70)
    print()

    fi = get_feature_importance(X_enhanced, y_enhanced, ENHANCED_FEATURES)
    print(f"{'Rank':<6} {'Feature':<25} {'Importance':>10}")
    print("-" * 43)
    for rank, (feat, imp) in enumerate(fi, 1):
        marker = " <-- 3D" if feat in NEW_3D_FEATURES else ""
        print(f"{rank:<6} {feat:<25} {imp:>10.4f}{marker}")

    print()

    # Also show baseline feature importance for comparison
    print("-" * 70)
    print("FEATURE IMPORTANCE: Baseline Model (RF, same subset)")
    print("-" * 70)
    print()
    fi_base = get_feature_importance(X_baseline, y_baseline, BASELINE_FEATURES)
    print(f"{'Rank':<6} {'Feature':<25} {'Importance':>10}")
    print("-" * 43)
    for rank, (feat, imp) in enumerate(fi_base, 1):
        print(f"{rank:<6} {feat:<25} {imp:>10.4f}")

    print()

    # --- Save comparison CSV ---
    comparison_data = []
    for name, n, res in rows:
        gap = res['train_r2'] - res['cv_r2']
        comparison_data.append({
            'model': name,
            'n_particles': n,
            'n_features': len(ENHANCED_FEATURES) if '3D' in name else len(BASELINE_FEATURES),
            'cv_r2': round(res['cv_r2'], 4),
            'cv_std': round(res['cv_std'], 4),
            'train_r2': round(res['train_r2'], 4),
            'overfit_gap': round(gap, 4)
        })

    df_comp = pd.DataFrame(comparison_data)
    df_comp.to_csv(OUTPUT_CSV, index=False)
    print(f"Comparison results saved to: {OUTPUT_CSV}")

    # --- Summary ---
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    best_name = rows[2][0]
    best_r2 = res_enhanced['cv_r2']
    base_r2 = res_baseline_ens['cv_r2']
    print(f"  Baseline Ensemble (n={n_merged}):  R2 = {base_r2:.4f}")
    print(f"  3D Enhanced Ensemble (n={n_merged}): R2 = {best_r2:.4f}")
    print(f"  Improvement from 3D features:   {delta_ens_to_3d:+.4f}")
    print()
    if delta_ens_to_3d > 0:
        print("  --> 3D trajectory features IMPROVE the model.")
    elif delta_ens_to_3d == 0:
        print("  --> 3D trajectory features have NO EFFECT on the model.")
    else:
        print("  --> 3D trajectory features do NOT improve the model (possible noise).")
    print()

    # Top 3D feature
    top_3d = [(f, i) for f, i in fi if f in NEW_3D_FEATURES]
    if top_3d:
        print(f"  Most important 3D feature: {top_3d[0][0]} (importance={top_3d[0][1]:.4f})")
    print()


if __name__ == '__main__':
    main()
