"""
Publication-quality figures for the 3D settling dynamics paper.

Generates:
    Figure 2 - Example 3D trajectories (one per shape)
    Figure 3 - Drift & tortuosity by shape (box plots)
    Figure 4 - Re-CD diagram (log-log scatter)
    Figure 5 - ML prediction (predicted vs measured + feature importance)

Usage
-----
    python analysis_3d/generate_figures.py
"""
import sys
import os
import json

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

RESULTS_CSV = os.path.join(_THIS_DIR, "results_3d.csv")
FIGURES_DIR = os.path.join(_THIS_DIR, "figures")
EXPERIMENTS_JSON = os.path.join(_PROJECT_ROOT, "processed_results", "experiments.json")
BASE_DIR = os.path.join(_PROJECT_ROOT, "processed_results")
TRAINING_CSV = os.path.join(_PROJECT_ROOT, "data", "training_data_particle_avg.csv")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "ml_models", "best_model")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
})

# Shape display names (shorter) and consistent colour palette
SHAPE_SHORT = {
    "Box Shape Prism": "Box",
    "Cylinder": "Cylinder",
    "Cube": "Cube",
    "Elliptic Cylinder": "Elliptic",
    "Half Cylinder": "Half Cyl.",
    "Wedge Shape Prism": "Wedge",
    "Sphere": "Sphere",
}

SHAPE_COLORS = {
    "Box Shape Prism": "#1f77b4",
    "Cylinder": "#ff7f0e",
    "Cube": "#2ca02c",
    "Elliptic Cylinder": "#d62728",
    "Half Cylinder": "#9467bd",
    "Wedge Shape Prism": "#8c564b",
    "Sphere": "#e377c2",
}


def _save(fig, name):
    """Save figure as both PNG and PDF."""
    png_path = os.path.join(FIGURES_DIR, f"{name}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{name}.pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Kaydedildi: {png_path}")
    print(f"  Kaydedildi: {pdf_path}")


# ===================================================================
# Figure 2: Example 3D Trajectories (one per shape)
# ===================================================================
def generate_figure2(df):
    """3D trajectory plot with one representative particle per shape."""
    from analysis_3d.reconstruct_3d import reconstruct_trajectory_3d

    print("\n--- Figure 2: 3D Trajectories ---")

    # Load experiments.json for path lookup
    with open(EXPERIMENTS_JSON, "r", encoding="utf-8") as f:
        exp_data = json.load(f)

    # Build lookup: (code, repeat, date, view) -> path
    path_lookup = {}
    for exp in exp_data["experiments"]:
        if exp["status"] == "success" and exp["view"] in ("MAK", "ANG"):
            key = (exp["code"], exp["repeat"], exp["date"], exp["view"])
            path_lookup[key] = exp["path"]

    # Pick one representative particle per shape (closest to median velocity)
    representatives = []
    for shape in sorted(df["shape_name"].unique()):
        sub = df[df["shape_name"] == shape].copy()
        median_v = sub["settling_velocity_cm_s"].median()
        idx = (sub["settling_velocity_cm_s"] - median_v).abs().idxmin()
        row = df.loc[idx]
        representatives.append(row)

    # Reconstruct each trajectory
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for row in representatives:
        code = row["code"]
        repeat = row["repeat"]
        date = row["date"]
        shape = row["shape_name"]

        mak_key = (code, repeat, date, "MAK")
        ang_key = (code, repeat, date, "ANG")

        if mak_key not in path_lookup or ang_key not in path_lookup:
            print(f"  UYARI: {code}/{repeat}/{date} icin path bulunamadi, atlaniyor")
            continue

        mak_csv = os.path.join(BASE_DIR, path_lookup[mak_key], "auto_tracking_results.csv")
        ang_csv = os.path.join(BASE_DIR, path_lookup[ang_key], "auto_tracking_results.csv")

        if not os.path.exists(mak_csv) or not os.path.exists(ang_csv):
            print(f"  UYARI: {code} CSV dosyasi bulunamadi, atlaniyor")
            continue

        try:
            traj = reconstruct_trajectory_3d(mak_csv, ang_csv)
            color = SHAPE_COLORS.get(shape, "#333333")
            label = SHAPE_SHORT.get(shape, shape)

            ax.plot(
                traj["x_cm"].values,
                traj["y_cm"].values,
                traj["z_cm"].values,
                color=color,
                label=f"{label} ({code})",
                linewidth=1.8,
                alpha=0.85,
            )
            # Mark start point
            ax.scatter(
                traj["x_cm"].iloc[0],
                traj["y_cm"].iloc[0],
                traj["z_cm"].iloc[0],
                color=color, marker="o", s=30, zorder=5,
            )
            print(f"  {shape} ({code}): {len(traj)} points OK")

        except Exception as e:
            print(f"  HATA: {code} - {e}")

    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.set_title("Figure 2: Representative 3D Settling Trajectories")

    # Invert z-axis so positive downward = bottom of plot
    ax.invert_zaxis()

    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.view_init(elev=20, azim=-60)

    _save(fig, "figure2_3d_trajectories")


# ===================================================================
# Figure 3: Drift & Tortuosity by Shape (box plots)
# ===================================================================
def generate_figure3(df):
    """Box plots: normalized drift, tortuosity, and drift gradient by shape."""
    print("\n--- Figure 3: Drift & Tortuosity Box Plots ---")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Sort shapes by median settling velocity for consistent ordering
    shape_order = (
        df.groupby("shape_name")["settling_velocity_cm_s"]
        .median()
        .sort_values()
        .index.tolist()
    )
    short_labels = [SHAPE_SHORT.get(s, s) for s in shape_order]

    # Prepare data lists for each shape (in order)
    def _box_data(column):
        return [df[df["shape_name"] == s][column].dropna().values for s in shape_order]

    # (a) Normalized drift (delta*)
    data_a = _box_data("drift_normalized")
    bp_a = axes[0].boxplot(
        data_a, tick_labels=short_labels, patch_artist=True, widths=0.6,
    )
    for patch, shape in zip(bp_a["boxes"], shape_order):
        patch.set_facecolor(SHAPE_COLORS.get(shape, "#cccccc"))
        patch.set_alpha(0.7)
    axes[0].set_ylabel(r"Normalized Drift $\delta^*$")
    axes[0].set_title("(a) Normalized Drift")
    axes[0].tick_params(axis="x", rotation=30)

    # (b) Tortuosity (%)
    data_b = _box_data("tortuosity_pct")
    bp_b = axes[1].boxplot(
        data_b, tick_labels=short_labels, patch_artist=True, widths=0.6,
    )
    for patch, shape in zip(bp_b["boxes"], shape_order):
        patch.set_facecolor(SHAPE_COLORS.get(shape, "#cccccc"))
        patch.set_alpha(0.7)
    axes[1].set_ylabel(r"Tortuosity $\phi$ (%)")
    axes[1].set_title("(b) Tortuosity")
    axes[1].tick_params(axis="x", rotation=30)

    # (c) Drift gradient (epsilon)
    data_c = _box_data("drift_gradient")
    bp_c = axes[2].boxplot(
        data_c, tick_labels=short_labels, patch_artist=True, widths=0.6,
    )
    for patch, shape in zip(bp_c["boxes"], shape_order):
        patch.set_facecolor(SHAPE_COLORS.get(shape, "#cccccc"))
        patch.set_alpha(0.7)
    axes[2].set_ylabel(r"Drift Gradient $\varepsilon$")
    axes[2].set_title("(c) Drift Gradient")
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle("Figure 3: Lateral Drift and Path Characteristics by Shape", y=1.02)
    fig.tight_layout()
    _save(fig, "figure3_drift_tortuosity")


# ===================================================================
# Figure 4: Re-CD Diagram
# ===================================================================
def generate_figure4(df):
    """Log-log scatter of Re vs CD, colored by shape, with standard sphere curve."""
    print("\n--- Figure 4: Re-CD Diagram ---")

    # Physical constants
    rho_f = 998.0        # kg/m^3  (water)
    g = 981.0            # cm/s^2
    mu = 0.01            # g/(cm*s) = poise  (water at ~20C)
    # For Re: Re = rho_f * w * d_eq / mu
    #   rho_f in g/cm^3 = 0.998, mu in g/(cm*s) = 0.01, w in cm/s, d_eq in cm
    rho_f_cgs = 0.998    # g/cm^3

    # Calculate CD for each experiment
    # CD = (4/3) * (rho_p - rho_f) * g * d_eq / (rho_f * w^2)
    #   rho_p in kg/m^3 -> g/cm^3 = /1000
    #   rho_f = 0.998 g/cm^3
    #   g = 981 cm/s^2
    #   d_eq in cm, w in cm/s
    df_plot = df.copy()
    rho_p_cgs = df_plot["density_kg_m3"].values / 1000.0  # g/cm^3
    d_eq = df_plot["d_eq_cm"].values
    w = df_plot["settling_velocity_cm_s"].values

    # Avoid division by zero
    valid = w > 0.01
    CD = np.full(len(df_plot), np.nan)
    CD[valid] = (4.0 / 3.0) * (rho_p_cgs[valid] - rho_f_cgs) * g * d_eq[valid] / (rho_f_cgs * w[valid] ** 2)

    Re = df_plot["reynolds_number"].values

    df_plot["CD"] = CD
    df_plot["Re"] = Re

    # Filter valid data
    mask = np.isfinite(CD) & (CD > 0) & (Re > 0)
    df_valid = df_plot[mask].copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter by shape
    for shape in sorted(df_valid["shape_name"].unique()):
        sub = df_valid[df_valid["shape_name"] == shape]
        color = SHAPE_COLORS.get(shape, "#333333")
        label = SHAPE_SHORT.get(shape, shape)
        ax.scatter(
            sub["Re"], sub["CD"],
            c=color, label=label, s=35, alpha=0.7, edgecolors="k", linewidths=0.3,
        )

    # Standard drag curve for a sphere: CD = 24/Re + 6/(1+sqrt(Re)) + 0.4
    Re_curve = np.logspace(np.log10(max(1, Re[mask].min() * 0.5)),
                           np.log10(Re[mask].max() * 2), 200)
    CD_sphere = 24.0 / Re_curve + 6.0 / (1.0 + np.sqrt(Re_curve)) + 0.4
    ax.plot(Re_curve, CD_sphere, "k--", linewidth=1.5, label="Sphere (standard)", alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Reynolds Number $Re_p$")
    ax.set_ylabel(r"Drag Coefficient $C_D$")
    ax.set_title("Figure 4: Drag Coefficient vs Reynolds Number")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)

    _save(fig, "figure4_re_cd")


# ===================================================================
# Figure 5: ML Prediction (predicted vs measured + feature importance)
# ===================================================================
def generate_figure5():
    """Predicted vs measured scatter and feature importance bar chart."""
    print("\n--- Figure 5: ML Prediction ---")

    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    # Load model components
    params_path = os.path.join(MODEL_DIR, "model_params.json")
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    feature_cols = params["feature_columns"]
    rf_weight = params["ensemble_weights"]["rf_weight"]
    nn_weight = params["ensemble_weights"]["nn_weight"]

    # Load training data first (needed for fallback retraining)
    train_df = pd.read_csv(TRAINING_CSV)
    X = train_df[feature_cols].values
    y = train_df["velocity_mean"].values

    # Try to load saved models; retrain if pickle version mismatch
    try:
        rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
    except Exception as e:
        print(f"  RF model yuklenemedi ({e}), yeniden egitiliyor...")
        rf_params = params["rf_params"]
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X, y)

    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    except Exception as e:
        print(f"  Scaler yuklenemedi ({e}), yeniden olusturuluyor...")
        scaler = StandardScaler()
        scaler.fit(X)

    try:
        nn_model = joblib.load(os.path.join(MODEL_DIR, "nn_model.joblib"))
    except Exception as e:
        print(f"  NN model yuklenemedi ({e}), yeniden egitiliyor...")
        nn_params = params["nn_params"]
        # hidden_layer_sizes needs to be tuple
        if "hidden_layer_sizes" in nn_params:
            nn_params["hidden_layer_sizes"] = tuple(nn_params["hidden_layer_sizes"])
        nn_model = MLPRegressor(**nn_params)
        X_scaled_train = scaler.transform(X)
        nn_model.fit(X_scaled_train, y)

    # Scale features for NN
    X_scaled = scaler.transform(X)

    # Predictions
    pred_rf = rf_model.predict(X)
    pred_nn = nn_model.predict(X_scaled)
    pred_ensemble = rf_weight * pred_rf + nn_weight * pred_nn

    # R^2
    ss_res = np.sum((y - pred_ensemble) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # RMSE
    rmse = np.sqrt(np.mean((y - pred_ensemble) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left: predicted vs measured ---
    ax1 = axes[0]
    # Color by shape
    for shape in sorted(train_df["shape_name"].unique()):
        mask = train_df["shape_name"].values == shape
        color = SHAPE_COLORS.get(shape, "#333333")
        label = SHAPE_SHORT.get(shape, shape)
        ax1.scatter(
            y[mask], pred_ensemble[mask],
            c=color, label=label, s=30, alpha=0.7,
            edgecolors="k", linewidths=0.3,
        )

    # 1:1 line
    vmin = min(y.min(), pred_ensemble.min()) * 0.9
    vmax = max(y.max(), pred_ensemble.max()) * 1.1
    ax1.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, alpha=0.7, label="1:1 line")

    # +/- 20% bands
    x_band = np.linspace(vmin, vmax, 100)
    ax1.fill_between(x_band, x_band * 0.8, x_band * 1.2,
                     alpha=0.08, color="gray", label=r"$\pm$20%")

    ax1.set_xlabel("Measured Velocity (cm/s)")
    ax1.set_ylabel("Predicted Velocity (cm/s)")
    ax1.set_title("(a) Predicted vs Measured")
    ax1.set_xlim(vmin, vmax)
    ax1.set_ylim(vmin, vmax)
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Annotation
    ax1.text(
        0.95, 0.05,
        f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} cm/s\nn = {len(y)}",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    # --- Right: feature importance (RF) ---
    ax2 = axes[1]
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)
    feature_labels = [feature_cols[i] for i in sorted_idx]

    # Clean up feature names for display
    display_names = {
        "a": "a (mm)",
        "b": "b (mm)",
        "c": "c (mm)",
        "density": "Density",
        "shape_enc": "Shape",
        "volume": "Volume",
        "surface_area": "Surface Area",
        "aspect_ratio": "Aspect Ratio",
        "vol_surf_ratio": "Vol/Surf Ratio",
    }
    feature_labels_display = [display_names.get(f, f) for f in feature_labels]

    bars = ax2.barh(
        range(len(sorted_idx)),
        importances[sorted_idx],
        color="#4c72b0",
        edgecolor="k",
        linewidth=0.5,
        alpha=0.8,
    )
    ax2.set_yticks(range(len(sorted_idx)))
    ax2.set_yticklabels(feature_labels_display)
    ax2.set_xlabel("Feature Importance (MDI)")
    ax2.set_title("(b) Random Forest Feature Importance")

    fig.suptitle("Figure 5: ML Ensemble Model Performance", y=1.01)
    fig.tight_layout()
    _save(fig, "figure5_ml_prediction")

    print(f"  R^2 = {r2:.4f}, RMSE = {rmse:.2f} cm/s")


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("PUBLICATION FIGURES - 3D Settling Dynamics Paper")
    print("=" * 60)

    # Load main results
    if not os.path.exists(RESULTS_CSV):
        print(f"HATA: {RESULTS_CSV} bulunamadi!")
        sys.exit(1)

    df = pd.read_csv(RESULTS_CSV)
    print(f"results_3d.csv yuklendi: {len(df)} kayit, {df['shape_name'].nunique()} sekil")

    # Generate all figures
    generate_figure2(df)
    generate_figure3(df)
    generate_figure4(df)
    generate_figure5()

    print("\n" + "=" * 60)
    print("Tum figurler olusturuldu!")
    print(f"Konum: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
