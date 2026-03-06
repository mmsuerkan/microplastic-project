"""
Export summary tables for the 3D settling dynamics paper.

Generates three tables:
  Table 1: Particle Properties Summary (by shape)
  Table 2: 3D Settling Metrics by Shape
  Table 3: Comparison with Lofty et al. (2026)

All tables are saved as CSV in analysis_3d/tables/ and printed to console.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_3D = os.path.join(PROJECT_DIR, "analysis_3d", "results_3d.csv")
TRAINING_DATA = os.path.join(PROJECT_DIR, "data", "training_data_particle_avg.csv")
ML_COMPARISON = os.path.join(PROJECT_DIR, "analysis_3d", "ml_comparison.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "analysis_3d", "tables")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fmt(mean: float, std: float, decimals: int = 2) -> str:
    """Format as 'mean +/- std' with given decimal places."""
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def print_table(title: str, df: pd.DataFrame) -> None:
    """Print a DataFrame with a title, nicely formatted."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(df.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(RESULTS_3D)
df_train = pd.read_csv(TRAINING_DATA)

# Shape display order
SHAPE_ORDER = [
    "Box Shape Prism",
    "Cube",
    "Cylinder",
    "Elliptic Cylinder",
    "Half Cylinder",
    "Wedge Shape Prism",
]

# ===========================================================================
# TABLE 1: Particle Properties Summary
# ===========================================================================
# We want per-particle properties (not per-experiment), so we take unique
# particles by (code, shape_name).  Dimensions and density are the same
# across repeats for the same particle.
particles = df.drop_duplicates(subset=["code"]).copy()

rows_t1 = []
for shape in SHAPE_ORDER:
    sub = particles[particles["shape_name"] == shape]
    n = len(sub)
    rows_t1.append({
        "Shape": shape,
        "N": n,
        "a (mm)": fmt(sub["a_mm"].mean(), sub["a_mm"].std()),
        "b (mm)": fmt(sub["b_mm"].mean(), sub["b_mm"].std()),
        "density (kg/m3)": fmt(sub["density_kg_m3"].mean(), sub["density_kg_m3"].std()),
        "d_eq (mm)": fmt(sub["d_eq_cm"].mean() * 10, sub["d_eq_cm"].std() * 10),
    })

# Add totals row
all_p = particles
rows_t1.append({
    "Shape": "ALL",
    "N": len(all_p),
    "a (mm)": fmt(all_p["a_mm"].mean(), all_p["a_mm"].std()),
    "b (mm)": fmt(all_p["b_mm"].mean(), all_p["b_mm"].std()),
    "density (kg/m3)": fmt(all_p["density_kg_m3"].mean(), all_p["density_kg_m3"].std()),
    "d_eq (mm)": fmt(all_p["d_eq_cm"].mean() * 10, all_p["d_eq_cm"].std() * 10),
})

table1 = pd.DataFrame(rows_t1)
table1.to_csv(os.path.join(OUTPUT_DIR, "table1_particle_properties.csv"), index=False)
print_table("Table 1: Particle Properties Summary", table1)

# ===========================================================================
# TABLE 2: 3D Settling Metrics by Shape
# ===========================================================================
# First average per particle (code), then summarise per shape.
# This avoids over-representing particles with more repeats.
per_particle = df.groupby(["code", "shape_name"]).agg(
    w_mean=("settling_velocity_cm_s", "mean"),
    V_h_mean=("horizontal_drift_velocity_cm_s", "mean"),
    drift_norm_mean=("drift_normalized", "mean"),
    tortuosity_mean=("tortuosity_pct", "mean"),
    drift_grad_mean=("drift_gradient", "mean"),
    Re_mean=("reynolds_number", "mean"),
).reset_index()

rows_t2 = []
for shape in SHAPE_ORDER:
    sub = per_particle[per_particle["shape_name"] == shape]
    n = len(sub)
    rows_t2.append({
        "Shape": shape,
        "N": n,
        "w (cm/s)": fmt(sub["w_mean"].mean(), sub["w_mean"].std()),
        "V_h (cm/s)": fmt(sub["V_h_mean"].mean(), sub["V_h_mean"].std()),
        "delta* (norm. drift)": fmt(sub["drift_norm_mean"].mean(), sub["drift_norm_mean"].std()),
        "phi (tortuosity %)": fmt(sub["tortuosity_mean"].mean(), sub["tortuosity_mean"].std()),
        "epsilon (drift grad.)": fmt(sub["drift_grad_mean"].mean(), sub["drift_grad_mean"].std(), 4),
        "Re_p": fmt(sub["Re_mean"].mean(), sub["Re_mean"].std()),
    })

# Add totals row
all_pp = per_particle
rows_t2.append({
    "Shape": "ALL",
    "N": len(all_pp),
    "w (cm/s)": fmt(all_pp["w_mean"].mean(), all_pp["w_mean"].std()),
    "V_h (cm/s)": fmt(all_pp["V_h_mean"].mean(), all_pp["V_h_mean"].std()),
    "delta* (norm. drift)": fmt(all_pp["drift_norm_mean"].mean(), all_pp["drift_norm_mean"].std()),
    "phi (tortuosity %)": fmt(all_pp["tortuosity_mean"].mean(), all_pp["tortuosity_mean"].std()),
    "epsilon (drift grad.)": fmt(all_pp["drift_grad_mean"].mean(), all_pp["drift_grad_mean"].std(), 4),
    "Re_p": fmt(all_pp["Re_mean"].mean(), all_pp["Re_mean"].std()),
})

table2 = pd.DataFrame(rows_t2)
table2.to_csv(os.path.join(OUTPUT_DIR, "table2_settling_metrics.csv"), index=False)
print_table("Table 2: 3D Settling Metrics by Shape", table2)

# ===========================================================================
# TABLE 3: Comparison with Lofty et al. (2026)
# ===========================================================================
# Calculate ranges from our data
d_eq_mm_min = df["d_eq_cm"].min() * 10
d_eq_mm_max = df["d_eq_cm"].max() * 10
re_min = df["reynolds_number"].min()
re_max = df["reynolds_number"].max()
n_particles_ours = particles["code"].nunique()

# Load ML best result
ml_df = pd.read_csv(ML_COMPARISON)
best_ml = ml_df.loc[ml_df["cv_r2"].idxmax()]
ml_r2 = best_ml["cv_r2"]

rows_t3 = [
    {"Metric": "Number of particles",
     "Lofty et al. (2026)": "127",
     "This Study": str(n_particles_ours)},
    {"Metric": "Camera setup",
     "Lofty et al. (2026)": "2 cameras, 90 deg, 60 fps",
     "This Study": "2 cameras, 90 deg, 50 fps"},
    {"Metric": "Column geometry",
     "Lofty et al. (2026)": "Rectangular 20x20x70 cm",
     "This Study": "Cylindrical, dia 30 cm"},
    {"Metric": "Particle density",
     "Lofty et al. (2026)": "Not measured",
     "This Study": f"Known ({particles['density_kg_m3'].min():.0f}-{particles['density_kg_m3'].max():.0f} kg/m3)"},
    {"Metric": "Particle shapes",
     "Lofty et al. (2026)": "4 types (Zingg classification)",
     "This Study": f"{len(SHAPE_ORDER)} types (geometric)"},
    {"Metric": "Repeats per particle",
     "Lofty et al. (2026)": "Mostly 1",
     "This Study": "Up to 3"},
    {"Metric": "d_eq range (mm)",
     "Lofty et al. (2026)": "0.40 - 1.86",
     "This Study": f"{d_eq_mm_min:.2f} - {d_eq_mm_max:.2f}"},
    {"Metric": "Re_p range",
     "Lofty et al. (2026)": "Not reported clearly",
     "This Study": f"{re_min:.0f} - {re_max:.0f}"},
    {"Metric": "ML settling velocity prediction",
     "Lofty et al. (2026)": "None",
     "This Study": f"R2 = {ml_r2}"},
    {"Metric": "3D trajectory reconstruction",
     "Lofty et al. (2026)": "Yes",
     "This Study": "Yes"},
]

table3 = pd.DataFrame(rows_t3)
table3.to_csv(os.path.join(OUTPUT_DIR, "table3_comparison_lofty.csv"), index=False)
print_table("Table 3: Comparison with Lofty et al. (2026)", table3)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("-" * 80)
print("Tablolar kaydedildi:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  -> {os.path.join(OUTPUT_DIR, f)}")
print("-" * 80)
