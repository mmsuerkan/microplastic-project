"""
Batch 3D analysis pipeline for all MAK+ANG paired experiments.

Loads 348 paired experiments, reconstructs 3D trajectories, computes
settling metrics (Lofty et al. equations), and saves results to CSV.

Usage
-----
    python analysis_3d/batch_3d_analysis.py
"""
import sys
import os
import time
import traceback

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# Ensure project root is on sys.path for imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis_3d.pair_finder import find_paired_experiments
from analysis_3d.reconstruct_3d import reconstruct_trajectory_3d
from analysis_3d.trajectory_metrics import compute_all_metrics


def load_particle_properties(csv_path: str) -> dict:
    """Load particle properties indexed by (category, code).

    Parameters
    ----------
    csv_path : str
        Path to training_data_particle_avg.csv.

    Returns
    -------
    dict
        Mapping of (category, code) -> row dict with keys:
        a, b, c, density, shape_name, shape_enc, volume, surface_area,
        aspect_ratio, vol_surf_ratio.
    """
    df = pd.read_csv(csv_path)
    props = {}
    for _, row in df.iterrows():
        key = (row["category"], row["code"])
        props[key] = row.to_dict()
    return props


def compute_d_eq_cm(a_mm: float, b_mm: float, c_mm: float) -> float:
    """Compute equivalent diameter in cm from dimensions in mm.

    d_eq = (a * b * c)^(1/3) in mm, then convert to cm by /10.
    When c == 0 (e.g. cylinders), use c = a as approximation.
    """
    if c_mm == 0.0:
        c_mm = a_mm
    d_eq_mm = (a_mm * b_mm * c_mm) ** (1.0 / 3.0)
    return d_eq_mm / 10.0


def run_batch_analysis():
    """Run 3D analysis for all paired experiments."""
    # Paths
    json_path = os.path.join(_PROJECT_ROOT, "processed_results", "experiments.json")
    training_csv = os.path.join(_PROJECT_ROOT, "data", "training_data_particle_avg.csv")
    base_dir = os.path.join(_PROJECT_ROOT, "processed_results")
    output_csv = os.path.join(_THIS_DIR, "results_3d.csv")

    # 1. Load pairs
    print("Paired experiment'lar yukleniyor...")
    pairs = find_paired_experiments(json_path)
    print(f"  Toplam pair: {len(pairs)}")

    # 2. Load particle properties
    print("Parcacik ozellikleri yukleniyor...")
    particle_props = load_particle_properties(training_csv)
    print(f"  Toplam parcacik: {len(particle_props)}")

    # 3. Process each pair
    results = []
    errors = []
    skipped_no_props = 0
    t_start = time.time()

    for i, pair in enumerate(pairs):
        code = pair["code"]
        repeat = pair["repeat"]
        date = pair["date"]
        category = pair["MAK"]["category"]

        # Progress
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(pairs)}] isleniyor... ({elapsed:.1f}s gecti)")

        # Look up particle properties by (category, code)
        prop_key = (category, code)
        if prop_key not in particle_props:
            skipped_no_props += 1
            continue

        props = particle_props[prop_key]
        a_mm = props["a"]
        b_mm = props["b"]
        c_mm = props["c"]
        density = props["density"]
        shape_name = props["shape_name"]
        shape_enc = props["shape_enc"]

        # Compute d_eq
        d_eq_cm = compute_d_eq_cm(a_mm, b_mm, c_mm)

        # Build CSV paths
        mak_csv = os.path.join(base_dir, pair["MAK"]["path"], "auto_tracking_results.csv")
        ang_csv = os.path.join(base_dir, pair["ANG"]["path"], "auto_tracking_results.csv")

        try:
            # 3D reconstruction
            traj_df = reconstruct_trajectory_3d(mak_csv, ang_csv)

            # Compute metrics
            metrics = compute_all_metrics(traj_df, d_eq_cm)

            # Collect result row
            row = {
                "code": code,
                "category": category,
                "repeat": repeat,
                "date": date,
                "shape_name": shape_name,
                "shape_enc": shape_enc,
                "a_mm": a_mm,
                "b_mm": b_mm,
                "c_mm": c_mm,
                "density_kg_m3": density,
                "d_eq_cm": d_eq_cm,
                "n_points": len(traj_df),
                "duration_s": float(traj_df["time"].iloc[-1] - traj_df["time"].iloc[0]),
                "z_travel_cm": float(traj_df["z_cm"].iloc[-1] - traj_df["z_cm"].iloc[0]),
            }
            row.update(metrics)
            results.append(row)

        except Exception as e:
            errors.append({
                "code": code,
                "category": category,
                "repeat": repeat,
                "date": date,
                "error": str(e),
            })

    elapsed_total = time.time() - t_start

    # 4. Save results
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"\nSonuclar kaydedildi: {output_csv}")
        print(f"  Basarili: {len(results)}")
    else:
        print("\nHicbir sonuc uretilmedi!")

    # 5. Summary
    print(f"\n{'='*60}")
    print(f"BATCH 3D ANALIZ OZETI")
    print(f"{'='*60}")
    print(f"Toplam pair:          {len(pairs)}")
    print(f"Parcacik bulunamayan: {skipped_no_props}")
    print(f"Basarili:             {len(results)}")
    print(f"Hatali:               {len(errors)}")
    print(f"Sure:                 {elapsed_total:.1f} saniye")

    if results:
        df_results = pd.DataFrame(results)
        print(f"\n--- Settling Velocity (cm/s) ---")
        print(f"  Ortalama: {df_results['settling_velocity_cm_s'].mean():.3f}")
        print(f"  Std:      {df_results['settling_velocity_cm_s'].std():.3f}")
        print(f"  Min:      {df_results['settling_velocity_cm_s'].min():.3f}")
        print(f"  Max:      {df_results['settling_velocity_cm_s'].max():.3f}")

        print(f"\n--- Reynolds Number ---")
        print(f"  Ortalama: {df_results['reynolds_number'].mean():.1f}")
        print(f"  Min:      {df_results['reynolds_number'].min():.1f}")
        print(f"  Max:      {df_results['reynolds_number'].max():.1f}")

        print(f"\n--- Tortuosity (%) ---")
        print(f"  Ortalama: {df_results['tortuosity_pct'].mean():.2f}")
        print(f"  Min:      {df_results['tortuosity_pct'].min():.2f}")
        print(f"  Max:      {df_results['tortuosity_pct'].max():.2f}")

        print(f"\n--- Drift Gradient ---")
        print(f"  Ortalama: {df_results['drift_gradient'].mean():.4f}")
        print(f"  Min:      {df_results['drift_gradient'].min():.4f}")
        print(f"  Max:      {df_results['drift_gradient'].max():.4f}")

        print(f"\n--- Horizontal Drift Velocity (cm/s) ---")
        print(f"  Ortalama: {df_results['horizontal_drift_velocity_cm_s'].mean():.3f}")
        print(f"  Min:      {df_results['horizontal_drift_velocity_cm_s'].min():.3f}")
        print(f"  Max:      {df_results['horizontal_drift_velocity_cm_s'].max():.3f}")

        # Per-category summary
        print(f"\n--- Kategori Bazli Ozet ---")
        cat_summary = df_results.groupby("category").agg(
            n=("settling_velocity_cm_s", "count"),
            w_mean=("settling_velocity_cm_s", "mean"),
            w_std=("settling_velocity_cm_s", "std"),
            Re_mean=("reynolds_number", "mean"),
            tort_mean=("tortuosity_pct", "mean"),
        ).round(3)
        print(cat_summary.to_string())

        # Per-shape summary
        print(f"\n--- Sekil Bazli Ozet ---")
        shape_summary = df_results.groupby("shape_name").agg(
            n=("settling_velocity_cm_s", "count"),
            w_mean=("settling_velocity_cm_s", "mean"),
            w_std=("settling_velocity_cm_s", "std"),
            Re_mean=("reynolds_number", "mean"),
            drift_mean=("drift_gradient", "mean"),
            tort_mean=("tortuosity_pct", "mean"),
        ).round(3)
        print(shape_summary.to_string())

    if errors:
        print(f"\n--- Hatalar (ilk 10) ---")
        for err in errors[:10]:
            print(f"  {err['category']}/{err['code']} ({err['repeat']}, {err['date']}): {err['error']}")

    return results, errors


if __name__ == "__main__":
    run_batch_analysis()
