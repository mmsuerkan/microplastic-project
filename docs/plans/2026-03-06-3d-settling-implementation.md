# 3D Settling Dynamics Paper - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build 3D trajectory reconstruction from paired MAK+ANG videos, compute Lofty-style settling metrics, update ML model with 3D features, and generate publication-ready figures.

**Architecture:** Read paired MAK+ANG tracking CSVs from processed_results/, merge frame-by-frame (x,z) from MAK and (y,z) from ANG into 3D coordinates, compute trajectory metrics (drift, tortuosity, amplitude, velocities), export to analysis CSV, retrain ML model with new features, generate figures with matplotlib.

**Tech Stack:** Python 3.10, pandas, numpy, scipy (LOWESS), matplotlib, scikit-learn, joblib

---

## Task 1: Build MAK+ANG Pair Finder

**Files:**
- Create: `analysis_3d/pair_finder.py`
- Read: `processed_results/experiments.json`
- Test: `analysis_3d/test_pair_finder.py`

**Step 1: Write the failing test**

```python
# analysis_3d/test_pair_finder.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from pair_finder import find_paired_experiments

def test_find_pairs():
    pairs = find_paired_experiments('processed_results/experiments.json')
    # We know from analysis: 348 paired experiments
    assert len(pairs) >= 300, f"Expected >= 300 pairs, got {len(pairs)}"
    # Each pair must have MAK and ANG
    for p in pairs[:5]:
        assert 'MAK' in p and 'ANG' in p
        assert p['MAK']['status'] == 'success'
        assert p['ANG']['status'] == 'success'
        assert p['MAK']['code'] == p['ANG']['code']
        assert p['MAK']['repeat'] == p['ANG']['repeat']
        assert p['MAK']['date'] == p['ANG']['date']

if __name__ == '__main__':
    test_find_pairs()
    print("PASS")
```

**Step 2: Run test to verify it fails**

Run: `cd analysis_3d && python test_pair_finder.py`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# analysis_3d/pair_finder.py
import json
from collections import defaultdict

def find_paired_experiments(json_path):
    """Find experiments that have both MAK and ANG successful recordings."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Group by (code, repeat, date)
    groups = defaultdict(dict)
    for exp in data['experiments']:
        if exp.get('status') != 'success':
            continue
        key = (exp['code'], exp['repeat'], exp['date'])
        view = exp['view']
        groups[key][view] = exp

    # Keep only pairs with both MAK and ANG
    pairs = []
    for key, views in groups.items():
        if 'MAK' in views and 'ANG' in views:
            pairs.append({
                'code': key[0],
                'repeat': key[1],
                'date': key[2],
                'MAK': views['MAK'],
                'ANG': views['ANG']
            })

    return pairs
```

**Step 4: Run test to verify it passes**

Run: `cd analysis_3d && python test_pair_finder.py`
Expected: PASS

**Step 5: Commit**

```bash
git add analysis_3d/pair_finder.py analysis_3d/test_pair_finder.py
git commit -m "feat: MAK+ANG pair finder for 3D reconstruction"
```

---

## Task 2: Pixel-to-CM Calibration Module

**Files:**
- Create: `analysis_3d/calibration.py`
- Test: `analysis_3d/test_calibration.py`

**Context:** Column is cylindrical, inner diameter 30 cm, calibration height 28.5 cm. Each camera sees the column from one side. We need to convert pixel coordinates to real-world cm. The tracking CSVs have pixel Y as vertical (top=0, increases downward). The calibration uses the known vertical distance (28.5 cm) visible in the detection window.

**Step 1: Write the failing test**

```python
# analysis_3d/test_calibration.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from calibration import PixelCalibrator

def test_calibration_from_summary():
    """Test calibration using known column height."""
    # From BSP-1 MAK SECOND: start_y=201, end_y=826 -> 625 pixels = 28.5 cm
    cal = PixelCalibrator(column_height_cm=28.5)
    cal.calibrate_from_vertical_pixels(vertical_pixels=625)

    assert abs(cal.cm_per_pixel - (28.5 / 625)) < 0.0001

    # Convert some coordinates
    x_cm, z_cm = cal.pixel_to_cm(x_px=537, y_px=201, ref_y=201)
    assert abs(z_cm - 0.0) < 0.01  # start = 0

    x_cm2, z_cm2 = cal.pixel_to_cm(x_px=564, y_px=826, ref_y=201)
    assert abs(z_cm2 - 28.5) < 0.5  # end ~ 28.5 cm

def test_horizontal_calibration():
    """Horizontal uses same scale factor (square pixels assumed)."""
    cal = PixelCalibrator(column_height_cm=28.5)
    cal.calibrate_from_vertical_pixels(vertical_pixels=625)

    # 27 pixels horizontal drift
    drift_cm = 27 * cal.cm_per_pixel
    assert abs(drift_cm - 1.23) < 0.1  # ~1.23 cm

if __name__ == '__main__':
    test_calibration_from_summary()
    test_horizontal_calibration()
    print("PASS")
```

**Step 2: Run test to verify it fails**

Run: `cd analysis_3d && python test_calibration.py`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# analysis_3d/calibration.py

class PixelCalibrator:
    """Convert pixel coordinates to real-world cm using known column height."""

    def __init__(self, column_height_cm=28.5):
        self.column_height_cm = column_height_cm
        self.cm_per_pixel = None

    def calibrate_from_vertical_pixels(self, vertical_pixels):
        """Set scale using known vertical distance in pixels."""
        self.cm_per_pixel = self.column_height_cm / vertical_pixels

    def pixel_to_cm(self, x_px, y_px, ref_y=0):
        """Convert pixel (x, y) to cm (x_cm, z_cm). y_px increases downward."""
        if self.cm_per_pixel is None:
            raise ValueError("Not calibrated. Call calibrate_from_vertical_pixels first.")
        x_cm = x_px * self.cm_per_pixel
        z_cm = (y_px - ref_y) * self.cm_per_pixel  # z positive downward
        return x_cm, z_cm
```

**Step 4: Run test to verify it passes**

Run: `cd analysis_3d && python test_calibration.py`
Expected: PASS

**Step 5: Commit**

```bash
git add analysis_3d/calibration.py analysis_3d/test_calibration.py
git commit -m "feat: pixel-to-cm calibration for settling column"
```

---

## Task 3: 3D Trajectory Reconstruction

**Files:**
- Create: `analysis_3d/reconstruct_3d.py`
- Read: `processed_results/success/{date}/{view}/{repeat}/{cat}/{code}/auto_tracking_results.csv`
- Test: `analysis_3d/test_reconstruct_3d.py`

**Context:** MAK camera gives (x_mak, z_mak) view. ANG camera gives (y_ang, z_ang) view. Both see vertical (z) axis. Frames may differ in count (MAK and ANG videos can have different total frames). We align by normalizing z-position, then interpolate the shorter trajectory to match.

**Step 1: Write the failing test**

```python
# analysis_3d/test_reconstruct_3d.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from reconstruct_3d import reconstruct_trajectory_3d

def test_reconstruct_known_pair():
    """Test 3D reconstruction with BSP-1 SECOND 01.11.23"""
    mak_csv = 'processed_results/success/01.11.23/MAK/SECOND/BSP/BSP-1/auto_tracking_results.csv'
    ang_csv = 'processed_results/success/01.11.23/ANG/SECOND/BSP/BSP-1/auto_tracking_results.csv'

    result = reconstruct_trajectory_3d(
        mak_csv_path=mak_csv,
        ang_csv_path=ang_csv,
        column_height_cm=28.5
    )

    assert result is not None
    assert 'x_cm' in result.columns  # from MAK horizontal
    assert 'y_cm' in result.columns  # from ANG horizontal
    assert 'z_cm' in result.columns  # averaged vertical
    assert 'vx' in result.columns
    assert 'vy' in result.columns
    assert 'vz' in result.columns
    assert len(result) > 50  # should have many frames

    # z should be monotonically increasing (particle falls down)
    z_diff = result['z_cm'].diff().dropna()
    assert (z_diff > -0.5).all(), "Particle should mostly move downward"

if __name__ == '__main__':
    os.chdir('..')  # run from project root
    test_reconstruct_known_pair()
    print("PASS")
```

**Step 2: Run test to verify it fails**

Run: `cd ObjectTrackingProject && python analysis_3d/test_reconstruct_3d.py`
Expected: FAIL

**Step 3: Write implementation**

```python
# analysis_3d/reconstruct_3d.py
import pandas as pd
import numpy as np
from calibration import PixelCalibrator

def reconstruct_trajectory_3d(mak_csv_path, ang_csv_path, column_height_cm=28.5):
    """
    Reconstruct 3D trajectory from paired MAK and ANG tracking CSVs.

    MAK camera sees (X_mak, Y_mak) -> provides (x, z) in real world
    ANG camera sees (X_ang, Y_ang) -> provides (y, z) in real world

    Both cameras see vertical axis (Y in pixel = z in world).
    Horizontal pixel X in MAK = x-axis, horizontal X in ANG = y-axis.
    """
    mak = pd.read_csv(mak_csv_path)
    ang = pd.read_csv(ang_csv_path)

    # Calibrate each camera independently
    cal_mak = PixelCalibrator(column_height_cm)
    mak_vertical_range = mak['Y'].max() - mak['Y'].min()
    cal_mak.calibrate_from_vertical_pixels(mak_vertical_range)

    cal_ang = PixelCalibrator(column_height_cm)
    ang_vertical_range = ang['Y'].max() - ang['Y'].min()
    cal_ang.calibrate_from_vertical_pixels(ang_vertical_range)

    # Convert to cm
    mak_ref_y = mak['Y'].iloc[0]
    ang_ref_y = ang['Y'].iloc[0]

    mak['x_cm'] = (mak['X'] - mak['X'].iloc[0]) * cal_mak.cm_per_pixel
    mak['z_mak_cm'] = (mak['Y'] - mak_ref_y) * cal_mak.cm_per_pixel

    ang['y_cm'] = (ang['X'] - ang['X'].iloc[0]) * cal_ang.cm_per_pixel
    ang['z_ang_cm'] = (ang['Y'] - ang_ref_y) * cal_ang.cm_per_pixel

    # Normalize both trajectories by z-position (0 to 1) for alignment
    mak['z_norm'] = mak['z_mak_cm'] / mak['z_mak_cm'].max()
    ang['z_norm'] = ang['z_ang_cm'] / ang['z_ang_cm'].max()

    # Interpolate to common z-normalized grid
    n_points = min(len(mak), len(ang))
    z_common = np.linspace(0, 1, n_points)

    x_interp = np.interp(z_common, mak['z_norm'].values, mak['x_cm'].values)
    y_interp = np.interp(z_common, ang['z_norm'].values, ang['y_cm'].values)
    z_mak_interp = np.interp(z_common, mak['z_norm'].values, mak['z_mak_cm'].values)
    z_ang_interp = np.interp(z_common, ang['z_norm'].values, ang['z_ang_cm'].values)

    # Average z from both cameras (Lofty method)
    z_cm = (z_mak_interp + z_ang_interp) / 2.0

    # Build 3D trajectory DataFrame
    fps = 50.0
    dt = 1.0 / fps
    # Estimate time from average frame count
    avg_frames = (len(mak) + len(ang)) / 2.0
    t = np.linspace(0, avg_frames * dt, n_points)

    traj = pd.DataFrame({
        'time': t,
        'x_cm': x_interp,
        'y_cm': y_interp,
        'z_cm': z_cm,
    })

    # Calculate velocities (cm/s)
    traj['vx'] = np.gradient(traj['x_cm'], traj['time'])
    traj['vy'] = np.gradient(traj['y_cm'], traj['time'])
    traj['vz'] = np.gradient(traj['z_cm'], traj['time'])

    return traj
```

**Step 4: Run test to verify it passes**

Run: `cd ObjectTrackingProject && python analysis_3d/test_reconstruct_3d.py`
Expected: PASS

**Step 5: Commit**

```bash
git add analysis_3d/reconstruct_3d.py analysis_3d/test_reconstruct_3d.py
git commit -m "feat: 3D trajectory reconstruction from MAK+ANG pairs"
```

---

## Task 4: Trajectory Metrics Calculator (Lofty Equations)

**Files:**
- Create: `analysis_3d/trajectory_metrics.py`
- Test: `analysis_3d/test_trajectory_metrics.py`

**Context:** Implements equations 7-15 from Lofty et al. 2026 Section 2.4.

**Step 1: Write the failing test**

```python
# analysis_3d/test_trajectory_metrics.py
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from trajectory_metrics import compute_all_metrics

def test_straight_line_trajectory():
    """A perfectly straight vertical trajectory should have zero drift and tortuosity."""
    traj = pd.DataFrame({
        'time': np.linspace(0, 5, 250),
        'x_cm': np.zeros(250),
        'y_cm': np.zeros(250),
        'z_cm': np.linspace(0, 28.5, 250),
        'vx': np.zeros(250),
        'vy': np.zeros(250),
        'vz': np.full(250, 28.5 / 5.0),
    })

    metrics = compute_all_metrics(traj, d_eq_cm=0.3)

    assert abs(metrics['w_cm_s'] - 5.7) < 0.1         # 28.5/5 = 5.7 cm/s
    assert abs(metrics['V_cm_s']) < 0.1                 # no horizontal velocity
    assert abs(metrics['drift_cm']) < 0.01              # no drift
    assert abs(metrics['drift_normalized']) < 0.1       # delta* ~ 0
    assert abs(metrics['tortuosity_pct']) < 0.1         # straight line = 0%
    assert abs(metrics['drift_gradient']) < 0.01        # no lateral/vertical ratio

def test_spiral_trajectory():
    """A spiral trajectory should have non-zero drift and tortuosity."""
    t = np.linspace(0, 5, 250)
    traj = pd.DataFrame({
        'time': t,
        'x_cm': 2.0 * np.sin(2 * np.pi * t),   # oscillating x
        'y_cm': 2.0 * np.cos(2 * np.pi * t),   # oscillating y
        'z_cm': np.linspace(0, 28.5, 250),
        'vx': 2.0 * 2 * np.pi * np.cos(2 * np.pi * t),
        'vy': -2.0 * 2 * np.pi * np.sin(2 * np.pi * t),
        'vz': np.full(250, 28.5 / 5.0),
    })

    metrics = compute_all_metrics(traj, d_eq_cm=0.3)

    assert metrics['tortuosity_pct'] > 1.0     # definitely tortuous
    assert metrics['amplitude_cm'] > 0.5       # has lateral oscillation

if __name__ == '__main__':
    test_straight_line_trajectory()
    test_spiral_trajectory()
    print("PASS")
```

**Step 2: Run test to verify it fails**

Run: `cd analysis_3d && python test_trajectory_metrics.py`
Expected: FAIL

**Step 3: Write implementation**

```python
# analysis_3d/trajectory_metrics.py
import numpy as np

def compute_all_metrics(traj, d_eq_cm):
    """
    Compute settling trajectory metrics following Lofty et al. (2026).

    Args:
        traj: DataFrame with columns [time, x_cm, y_cm, z_cm, vx, vy, vz]
        d_eq_cm: equivalent diameter in cm = (a*b*c)^(1/3) or geometric mean

    Returns:
        dict with all metrics
    """
    x = traj['x_cm'].values
    y = traj['y_cm'].values
    z = traj['z_cm'].values
    vx = traj['vx'].values
    vy = traj['vy'].values
    vz = traj['vz'].values

    # Eq 7: Settling velocity (ensemble-averaged vertical)
    w = np.mean(vz)

    # Eq 7: Horizontal drift velocity
    V_h = np.mean(np.sqrt(vx**2 + vy**2))

    # Linear regression for average trajectory (3D)
    from numpy.polynomial.polynomial import polyfit
    z_norm = (z - z[0]) / (z[-1] - z[0]) if z[-1] != z[0] else np.zeros_like(z)
    x_fit = np.polyfit(z_norm, x, 1)  # linear fit x vs z_norm
    y_fit = np.polyfit(z_norm, y, 1)  # linear fit y vs z_norm
    x_avg = np.polyval(x_fit, z_norm)
    y_avg = np.polyval(y_fit, z_norm)

    # Eq 9: Drift (Euclidean distance in horizontal plane)
    x_prime = x_avg[-1]  # average trajectory final x
    y_prime = y_avg[-1]  # average trajectory final y
    drift = np.sqrt((x_prime - x[0])**2 + (y_prime - y[0])**2)

    # Eq 10: Normalized drift
    drift_star = drift / d_eq_cm if d_eq_cm > 0 else 0

    # Eq 11: Amplitude - std of lateral deviations from average trajectory
    lateral_dev = np.sqrt((x - x_avg)**2 + (y - y_avg)**2)
    amplitude = np.mean(np.std(lateral_dev))  # sigma
    amplitude_star = amplitude / d_eq_cm if d_eq_cm > 0 else 0

    # Eq 12: Drift gradient
    vertical_distance = abs(z[-1] - z[0])
    drift_gradient = drift / vertical_distance if vertical_distance > 0 else 0

    # Eq 13-15: Tortuosity
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    L = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))  # actual path length
    D = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2 + (z[-1]-z[0])**2)  # straight line
    tortuosity = ((L - D) / D) * 100 if D > 0 else 0

    # Eq 8: Particle Reynolds number
    nu = 1.004e-2  # kinematic viscosity water at ~20C in cm^2/s
    Re_p = abs(w) * d_eq_cm / nu

    return {
        'w_cm_s': w,                          # settling velocity
        'V_cm_s': V_h,                        # horizontal drift velocity
        'drift_cm': drift,                    # lateral drift
        'drift_normalized': drift_star,       # delta*
        'amplitude_cm': amplitude,            # sigma
        'amplitude_normalized': amplitude_star,  # sigma*
        'drift_gradient': drift_gradient,     # epsilon
        'tortuosity_pct': tortuosity,         # phi (%)
        'Re_p': Re_p,                         # particle Reynolds number
        'path_length_cm': L,                  # actual path length
        'straight_distance_cm': D,            # straight-line distance
        'vertical_distance_cm': vertical_distance,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd analysis_3d && python test_trajectory_metrics.py`
Expected: PASS

**Step 5: Commit**

```bash
git add analysis_3d/trajectory_metrics.py analysis_3d/test_trajectory_metrics.py
git commit -m "feat: Lofty-style trajectory metrics (drift, tortuosity, amplitude)"
```

---

## Task 5: Batch 3D Analysis Pipeline

**Files:**
- Create: `analysis_3d/batch_3d_analysis.py`
- Read: `processed_results/experiments.json`, `data/training_data_particle_avg.csv`
- Output: `analysis_3d/results_3d.csv`

**Context:** Run 3D reconstruction + metrics for all 348 paired experiments, merge with particle properties (density, shape, dimensions), export a single analysis CSV.

**Step 1: Write the script**

```python
# analysis_3d/batch_3d_analysis.py
"""
Batch 3D trajectory analysis for all paired MAK+ANG experiments.
Outputs: analysis_3d/results_3d.csv
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from pair_finder import find_paired_experiments
from reconstruct_3d import reconstruct_trajectory_3d
from trajectory_metrics import compute_all_metrics

def get_particle_properties(training_csv):
    """Load particle properties (a, b, c, density, shape) from training data."""
    df = pd.read_csv(training_csv)
    props = {}
    for _, row in df.iterrows():
        code = row['code']
        cat = row['category']
        key = f"{cat}/{code}" if '/' not in code else code
        # Also store by just code for matching
        props[code] = {
            'category': cat,
            'a': row['a'], 'b': row['b'], 'c': row['c'],
            'density': row['density'],
            'shape_name': row['shape_name'],
            'shape_enc': row['shape_enc'],
            'd_eq': (row['a'] * row['b'] * max(row['c'], row['a']))**(1/3),  # geometric mean
            'volume': row.get('volume', 0),
            'aspect_ratio': row.get('aspect_ratio', 0),
        }
    return props

def run_batch_analysis():
    base = os.path.join(os.path.dirname(__file__), '..')
    json_path = os.path.join(base, 'processed_results', 'experiments.json')
    training_path = os.path.join(base, 'data', 'training_data_particle_avg.csv')
    results_dir = os.path.join(base, 'processed_results')

    pairs = find_paired_experiments(json_path)
    props = get_particle_properties(training_path)

    print(f"Found {len(pairs)} paired experiments")

    results = []
    errors = 0
    for i, pair in enumerate(pairs):
        code = pair['code']
        mak_path = os.path.join(results_dir, pair['MAK']['path'], 'auto_tracking_results.csv')
        ang_path = os.path.join(results_dir, pair['ANG']['path'], 'auto_tracking_results.csv')

        if not os.path.exists(mak_path) or not os.path.exists(ang_path):
            errors += 1
            continue

        try:
            traj = reconstruct_trajectory_3d(mak_path, ang_path)

            # Get d_eq from particle properties
            particle = props.get(code, {})
            a = particle.get('a', 0.4)
            b = particle.get('b', 0.4)
            c = particle.get('c', a)  # if c=0, use a
            if c == 0:
                c = a
            d_eq = (a * b * c) ** (1/3) / 10.0  # mm to cm

            metrics = compute_all_metrics(traj, d_eq_cm=d_eq)

            row = {
                'code': code,
                'repeat': pair['repeat'],
                'date': pair['date'],
                'category': pair['MAK'].get('category', ''),
                'shape': particle.get('shape_name', ''),
                'a_mm': a, 'b_mm': b, 'c_mm': c,
                'density': particle.get('density', 0),
                'd_eq_mm': d_eq * 10,
                'mak_frames': len(pd.read_csv(mak_path)),
                'ang_frames': len(pd.read_csv(ang_path)),
            }
            row.update(metrics)
            results.append(row)

            if (i+1) % 50 == 0:
                print(f"  Processed {i+1}/{len(pairs)}")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error {code} {pair['repeat']} {pair['date']}: {e}")

    df = pd.DataFrame(results)
    out_path = os.path.join(os.path.dirname(__file__), 'results_3d.csv')
    df.to_csv(out_path, index=False)
    print(f"\nDone: {len(results)} trajectories analyzed, {errors} errors")
    print(f"Saved to: {out_path}")
    return df

if __name__ == '__main__':
    run_batch_analysis()
```

**Step 2: Run it**

Run: `cd ObjectTrackingProject && python analysis_3d/batch_3d_analysis.py`
Expected: Processes ~348 pairs, outputs `analysis_3d/results_3d.csv`

**Step 3: Verify output**

```python
import pandas as pd
df = pd.read_csv('analysis_3d/results_3d.csv')
print(f"Rows: {len(df)}")
print(df.describe())
print(df.groupby('shape')['tortuosity_pct'].mean())
```

**Step 4: Commit**

```bash
git add analysis_3d/batch_3d_analysis.py analysis_3d/results_3d.csv
git commit -m "feat: batch 3D analysis for 348 paired experiments"
```

---

## Task 6: Publication Figures

**Files:**
- Create: `analysis_3d/generate_figures.py`
- Read: `analysis_3d/results_3d.csv`
- Output: `analysis_3d/figures/fig1_setup.png` through `fig6_comparison.png`

**Step 1: Write figure generation script**

Key figures to generate:

```python
# analysis_3d/generate_figures.py
"""Generate publication-ready figures for the 3D settling paper."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12, 'figure.dpi': 300})
import os

def load_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'results_3d.csv'))

def fig2_3d_trajectories(df):
    """Fig 2: Example 3D trajectories colored by shape type."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Select 1 representative particle per shape
    shapes = df['shape'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(shapes)))
    # This needs actual trajectory data - will load from CSVs
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title('3D Settling Trajectories by Shape')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'fig2_3d_trajectories.png'),
                bbox_inches='tight')
    plt.close()

def fig3_drift_tortuosity(df):
    """Fig 3: Drift and tortuosity box plots by shape."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    shapes_order = df.groupby('shape')['drift_normalized'].median().sort_values().index

    # Normalized drift
    df.boxplot(column='drift_normalized', by='shape', ax=axes[0])
    axes[0].set_title('Normalized Drift (delta*)')
    axes[0].set_ylabel('delta / D_eq')

    # Tortuosity
    df.boxplot(column='tortuosity_pct', by='shape', ax=axes[1])
    axes[1].set_title('Tortuosity (%)')
    axes[1].set_ylabel('phi (%)')

    # Drift gradient
    df.boxplot(column='drift_gradient', by='shape', ax=axes[2])
    axes[2].set_title('Drift Gradient (epsilon)')
    axes[2].set_ylabel('delta / vertical distance')

    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'fig3_drift_tortuosity.png'),
                bbox_inches='tight')
    plt.close()

def fig4_re_cd(df):
    """Fig 4: Re-CD diagram."""
    nu = 1.004e-6  # m^2/s
    rho_f = 998     # kg/m^3
    g = 9.81

    fig, ax = plt.subplots(figsize=(8, 6))

    d_eq_m = df['d_eq_mm'].values / 1000
    w_m = df['w_cm_s'].values / 100
    rho_p = df['density'].values

    Re = rho_f * np.abs(w_m) * d_eq_m / (nu * rho_f)  # simplified
    CD = np.where(w_m > 0,
                  4/3 * (rho_p - rho_f) * g * d_eq_m / (rho_f * w_m**2),
                  np.nan)

    shapes = df['shape'].unique()
    for shape in shapes:
        mask = df['shape'] == shape
        ax.scatter(Re[mask], CD[mask], label=shape, alpha=0.6, s=30)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Re_p')
    ax.set_ylabel('C_D')
    ax.legend()
    ax.set_title('Drag Coefficient vs Reynolds Number')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'fig4_re_cd.png'),
                bbox_inches='tight')
    plt.close()

def fig5_ml_prediction(df):
    """Fig 5: ML predicted vs measured + feature importance."""
    import joblib
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'ml_models', 'best_model')

    rf = joblib.load(os.path.join(model_dir, 'rf_model.joblib'))
    # Feature importance from RF
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: predicted vs measured (placeholder - needs actual prediction)
    axes[0].set_xlabel('Measured w (cm/s)')
    axes[0].set_ylabel('Predicted w (cm/s)')
    axes[0].set_title('ML Prediction Performance')

    # Right: feature importance
    import json
    with open(os.path.join(model_dir, 'model_params.json')) as f:
        params = json.load(f)
    features = params['feature_columns']
    importances = rf.feature_importances_
    idx = np.argsort(importances)
    axes[1].barh([features[i] for i in idx], importances[idx])
    axes[1].set_title('Feature Importance (Random Forest)')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'fig5_ml_prediction.png'),
                bbox_inches='tight')
    plt.close()

def main():
    os.makedirs(os.path.join(os.path.dirname(__file__), 'figures'), exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} records")

    fig3_drift_tortuosity(df)
    print("Fig 3: drift & tortuosity done")

    fig4_re_cd(df)
    print("Fig 4: Re-CD done")

    fig5_ml_prediction(df)
    print("Fig 5: ML prediction done")

if __name__ == '__main__':
    main()
```

**Step 2: Run and iterate**

Run: `cd ObjectTrackingProject && python analysis_3d/generate_figures.py`

**Step 3: Commit**

```bash
git add analysis_3d/generate_figures.py analysis_3d/figures/
git commit -m "feat: publication figures for 3D settling paper"
```

---

## Task 7: ML Model with 3D Features

**Files:**
- Create: `analysis_3d/train_with_3d_features.py`
- Read: `analysis_3d/results_3d.csv`, `data/training_data_particle_avg.csv`
- Output: model comparison report

**Step 1: Write training script**

```python
# analysis_3d/train_with_3d_features.py
"""
Compare ML model with and without 3D trajectory features.
Baseline: 9 features, R^2 = 0.859
New: 9 + 3D features (drift, tortuosity, amplitude)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def train_and_compare():
    # Load baseline training data
    baseline = pd.read_csv('../data/training_data_particle_avg.csv')

    # Load 3D results
    results_3d = pd.read_csv('results_3d.csv')

    # Average 3D metrics per particle (across repeats)
    metrics_avg = results_3d.groupby('code').agg({
        'w_cm_s': 'mean',
        'V_cm_s': 'mean',
        'drift_normalized': 'mean',
        'tortuosity_pct': 'mean',
        'amplitude_normalized': 'mean',
        'drift_gradient': 'mean',
    }).reset_index()

    # Merge with baseline
    merged = baseline.merge(metrics_avg, on='code', how='inner')
    print(f"Merged: {len(merged)} particles (from {len(baseline)} baseline)")

    # Baseline features
    base_features = ['a', 'b', 'c', 'density', 'shape_enc',
                     'volume', 'surface_area', 'aspect_ratio', 'vol_surf_ratio']

    # New features
    new_features = base_features + ['drift_normalized', 'tortuosity_pct', 'amplitude_normalized']

    y = merged['velocity_mean'].values

    # Baseline model
    X_base = merged[base_features].values
    scaler_base = StandardScaler()
    X_base_scaled = scaler_base.fit_transform(X_base)

    rf_base = RandomForestRegressor(n_estimators=100, max_depth=10,
                                     min_samples_leaf=4, random_state=42)
    scores_base = cross_val_score(rf_base, X_base_scaled, y, cv=5, scoring='r2')

    # New model with 3D features
    X_new = merged[new_features].values
    scaler_new = StandardScaler()
    X_new_scaled = scaler_new.fit_transform(X_new)

    rf_new = RandomForestRegressor(n_estimators=100, max_depth=10,
                                    min_samples_leaf=4, random_state=42)
    scores_new = cross_val_score(rf_new, X_new_scaled, y, cv=5, scoring='r2')

    print(f"\n=== COMPARISON ===")
    print(f"Baseline (9 features):     R^2 = {scores_base.mean():.3f} +/- {scores_base.std():.3f}")
    print(f"With 3D  ({len(new_features)} features): R^2 = {scores_new.mean():.3f} +/- {scores_new.std():.3f}")
    print(f"Improvement: {(scores_new.mean() - scores_base.mean()):.3f}")

    # Feature importance
    rf_new.fit(X_new_scaled, y)
    for feat, imp in sorted(zip(new_features, rf_new.feature_importances_),
                            key=lambda x: -x[1]):
        marker = " <-- NEW" if feat in ['drift_normalized', 'tortuosity_pct', 'amplitude_normalized'] else ""
        print(f"  {feat:25s}: {imp:.3f}{marker}")

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(__file__))
    train_and_compare()
```

**Step 2: Run**

Run: `cd ObjectTrackingProject/analysis_3d && python train_with_3d_features.py`

**Step 3: Commit**

```bash
git add analysis_3d/train_with_3d_features.py
git commit -m "feat: ML comparison with and without 3D trajectory features"
```

---

## Task 8: Export Summary Table for Paper

**Files:**
- Create: `analysis_3d/export_paper_tables.py`
- Output: `analysis_3d/tables/` (CSV tables ready for paper)

Generates summary statistics tables:
- Table 1: Particle properties summary
- Table 2: 3D settling metrics by shape
- Table 3: ML model comparison
- Table 4: Comparison with Lofty et al.

**Step 1: Write and run**

**Step 2: Commit**

```bash
git add analysis_3d/export_paper_tables.py analysis_3d/tables/
git commit -m "feat: export summary tables for paper"
```

---

## Task 9: Final Integration & Cleanup

**Step 1:** Run full pipeline end-to-end:
```bash
cd ObjectTrackingProject
python analysis_3d/batch_3d_analysis.py
python analysis_3d/generate_figures.py
python analysis_3d/train_with_3d_features.py
python analysis_3d/export_paper_tables.py
```

**Step 2:** Review all outputs, fix any issues

**Step 3:** Final commit and push
```bash
git add analysis_3d/
git commit -m "feat: complete 3D settling analysis pipeline for paper"
git push microplastic main
```

---

## Dependency Graph

```
Task 1 (pair finder) ─────┐
Task 2 (calibration) ─────┼──> Task 3 (3D reconstruction) ──> Task 5 (batch analysis) ──┐
                           │                                                              │
Task 4 (metrics) ──────────┘                                                              │
                                                                                          ├──> Task 9 (integration)
Task 6 (figures) <────────────────────────────────────────────────────────────────────────┤
Task 7 (ML with 3D) <────────────────────────────────────────────────────────────────────┤
Task 8 (tables) <─────────────────────────────────────────────────────────────────────────┘
```
