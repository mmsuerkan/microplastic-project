"""
Trajectory metrics from Lofty et al. (2026) Section 2.4.

Computes settling velocity, horizontal drift, Reynolds number,
drift, amplitude, tortuosity, and related normalized quantities
from a 3D particle trajectory DataFrame.

Input DataFrame columns: [time, x_cm, y_cm, z_cm, vx, vy, vz]
  - z_cm increases downward (particle falls)
  - time in seconds
  - velocities in cm/s
"""
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Kinematic viscosity of water at ~20 C [cm^2/s]
NU_WATER = 1.004e-2


def _settling_velocity(traj: pd.DataFrame) -> float:
    """Eq 7: w = mean(vz) -- ensemble-averaged vertical velocity."""
    return float(traj['vz'].mean())


def _horizontal_drift_velocity(traj: pd.DataFrame) -> float:
    """Eq 7: V = mean(sqrt(vx^2 + vy^2)) -- mean horizontal speed."""
    v_horiz = np.sqrt(traj['vx'] ** 2 + traj['vy'] ** 2)
    return float(v_horiz.mean())


def _reynolds_number(w: float, d_eq_cm: float, nu: float = NU_WATER) -> float:
    """Eq 8: Re_p = w * D_eq / nu."""
    if d_eq_cm <= 0:
        return 0.0
    return abs(w) * d_eq_cm / nu


def _drift_and_amplitude(traj: pd.DataFrame, d_eq_cm: float):
    """
    Eq 9-11: Drift, normalized drift, amplitude, normalized amplitude.

    Drift (delta):
        Fit linear regression x'(z), y'(z). Then:
        delta = sqrt((x'(z*) - x'(z0))^2 + (y'(z*) - y'(z0))^2)
        where z0 = first z, z* = last z.

    Amplitude (sigma):
        Standard deviation of lateral deviations from the average
        trajectory line, projected perpendicular to the drift direction.
    """
    z = traj['z_cm'].values
    x = traj['x_cm'].values
    y = traj['y_cm'].values

    z0 = z[0]
    z_star = z[-1]
    dz = z_star - z0

    # Linear regression of x and y as functions of z
    if len(z) < 2 or abs(dz) < 1e-12:
        return {
            'drift_cm': 0.0,
            'drift_normalized': 0.0,
            'amplitude_cm': 0.0,
            'amplitude_normalized': 0.0,
        }

    # polyfit: x = a_x * z + b_x
    coeffs_x = np.polyfit(z, x, 1)  # [slope, intercept]
    coeffs_y = np.polyfit(z, y, 1)

    a_x, b_x = coeffs_x
    a_y, b_y = coeffs_y

    # Predicted positions at start and end
    x_pred_start = a_x * z0 + b_x
    y_pred_start = a_y * z0 + b_y
    x_pred_end = a_x * z_star + b_x
    y_pred_end = a_y * z_star + b_y

    # Eq 9: drift
    delta = np.sqrt((x_pred_end - x_pred_start) ** 2 +
                    (y_pred_end - y_pred_start) ** 2)

    # Eq 10: normalized drift
    delta_star = delta / d_eq_cm if d_eq_cm > 0 else 0.0

    # Eq 11: Amplitude - std of lateral deviations from the average
    # trajectory line, projected perpendicular to the drift direction.
    # Residuals from the fit line
    x_resid = x - (a_x * z + b_x)
    y_resid = y - (a_y * z + b_y)

    # Drift direction vector in x-y plane
    dx_drift = x_pred_end - x_pred_start
    dy_drift = y_pred_end - y_pred_start
    drift_mag = np.sqrt(dx_drift ** 2 + dy_drift ** 2)

    if drift_mag > 1e-12:
        # Perpendicular direction: rotate 90 degrees
        perp_x = -dy_drift / drift_mag
        perp_y = dx_drift / drift_mag
        # Project residuals onto perpendicular direction
        perp_deviations = x_resid * perp_x + y_resid * perp_y
    else:
        # No drift direction -- use total residual magnitude
        perp_deviations = np.sqrt(x_resid ** 2 + y_resid ** 2)

    sigma = float(np.std(perp_deviations))
    sigma_star = sigma / d_eq_cm if d_eq_cm > 0 else 0.0

    return {
        'drift_cm': float(delta),
        'drift_normalized': float(delta_star),
        'amplitude_cm': sigma,
        'amplitude_normalized': sigma_star,
    }


def _drift_gradient(traj: pd.DataFrame, drift_cm: float) -> float:
    """
    Eq 12: epsilon = delta / (z0 - z*).
    Since z increases downward, vertical travel = z* - z0.
    We use abs to get a positive gradient.
    """
    z0 = traj['z_cm'].values[0]
    z_star = traj['z_cm'].values[-1]
    vertical_distance = abs(z_star - z0)

    if vertical_distance < 1e-12:
        return 0.0

    return drift_cm / vertical_distance


def _tortuosity(traj: pd.DataFrame):
    """
    Eq 13-15: Tortuosity.
    L = sum of 3D step distances (actual path length)
    D = straight-line distance between start and end
    phi = ((L - D) / D) * 100 [percent]
    """
    x = traj['x_cm'].values
    y = traj['y_cm'].values
    z = traj['z_cm'].values

    # Step distances
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    step_distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # Eq 13: L = total path length
    L = float(np.sum(step_distances))

    # Eq 14: D = straight-line distance
    D = float(np.sqrt((x[-1] - x[0]) ** 2 +
                       (y[-1] - y[0]) ** 2 +
                       (z[-1] - z[0]) ** 2))

    # Eq 15: phi = ((L - D) / D) * 100
    if D < 1e-12:
        phi = 0.0
    else:
        phi = ((L - D) / D) * 100.0

    return {
        'path_length_cm': L,
        'straight_distance_cm': D,
        'tortuosity_pct': phi,
    }


def compute_all_metrics(traj_df: pd.DataFrame, d_eq_cm: float) -> dict:
    """
    Compute all trajectory metrics from Lofty et al. (2026) Section 2.4.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Trajectory data with columns [time, x_cm, y_cm, z_cm, vx, vy, vz].
        z_cm increases downward. time in seconds. Velocities in cm/s.
    d_eq_cm : float
        Equivalent diameter of the particle in cm.

    Returns
    -------
    dict with keys:
        - settling_velocity_cm_s: mean vertical velocity (Eq 7)
        - horizontal_drift_velocity_cm_s: mean horizontal speed (Eq 7)
        - reynolds_number: particle Reynolds number (Eq 8)
        - drift_cm: total horizontal drift (Eq 9)
        - drift_normalized: drift / D_eq (Eq 10)
        - amplitude_cm: std of lateral deviations (Eq 11)
        - amplitude_normalized: amplitude / D_eq (Eq 11)
        - drift_gradient: drift / vertical_distance (Eq 12)
        - path_length_cm: total 3D path length L (Eq 13)
        - straight_distance_cm: straight-line distance D (Eq 14)
        - tortuosity_pct: ((L-D)/D)*100 (Eq 15)
    """
    required_cols = {'time', 'x_cm', 'y_cm', 'z_cm', 'vx', 'vy', 'vz'}
    missing = required_cols - set(traj_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(traj_df) < 2:
        raise ValueError("Trajectory must have at least 2 data points.")

    if d_eq_cm <= 0:
        raise ValueError(f"d_eq_cm must be positive, got {d_eq_cm}")

    # Eq 7
    w = _settling_velocity(traj_df)
    v_horiz = _horizontal_drift_velocity(traj_df)

    # Eq 8
    re_p = _reynolds_number(w, d_eq_cm)

    # Eq 9-11
    drift_amp = _drift_and_amplitude(traj_df, d_eq_cm)

    # Eq 12
    eps = _drift_gradient(traj_df, drift_amp['drift_cm'])

    # Eq 13-15
    tort = _tortuosity(traj_df)

    return {
        'settling_velocity_cm_s': w,
        'horizontal_drift_velocity_cm_s': v_horiz,
        'reynolds_number': re_p,
        'drift_cm': drift_amp['drift_cm'],
        'drift_normalized': drift_amp['drift_normalized'],
        'amplitude_cm': drift_amp['amplitude_cm'],
        'amplitude_normalized': drift_amp['amplitude_normalized'],
        'drift_gradient': eps,
        'path_length_cm': tort['path_length_cm'],
        'straight_distance_cm': tort['straight_distance_cm'],
        'tortuosity_pct': tort['tortuosity_pct'],
    }
