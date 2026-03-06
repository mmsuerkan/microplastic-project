"""
Tests for Lofty et al. (2026) trajectory metrics.

Three test scenarios:
1. Straight vertical fall -- no drift, no tortuosity
2. Spiral (oscillating) descent -- tortuosity > 0, amplitude > 0
3. Drifting descent -- drift > 0, drift gradient > 0
"""
import sys
import os
import unittest
import math

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis_3d.trajectory_metrics import compute_all_metrics


def _make_trajectory(x, y, z, dt=0.02):
    """
    Build a trajectory DataFrame from position arrays.
    Computes velocities via finite differences.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(x)
    time = np.arange(n) * dt

    # Central differences for velocity, forward/backward at edges
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vz = np.gradient(z, dt)

    return pd.DataFrame({
        'time': time,
        'x_cm': x,
        'y_cm': y,
        'z_cm': z,
        'vx': vx,
        'vy': vy,
        'vz': vz,
    })


class TestStraightVerticalFall(unittest.TestCase):
    """Particle falls perfectly vertical: x=0, y=0, z increases linearly."""

    def setUp(self):
        n = 100
        z = np.linspace(0, 20, n)  # 20 cm fall
        x = np.zeros(n)
        y = np.zeros(n)
        self.traj = _make_trajectory(x, y, z)
        self.d_eq = 0.4  # 4 mm particle
        self.metrics = compute_all_metrics(self.traj, self.d_eq)

    def test_settling_velocity_positive(self):
        """Vertical velocity should be positive (z increases downward)."""
        self.assertGreater(self.metrics['settling_velocity_cm_s'], 0)

    def test_settling_velocity_value(self):
        """w should equal dz/dt = 20 cm / (99 * 0.02 s) = ~10.1 cm/s."""
        expected_w = 20.0 / (99 * 0.02)
        self.assertAlmostEqual(
            self.metrics['settling_velocity_cm_s'], expected_w, places=1
        )

    def test_horizontal_drift_velocity_near_zero(self):
        """No horizontal movement -> V ~ 0."""
        self.assertAlmostEqual(
            self.metrics['horizontal_drift_velocity_cm_s'], 0.0, places=6
        )

    def test_drift_near_zero(self):
        """No lateral displacement -> drift ~ 0."""
        self.assertAlmostEqual(self.metrics['drift_cm'], 0.0, places=6)

    def test_drift_normalized_near_zero(self):
        """Normalized drift should also be ~ 0."""
        self.assertAlmostEqual(self.metrics['drift_normalized'], 0.0, places=6)

    def test_amplitude_near_zero(self):
        """No oscillation -> amplitude ~ 0."""
        self.assertAlmostEqual(self.metrics['amplitude_cm'], 0.0, places=6)

    def test_tortuosity_near_zero(self):
        """Straight path: L == D, so tortuosity ~ 0%."""
        self.assertAlmostEqual(self.metrics['tortuosity_pct'], 0.0, places=3)

    def test_reynolds_number_positive(self):
        """Re should be positive for a falling particle."""
        self.assertGreater(self.metrics['reynolds_number'], 0)

    def test_reynolds_number_value(self):
        """Re = w * d_eq / nu."""
        w = self.metrics['settling_velocity_cm_s']
        expected_re = abs(w) * 0.4 / 1.004e-2
        self.assertAlmostEqual(
            self.metrics['reynolds_number'], expected_re, places=1
        )

    def test_drift_gradient_near_zero(self):
        """No drift -> gradient ~ 0."""
        self.assertAlmostEqual(self.metrics['drift_gradient'], 0.0, places=6)


class TestSpiralDescent(unittest.TestCase):
    """Particle oscillates (x=A*sin, y=A*cos) while falling."""

    def setUp(self):
        n = 200
        t_param = np.linspace(0, 4 * np.pi, n)  # 2 full oscillations
        A = 0.5  # amplitude in cm
        z = np.linspace(0, 15, n)  # 15 cm fall
        x = A * np.sin(t_param)
        y = A * np.cos(t_param)
        self.traj = _make_trajectory(x, y, z)
        self.d_eq = 0.3  # 3 mm particle
        self.metrics = compute_all_metrics(self.traj, self.d_eq)

    def test_settling_velocity_positive(self):
        """Particle is falling, w > 0."""
        self.assertGreater(self.metrics['settling_velocity_cm_s'], 0)

    def test_horizontal_drift_velocity_positive(self):
        """There is horizontal movement (oscillation)."""
        self.assertGreater(
            self.metrics['horizontal_drift_velocity_cm_s'], 0
        )

    def test_tortuosity_positive(self):
        """Spiral path is longer than straight line -> tortuosity > 0."""
        self.assertGreater(self.metrics['tortuosity_pct'], 0)

    def test_amplitude_positive(self):
        """Oscillation produces lateral deviations -> amplitude > 0."""
        self.assertGreater(self.metrics['amplitude_cm'], 0)

    def test_drift_small_for_symmetric_oscillation(self):
        """
        Symmetric oscillation over full periods: linear fit should show
        small net drift relative to vertical fall (15 cm).
        Drift / vertical_distance should be much less than 1.
        """
        drift_ratio = self.metrics['drift_cm'] / 15.0
        self.assertLess(drift_ratio, 0.1)

    def test_path_length_greater_than_straight(self):
        """Spiral path L > straight-line distance D."""
        self.assertGreater(
            self.metrics['path_length_cm'],
            self.metrics['straight_distance_cm']
        )

    def test_amplitude_reasonable_magnitude(self):
        """
        With A=0.5 cm oscillation, amplitude (std of perpendicular
        deviations) should be in the ballpark of A/sqrt(2) ~ 0.35.
        """
        self.assertGreater(self.metrics['amplitude_cm'], 0.1)
        self.assertLess(self.metrics['amplitude_cm'], 1.0)


class TestDriftingDescent(unittest.TestCase):
    """Particle drifts consistently to one side while falling."""

    def setUp(self):
        n = 100
        z = np.linspace(0, 20, n)  # 20 cm fall
        # Linear drift: x increases proportionally with z
        x = 0.3 * z  # drift slope = 0.3 (x/z)
        y = np.zeros(n)  # no y drift
        self.traj = _make_trajectory(x, y, z)
        self.d_eq = 0.5  # 5 mm particle
        self.metrics = compute_all_metrics(self.traj, self.d_eq)

    def test_drift_positive(self):
        """Consistent lateral drift -> drift > 0."""
        self.assertGreater(self.metrics['drift_cm'], 0)

    def test_drift_value(self):
        """
        x goes from 0 to 0.3*20 = 6 cm.  Linear fit is exact,
        so drift = 6.0 cm.
        """
        self.assertAlmostEqual(self.metrics['drift_cm'], 6.0, places=1)

    def test_drift_normalized(self):
        """drift* = 6.0 / 0.5 = 12.0."""
        self.assertAlmostEqual(self.metrics['drift_normalized'], 12.0, places=1)

    def test_drift_gradient_positive(self):
        """epsilon = drift / vertical_distance > 0."""
        self.assertGreater(self.metrics['drift_gradient'], 0)

    def test_drift_gradient_value(self):
        """epsilon = 6.0 / 20.0 = 0.3."""
        self.assertAlmostEqual(self.metrics['drift_gradient'], 0.3, places=2)

    def test_amplitude_near_zero(self):
        """
        Pure linear drift with no oscillation: all points on the fit line,
        so amplitude (std of perpendicular residuals) ~ 0.
        """
        self.assertAlmostEqual(self.metrics['amplitude_cm'], 0.0, places=4)

    def test_tortuosity_near_zero(self):
        """
        Straight drifting path (just at an angle): L ~ D, so
        tortuosity should be very small.
        """
        self.assertLess(self.metrics['tortuosity_pct'], 0.1)

    def test_settling_velocity_positive(self):
        """Particle is falling, w > 0."""
        self.assertGreater(self.metrics['settling_velocity_cm_s'], 0)

    def test_horizontal_drift_velocity_positive(self):
        """Consistent horizontal movement -> V > 0."""
        self.assertGreater(
            self.metrics['horizontal_drift_velocity_cm_s'], 0
        )


class TestEdgeCases(unittest.TestCase):
    """Edge cases and input validation."""

    def test_missing_columns_raises(self):
        """Missing required columns should raise ValueError."""
        df = pd.DataFrame({'time': [0, 1], 'x_cm': [0, 0]})
        with self.assertRaises(ValueError):
            compute_all_metrics(df, 0.5)

    def test_single_point_raises(self):
        """Single data point should raise ValueError."""
        df = pd.DataFrame({
            'time': [0], 'x_cm': [0], 'y_cm': [0],
            'z_cm': [0], 'vx': [0], 'vy': [0], 'vz': [0],
        })
        with self.assertRaises(ValueError):
            compute_all_metrics(df, 0.5)

    def test_zero_d_eq_raises(self):
        """Zero d_eq should raise ValueError."""
        df = pd.DataFrame({
            'time': [0, 1], 'x_cm': [0, 0], 'y_cm': [0, 0],
            'z_cm': [0, 1], 'vx': [0, 0], 'vy': [0, 0], 'vz': [1, 1],
        })
        with self.assertRaises(ValueError):
            compute_all_metrics(df, 0.0)

    def test_negative_d_eq_raises(self):
        """Negative d_eq should raise ValueError."""
        df = pd.DataFrame({
            'time': [0, 1], 'x_cm': [0, 0], 'y_cm': [0, 0],
            'z_cm': [0, 1], 'vx': [0, 0], 'vy': [0, 0], 'vz': [1, 1],
        })
        with self.assertRaises(ValueError):
            compute_all_metrics(df, -0.5)

    def test_two_points_minimum(self):
        """Two points should work without error."""
        df = pd.DataFrame({
            'time': [0.0, 0.02], 'x_cm': [0, 0], 'y_cm': [0, 0],
            'z_cm': [0, 1], 'vx': [0, 0], 'vy': [0, 0], 'vz': [50, 50],
        })
        metrics = compute_all_metrics(df, 0.5)
        self.assertIsInstance(metrics, dict)
        self.assertEqual(len(metrics), 11)


if __name__ == '__main__':
    unittest.main()
