"""
Tests for 3D trajectory reconstruction from MAK + ANG camera pairs.

Uses real tracking data: BSP-1 SECOND 01.11.23.
"""
import sys
import os
import unittest

sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis_3d.reconstruct_3d import reconstruct_trajectory_3d

# Paths to the real test data
MAK_CSV = os.path.join(
    PROJECT_ROOT,
    "processed_results", "success", "01.11.23", "MAK", "SECOND",
    "BSP", "BSP-1", "auto_tracking_results.csv",
)
ANG_CSV = os.path.join(
    PROJECT_ROOT,
    "processed_results", "success", "01.11.23", "ANG", "SECOND",
    "BSP", "BSP-1", "auto_tracking_results.csv",
)


class TestReconstructTrajectory3D(unittest.TestCase):
    """Integration tests using real BSP-1 data."""

    @classmethod
    def setUpClass(cls):
        """Run reconstruction once for all tests."""
        cls.df = reconstruct_trajectory_3d(MAK_CSV, ANG_CSV)

    # ------------------------------------------------------------------
    # Structure checks
    # ------------------------------------------------------------------

    def test_output_columns(self):
        """Output DataFrame must have the 7 required columns."""
        expected = {"time", "x_cm", "y_cm", "z_cm", "vx", "vy", "vz"}
        self.assertEqual(set(self.df.columns), expected)

    def test_length_above_50(self):
        """Reconstructed trajectory must have more than 50 data points."""
        self.assertGreater(len(self.df), 50)

    def test_no_nan_values(self):
        """There should be no NaN values in the output."""
        self.assertFalse(self.df.isna().any().any(),
                         f"NaN found in columns: {self.df.columns[self.df.isna().any()].tolist()}")

    # ------------------------------------------------------------------
    # Physical sanity checks
    # ------------------------------------------------------------------

    def test_z_mostly_increasing(self):
        """z_cm should be mostly increasing (particle settles downward).

        We allow up to 10% of steps to be non-increasing (noise / jitter).
        """
        dz = self.df["z_cm"].diff().dropna()
        frac_increasing = (dz > 0).mean()
        self.assertGreater(frac_increasing, 0.90,
                           f"Only {frac_increasing:.1%} of z steps are increasing")

    def test_z_starts_near_zero(self):
        """z_cm should start at approximately 0."""
        self.assertAlmostEqual(self.df["z_cm"].iloc[0], 0.0, places=1)

    def test_z_range_reasonable(self):
        """Total z range should be close to column height (28.5 cm).

        We accept 20-30 cm as reasonable range.
        """
        z_range = self.df["z_cm"].iloc[-1] - self.df["z_cm"].iloc[0]
        self.assertGreater(z_range, 20.0,
                           f"z range too small: {z_range:.1f} cm")
        self.assertLess(z_range, 35.0,
                        f"z range too large: {z_range:.1f} cm")

    def test_settling_velocity_range(self):
        """Mean vz should be in reasonable range for microplastics (1-15 cm/s)."""
        mean_vz = self.df["vz"].mean()
        self.assertGreater(mean_vz, 1.0,
                           f"Mean vz too low: {mean_vz:.2f} cm/s")
        self.assertLess(mean_vz, 15.0,
                        f"Mean vz too high: {mean_vz:.2f} cm/s")

    def test_horizontal_drift_small(self):
        """x_cm and y_cm drift should be small compared to z travel.

        Horizontal drift < 5 cm is reasonable for settling particles.
        """
        x_range = self.df["x_cm"].max() - self.df["x_cm"].min()
        y_range = self.df["y_cm"].max() - self.df["y_cm"].min()
        self.assertLess(x_range, 5.0,
                        f"x drift too large: {x_range:.2f} cm")
        self.assertLess(y_range, 5.0,
                        f"y drift too large: {y_range:.2f} cm")

    def test_x_y_start_near_zero(self):
        """x_cm and y_cm should start at approximately 0 (relative coords)."""
        self.assertAlmostEqual(self.df["x_cm"].iloc[0], 0.0, places=1)
        self.assertAlmostEqual(self.df["y_cm"].iloc[0], 0.0, places=1)

    def test_time_starts_at_zero(self):
        """Time should start at 0."""
        self.assertAlmostEqual(self.df["time"].iloc[0], 0.0, places=6)

    def test_time_monotonically_increasing(self):
        """Time must be strictly increasing."""
        dt = self.df["time"].diff().dropna()
        self.assertTrue((dt > 0).all(), "Time is not strictly increasing")

    def test_vx_vy_smaller_than_vz(self):
        """Horizontal velocities should generally be smaller than vertical."""
        mean_abs_vx = self.df["vx"].abs().mean()
        mean_abs_vy = self.df["vy"].abs().mean()
        mean_vz = self.df["vz"].mean()
        self.assertLess(mean_abs_vx, mean_vz,
                        f"|vx| mean ({mean_abs_vx:.2f}) >= vz mean ({mean_vz:.2f})")
        self.assertLess(mean_abs_vy, mean_vz,
                        f"|vy| mean ({mean_abs_vy:.2f}) >= vz mean ({mean_vz:.2f})")


class TestReconstructEdgeCases(unittest.TestCase):
    """Test edge cases and parameter variations."""

    def test_custom_n_grid(self):
        """Custom n_grid should produce the requested number of rows."""
        df = reconstruct_trajectory_3d(MAK_CSV, ANG_CSV, n_grid=100)
        self.assertEqual(len(df), 100)

    def test_custom_fps(self):
        """Custom FPS should affect time and velocity scales."""
        df_50 = reconstruct_trajectory_3d(MAK_CSV, ANG_CSV, fps=50, n_grid=100)
        df_25 = reconstruct_trajectory_3d(MAK_CSV, ANG_CSV, fps=25, n_grid=100)

        # Same z values (spatial reconstruction is independent of FPS)
        self.assertAlmostEqual(
            df_50["z_cm"].iloc[-1], df_25["z_cm"].iloc[-1], places=4
        )

        # df_25 has longer duration (same frames, half the rate)
        self.assertAlmostEqual(
            df_25["time"].iloc[-1], df_50["time"].iloc[-1] * 2, places=4
        )

    def test_missing_file_raises(self):
        """Non-existent CSV path should raise an error."""
        with self.assertRaises(Exception):
            reconstruct_trajectory_3d("nonexistent.csv", ANG_CSV)


if __name__ == "__main__":
    unittest.main()
