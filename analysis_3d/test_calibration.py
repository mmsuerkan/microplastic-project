"""
Tests for pixel-to-cm calibration module.
"""
import sys
import os
import unittest
import math

sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis_3d.calibration import PixelCalibrator


class TestCalibrationFromVerticalPixels(unittest.TestCase):
    """Test calibrate_from_vertical_pixels and cm_per_pixel."""

    def test_known_calibration(self):
        """625 px = 28.5 cm  -->  cm_per_pixel = 0.0456."""
        cal = PixelCalibrator(column_height_cm=28.5)
        cal.calibrate_from_vertical_pixels(625)
        self.assertAlmostEqual(cal.cm_per_pixel, 28.5 / 625, places=6)
        self.assertAlmostEqual(cal.cm_per_pixel, 0.0456, places=4)

    def test_different_height(self):
        """Custom column height produces correct scale."""
        cal = PixelCalibrator(column_height_cm=30.0)
        cal.calibrate_from_vertical_pixels(600)
        self.assertAlmostEqual(cal.cm_per_pixel, 30.0 / 600, places=6)
        self.assertAlmostEqual(cal.cm_per_pixel, 0.05, places=6)

    def test_zero_pixels_raises(self):
        """Zero vertical_pixels must raise ValueError."""
        cal = PixelCalibrator()
        with self.assertRaises(ValueError):
            cal.calibrate_from_vertical_pixels(0)

    def test_negative_pixels_raises(self):
        """Negative vertical_pixels must raise ValueError."""
        cal = PixelCalibrator()
        with self.assertRaises(ValueError):
            cal.calibrate_from_vertical_pixels(-100)

    def test_uncalibrated_raises(self):
        """Accessing cm_per_pixel before calibration must raise RuntimeError."""
        cal = PixelCalibrator()
        with self.assertRaises(RuntimeError):
            _ = cal.cm_per_pixel


class TestPixelToCm(unittest.TestCase):
    """Test coordinate conversion pixel_to_cm."""

    def setUp(self):
        self.cal = PixelCalibrator(column_height_cm=28.5)
        self.cal.calibrate_from_vertical_pixels(625)
        # BSP-1 MAK SECOND reference points
        self.start_x, self.start_y = 537, 201
        self.end_x, self.end_y = 564, 826

    def test_start_point_is_zero(self):
        """Start point with ref_y = start_y should give z = 0."""
        x_cm, z_cm = self.cal.pixel_to_cm(
            self.start_x, self.start_y, ref_y=self.start_y
        )
        self.assertAlmostEqual(z_cm, 0.0, places=6)

    def test_end_point_z(self):
        """End point should be approximately 28.5 cm below start."""
        _, z_cm = self.cal.pixel_to_cm(
            self.end_x, self.end_y, ref_y=self.start_y
        )
        self.assertAlmostEqual(z_cm, 28.5, places=1)

    def test_horizontal_drift(self):
        """Horizontal drift: 27 px * 0.0456 cm/px ~ 1.23 cm."""
        x0, _ = self.cal.pixel_to_cm(
            self.start_x, self.start_y, ref_y=self.start_y
        )
        x1, _ = self.cal.pixel_to_cm(
            self.end_x, self.end_y, ref_y=self.start_y
        )
        drift_cm = x1 - x0
        expected_drift = 27 * (28.5 / 625)
        self.assertAlmostEqual(drift_cm, expected_drift, places=4)
        self.assertAlmostEqual(drift_cm, 1.232, places=2)

    def test_horizontal_uses_same_scale(self):
        """Horizontal scale factor equals vertical (square pixels)."""
        # 100 px horizontal should equal 100 px vertical in cm
        _, z100 = self.cal.pixel_to_cm(0, 100, ref_y=0)
        x100, _ = self.cal.pixel_to_cm(100, 0, ref_y=0)
        self.assertAlmostEqual(x100, z100, places=6)

    def test_default_ref_y_is_zero(self):
        """With default ref_y=0, z_cm equals y_px * cm_per_pixel."""
        x_cm, z_cm = self.cal.pixel_to_cm(100, 500)
        scale = self.cal.cm_per_pixel
        self.assertAlmostEqual(z_cm, 500 * scale, places=6)
        self.assertAlmostEqual(x_cm, 100 * scale, places=6)

    def test_negative_z_when_above_ref(self):
        """If y_px < ref_y, z_cm should be negative (particle above ref)."""
        _, z_cm = self.cal.pixel_to_cm(500, 100, ref_y=200)
        self.assertLess(z_cm, 0)

    def test_uncalibrated_conversion_raises(self):
        """pixel_to_cm on uncalibrated instance must raise RuntimeError."""
        cal = PixelCalibrator()
        with self.assertRaises(RuntimeError):
            cal.pixel_to_cm(100, 200)


class TestRecalibration(unittest.TestCase):
    """Test that recalibration updates the scale factor."""

    def test_recalibrate_changes_scale(self):
        """Calling calibrate_from_vertical_pixels again updates cm_per_pixel."""
        cal = PixelCalibrator(column_height_cm=28.5)
        cal.calibrate_from_vertical_pixels(625)
        first_scale = cal.cm_per_pixel

        cal.calibrate_from_vertical_pixels(500)
        second_scale = cal.cm_per_pixel

        self.assertNotAlmostEqual(first_scale, second_scale, places=6)
        self.assertAlmostEqual(second_scale, 28.5 / 500, places=6)


if __name__ == '__main__':
    unittest.main()
