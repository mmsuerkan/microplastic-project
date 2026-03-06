"""
Pixel-to-centimeter calibration for the settling column.

The settling column is cylindrical (inner diameter 30 cm) with a calibrated
visible height of 28.5 cm. Both cameras capture the same column, so the
vertical pixel span corresponding to 28.5 cm gives us the scale factor.
Square pixels are assumed (same scale horizontal and vertical).

Usage
-----
    cal = PixelCalibrator(column_height_cm=28.5)
    cal.calibrate_from_vertical_pixels(625)   # 625 px span = 28.5 cm
    x_cm, z_cm = cal.pixel_to_cm(564, 826, ref_y=201)
"""
import sys

sys.stdout.reconfigure(encoding='utf-8')


class PixelCalibrator:
    """Convert pixel coordinates to real-world centimeters.

    Parameters
    ----------
    column_height_cm : float
        Physical height of the visible calibration region in cm (default 28.5).
    """

    def __init__(self, column_height_cm: float = 28.5):
        self.column_height_cm = column_height_cm
        self._cm_per_pixel: float | None = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_from_vertical_pixels(self, vertical_pixels: float) -> None:
        """Set the scale factor from a known vertical pixel span.

        Parameters
        ----------
        vertical_pixels : float
            Number of pixels corresponding to ``column_height_cm``.
            Must be positive.

        Raises
        ------
        ValueError
            If *vertical_pixels* is not positive.
        """
        if vertical_pixels <= 0:
            raise ValueError(
                f"vertical_pixels must be positive, got {vertical_pixels}"
            )
        self._cm_per_pixel = self.column_height_cm / vertical_pixels

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cm_per_pixel(self) -> float:
        """Scale factor (cm per pixel).

        Raises
        ------
        RuntimeError
            If calibration has not been performed yet.
        """
        if self._cm_per_pixel is None:
            raise RuntimeError(
                "Calibration has not been performed. "
                "Call calibrate_from_vertical_pixels() first."
            )
        return self._cm_per_pixel

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def pixel_to_cm(
        self, x_px: float, y_px: float, ref_y: float = 0
    ) -> tuple[float, float]:
        """Convert pixel coordinates to centimeters.

        Parameters
        ----------
        x_px : float
            Horizontal pixel position.
        y_px : float
            Vertical pixel position (increases downward in the image).
        ref_y : float
            Reference vertical pixel position (the "zero" for z). Typically
            the Y coordinate of the first tracked point so that the particle
            starts at z = 0.

        Returns
        -------
        (x_cm, z_cm) : tuple[float, float]
            *x_cm* — horizontal position in cm, measured from the left edge
            of the frame (x_px = 0).
            *z_cm* — downward settling distance in cm relative to *ref_y*.
            Positive z means the particle has moved downward.

        Notes
        -----
        The same ``cm_per_pixel`` is used for both axes (square-pixel
        assumption).
        """
        scale = self.cm_per_pixel  # may raise RuntimeError
        x_cm = x_px * scale
        z_cm = (y_px - ref_y) * scale
        return x_cm, z_cm


if __name__ == '__main__':
    # Quick demo with BSP-1 MAK SECOND data
    cal = PixelCalibrator(column_height_cm=28.5)
    cal.calibrate_from_vertical_pixels(625)

    print(f"cm_per_pixel = {cal.cm_per_pixel:.4f}")

    start_x, start_y = 537, 201
    end_x, end_y = 564, 826

    x0, z0 = cal.pixel_to_cm(start_x, start_y, ref_y=start_y)
    x1, z1 = cal.pixel_to_cm(end_x, end_y, ref_y=start_y)

    print(f"Baslangic: ({x0:.2f} cm, {z0:.2f} cm)")
    print(f"Bitis:     ({x1:.2f} cm, {z1:.2f} cm)")
    print(f"Yatay drift: {x1 - x0:.2f} cm")
    print(f"Dusey mesafe: {z1:.2f} cm")
