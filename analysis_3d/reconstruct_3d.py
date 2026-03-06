"""
3D trajectory reconstruction from MAK + ANG camera pairs.

MAK camera sees the X-Z plane (pixel X = world x, pixel Y = world z).
ANG camera sees the Y-Z plane (pixel X = world y, pixel Y = world z).
Both cameras share the vertical axis (z) through their pixel-Y coordinate.

Since the two cameras may have different frame counts, trajectories are
aligned by normalised z-position (0 to 1), interpolated onto a common
grid, and the z values are averaged (Lofty method).

Usage
-----
    df = reconstruct_trajectory_3d(mak_csv, ang_csv)
    # df columns: time, x_cm, y_cm, z_cm, vx, vy, vz
"""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Allow both direct execution and package import
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis_3d.calibration import PixelCalibrator


# ------------------------------------------------------------------
# Helper: read a tracking CSV and return calibrated cm arrays
# ------------------------------------------------------------------

def _read_and_calibrate(csv_path: str, column_height_cm: float = 28.5):
    """Read a tracking CSV and convert pixels to centimeters.

    Parameters
    ----------
    csv_path : str
        Path to ``auto_tracking_results.csv`` with columns
        Frame, X, Y, dX, dY, Magnitude.
    column_height_cm : float
        Physical height of the visible calibration region (cm).

    Returns
    -------
    horiz_cm : np.ndarray
        Horizontal position in cm, *relative* to the first frame
        (starts at 0).
    z_cm : np.ndarray
        Vertical (downward) position in cm, relative to the first
        frame (starts at 0, increases as the particle falls).
    n_frames : int
        Number of data rows (frames).
    """
    df = pd.read_csv(csv_path)

    x_px = df["X"].values.astype(float)
    y_px = df["Y"].values.astype(float)

    # Vertical pixel range for calibration
    vertical_pixels = y_px.max() - y_px.min()

    cal = PixelCalibrator(column_height_cm=column_height_cm)
    cal.calibrate_from_vertical_pixels(vertical_pixels)

    # Convert every point; use first Y pixel as ref so z starts at 0
    ref_y = y_px[0]
    horiz_cm = np.empty(len(x_px))
    z_cm = np.empty(len(x_px))
    for i in range(len(x_px)):
        h, z = cal.pixel_to_cm(x_px[i], y_px[i], ref_y=ref_y)
        horiz_cm[i] = h
        z_cm[i] = z

    # Make horizontal relative to start (subtract initial)
    horiz_cm = horiz_cm - horiz_cm[0]

    return horiz_cm, z_cm, len(df)


# ------------------------------------------------------------------
# Main reconstruction function
# ------------------------------------------------------------------

def reconstruct_trajectory_3d(
    mak_csv_path: str,
    ang_csv_path: str,
    column_height_cm: float = 28.5,
    fps: int = 50,
    n_grid: int | None = None,
) -> pd.DataFrame:
    """Reconstruct a 3D trajectory from MAK and ANG tracking CSVs.

    Parameters
    ----------
    mak_csv_path : str
        Path to the MAK camera tracking CSV.
    ang_csv_path : str
        Path to the ANG camera tracking CSV.
    column_height_cm : float
        Calibration column height in cm (default 28.5).
    fps : int
        Camera frame rate (default 50).
    n_grid : int or None
        Number of points in the common z grid.  If *None*, the average
        frame count of the two cameras is used.

    Returns
    -------
    pd.DataFrame
        Columns: ``time``, ``x_cm``, ``y_cm``, ``z_cm``,
        ``vx``, ``vy``, ``vz``.
        ``z_cm`` starts at 0 and increases downward.
        ``x_cm`` and ``y_cm`` are relative to the starting position.
    """
    # 1. Read & calibrate each camera independently
    x_mak, z_mak, n_mak = _read_and_calibrate(mak_csv_path, column_height_cm)
    y_ang, z_ang, n_ang = _read_and_calibrate(ang_csv_path, column_height_cm)

    # 2. Normalise z to [0, 1] for alignment.
    #    Both cameras observe the same vertical settling event, so
    #    normalised z serves as a common "progress" variable that is
    #    independent of each camera's frame count.
    z_mak_norm = z_mak / z_mak[-1] if z_mak[-1] != 0 else z_mak
    z_ang_norm = z_ang / z_ang[-1] if z_ang[-1] != 0 else z_ang

    # 3. Build interpolators parameterised by normalised z.
    #    z_norm must be strictly increasing for interp1d; tracking jitter
    #    can cause tiny reversals so we enforce monotonicity.
    def _monotonic_interp(z_norm, values):
        """Return interp1d after enforcing strict monotonicity in z_norm."""
        mask = np.ones(len(z_norm), dtype=bool)
        for i in range(1, len(z_norm)):
            if z_norm[i] <= z_norm[i - 1]:
                mask[i] = False
        z_clean = z_norm[mask]
        v_clean = values[mask]
        if len(z_clean) < 2:
            raise ValueError("Not enough monotonic z points for interpolation")
        return interp1d(z_clean, v_clean, kind="linear", fill_value="extrapolate")

    # MAK gives x(z_norm), z_mak(z_norm), and frame_index_mak(z_norm)
    mak_frames = np.arange(n_mak, dtype=float)
    interp_x = _monotonic_interp(z_mak_norm, x_mak)
    interp_z_mak = _monotonic_interp(z_mak_norm, z_mak)
    interp_frame_mak = _monotonic_interp(z_mak_norm, mak_frames)

    # ANG gives y(z_norm), z_ang(z_norm), and frame_index_ang(z_norm)
    ang_frames = np.arange(n_ang, dtype=float)
    interp_y = _monotonic_interp(z_ang_norm, y_ang)
    interp_z_ang = _monotonic_interp(z_ang_norm, z_ang)
    interp_frame_ang = _monotonic_interp(z_ang_norm, ang_frames)

    # 4. Common normalised-z grid
    if n_grid is None:
        n_grid = int(round((n_mak + n_ang) / 2))
    z_common = np.linspace(0, 1, n_grid)

    # 5. Interpolate all quantities onto the common grid
    x_cm = interp_x(z_common)
    y_cm = interp_y(z_common)
    z_from_mak = interp_z_mak(z_common)
    z_from_ang = interp_z_ang(z_common)

    # 6. Average z from both cameras (Lofty method)
    z_cm = (z_from_mak + z_from_ang) / 2.0

    # 7. Estimate time from FPS and average frame count.
    #    Each grid point corresponds to a specific "progress" through the
    #    settling event.  We map that progress back to physical time by
    #    averaging the frame-index from both cameras and dividing by FPS.
    frame_mak = interp_frame_mak(z_common)
    frame_ang = interp_frame_ang(z_common)
    avg_frame = (frame_mak + frame_ang) / 2.0
    time = avg_frame / fps

    # 8. Calculate velocities using np.gradient (cm/s)
    vx = np.gradient(x_cm, time)
    vy = np.gradient(y_cm, time)
    vz = np.gradient(z_cm, time)

    # 9. Build output DataFrame
    result = pd.DataFrame({
        "time": time,
        "x_cm": x_cm,
        "y_cm": y_cm,
        "z_cm": z_cm,
        "vx": vx,
        "vy": vy,
        "vz": vz,
    })

    return result


# ------------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mak_csv = os.path.join(
        project_root,
        "processed_results", "success", "01.11.23", "MAK", "SECOND",
        "BSP", "BSP-1", "auto_tracking_results.csv",
    )
    ang_csv = os.path.join(
        project_root,
        "processed_results", "success", "01.11.23", "ANG", "SECOND",
        "BSP", "BSP-1", "auto_tracking_results.csv",
    )

    df = reconstruct_trajectory_3d(mak_csv, ang_csv)
    print(f"Satir sayisi: {len(df)}")
    print(f"Sutunlar: {list(df.columns)}")
    print(f"\nIlk 5 satir:")
    print(df.head().to_string(index=False))
    print(f"\nSon 5 satir:")
    print(df.tail().to_string(index=False))
    print(f"\nz_cm aralik: {df['z_cm'].min():.2f} - {df['z_cm'].max():.2f} cm")
    print(f"vz ortalama: {df['vz'].mean():.2f} cm/s")
    print(f"vz aralik: {df['vz'].min():.2f} - {df['vz'].max():.2f} cm/s")
