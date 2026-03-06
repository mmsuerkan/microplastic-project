# Design: 3D Settling Dynamics & ML Prediction Paper

**Date:** 2026-03-06
**Target Journal:** Water Research / Marine Pollution Bulletin
**Status:** Approved

## Title

"Three-dimensional settling dynamics and machine learning-based velocity prediction of microplastic particles with controlled geometries"

## Novelty (vs Lofty et al. 2026, EST)

| Aspect | Lofty et al. 2026 | This Study |
|--------|-------------------|------------|
| Particles | 127 environmental, no density | **240 controlled, known density** |
| Shapes | 4 types (Zingg classification) | **7 types (cylinder, cube, wedge, box, sphere, elliptic, half cylinder)** |
| Repeats | Most tested once | **3 repeats per particle** |
| Column | Rectangular tank (20x20x70 cm) | **Cylindrical column (dia 30 cm)** - symmetric wall effects |
| 3D Tracking | 2 cameras, 90 deg, 60 FPS | **2 cameras (MAK+ANG), 90 deg, 50 FPS** |
| ML Prediction | None | **Ensemble RF+NN, R^2=0.859** |
| Dataset | 127 trajectories | **348 paired MAK+ANG trajectories (696 videos)** |
| Density | Not measured (acknowledged limitation) | **Known for all particles (4 polymers)** |

## Key Methodological Advantages

1. **Density is known** - Lofty explicitly states this as a limitation. We have it for all particles.
2. **Controlled geometry** - Isolates shape-velocity relationship without environmental confounds.
3. **Statistical rigor** - 3 repeats per particle enables variance quantification.
4. **ML prediction** - Goes beyond observation to prediction (no ML in Lofty).
5. **Cylindrical column** - Symmetric wall effects vs rectangular tank.

## Technical Work Required

### A. 3D Trajectory Reconstruction (new code)

- Synchronize MAK and ANG frames for 348 paired experiments
- Extract per-frame coordinates: MAK gives (x, z), ANG gives (y, z)
- Combine into real (x, y, z) using averaged z from both cameras
- Pixel-to-cm calibration using cylindrical column geometry (dia 30 cm, height 28.5 cm)
- Apply LOWESS smoothing (as in Lofty) or use existing centroid data

### B. New Trajectory Metrics (Lofty Equations)

From Lofty et al. 2026 Section 2.4:

- **Settling velocity:** w = <v_zi> (ensemble-averaged vertical velocity)
- **Horizontal drift velocity:** V = <sqrt(v_xi^2 + v_yi^2)>
- **Drift:** delta = sqrt((x'-x0)^2 + (y'-y0)^2), normalized: delta* = delta / D_eq
- **Amplitude:** sigma = mean std of lateral deviations from average trajectory, normalized: sigma* = sigma / D_eq
- **Drift gradient:** epsilon = delta / (z0 - z*)
- **Tortuosity:** phi = ((L - D) / D) * 100, where L = actual path length, D = straight-line distance
- **Particle Reynolds number:** Re_p = w * D_eq / nu

### C. ML Model Update

- Keep existing ensemble model (R^2=0.859) as baseline
- Add 3D features (drift, tortuosity, amplitude) and test improvement
- Report feature importance with and without 3D metrics

## Paper Structure

### 1. Introduction (~800 words)
- MP transport problem and settling velocity importance
- Gap: most studies 1D, no controlled 3D studies with known density
- Lofty et al. 2026 contribution and remaining gaps
- This study's contributions: (i) controlled 3D tracking, (ii) known density, (iii) ML prediction

### 2. Materials & Methods (~1500 words)
- 2.1 Particle preparation (7 shapes, 4 polymers: PMMA, PA6, RESIN, WSP; dimensions a,b,c)
- 2.2 Settling column setup (cylindrical dia 30 cm, 2 cameras at 90 deg, 50 FPS, 28.5 cm calibration)
- 2.3 Particle tracking algorithm (background subtraction + centroid detection, automatic)
- 2.4 3D trajectory reconstruction (MAK+ANG frame sync, pixel-to-cm, coordinate merging)
- 2.5 Trajectory metrics (w, V, delta*, sigma*, epsilon, phi - adapted from Lofty)
- 2.6 ML model (Ensemble RF 70% + NN 30%, 9 features, 5-fold CV)

### 3. Results & Discussion (~2500 words)
- 3.1 Particle properties & shape classification (CSF, flatness, elongation)
- 3.2 3D settling trajectories (drift and tortuosity by shape type)
- 3.3 Velocity analysis (vertical vs horizontal, Re-CD relationships)
- 3.4 ML prediction performance & feature importance
- 3.5 Comparison with literature (Lofty et al. 2026, Gogus et al. 2001)

### 4. Conclusions (~400 words)

## Expected Figures

1. **Experimental setup** - Schematic of cylindrical column + 2 camera positions
2. **3D trajectories** - Example trajectories colored by shape type (like Lofty Fig 3)
3. **Drift & tortuosity vs shape** - Box plots or violin plots per shape category
4. **Re-CD diagram** - With Gogus et al. comparison curves
5. **ML prediction** - Predicted vs measured scatter + feature importance bar chart
6. **Literature comparison** - Our metrics vs Lofty's environmental MPs

## Data Available

- **experiments.json:** 1562 experiments (1103 success, 459 fail)
- **348 paired MAK+ANG** successful experiments
- **tracking CSVs:** frame-by-frame (dX, dY, magnitude) per experiment
- **training_data_particle_avg.csv:** 240 particles with features
- **epe_analysis/:** EPE, Re-CD, Gogus shape factors already computed
- **ml_models/best_model/:** Trained RF+NN ensemble

## Key References

- Lofty et al. (2026) EST - 3D settling dynamics of environmental MPs
- Gogus et al. (2001) ASCE J. Hydraulic Eng. - Shape factors for settling
- Zingg (1935) - Shape classification (sphere, rod, disk, blade)
- Russell et al. (2025) - Zingg adapted for plastics
