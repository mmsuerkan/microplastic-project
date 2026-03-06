# Microplastic Settling Velocity Prediction

Automated video tracking and machine learning pipeline for predicting settling velocities of microplastic particles in water.

## Project Overview

This project combines computer vision-based particle tracking with ensemble machine learning to predict terminal settling velocities of microplastic particles. Developed as part of a TUBiTAK 1001 research project.

**Key Results:**
- **ML Model:** Balanced Ensemble (RF 70% + NN 30%), R² = 0.859
- **Tracking Accuracy:** End Point Error (EPE) = 0.47 pixels (target: 3-4 px)
- **Dataset:** 240 particles, 7 shape types, 1103 successful experiments

## Repository Structure

```
├── trackers/                  # Video tracking pipeline
│   ├── auto_particle_tracker.py    # Core particle tracker (background subtraction + centroid)
│   ├── batch_processor_full.py     # Batch processing for all experiments
│   └── batch_tracker_v*.py         # Batch tracker iterations
│
├── ml_models/                 # Machine learning models
│   ├── best_model/                 # Final trained model files
│   │   ├── rf_model.joblib         # Random Forest model
│   │   ├── nn_model.joblib         # Neural Network model
│   │   ├── scaler.joblib           # StandardScaler
│   │   └── model_params.json       # Model parameters & feature list
│   ├── train_particle_avg.py       # Main training script
│   ├── hyperparameter_tuning.py    # Parameter optimization
│   ├── error_analysis.py           # Error analysis (Andrew Ng approach)
│   ├── predict.py                  # Prediction script
│   ├── ML_EXPERIMENT_LOG.md        # Detailed experiment log
│   └── utils/                      # Debug & verification scripts
│
├── epe_analysis/              # Tracking accuracy & fluid dynamics analysis
│   ├── epe_analysis_particle.csv   # EPE per particle
│   ├── re_cd_analysis.csv          # Reynolds number & drag coefficient
│   └── gogus_analysis.csv          # Gogus et al. shape factors
│
├── data/                      # Training data & datasets
│   ├── training_data_particle_avg.csv  # Main training data (240 particles)
│   └── training_data_v2_features.csv   # Raw measurements (879 observations)
│
├── reports/                   # Project reports & documentation
├── scripts/                   # Utility scripts (S3 sync, video conversion, etc.)
├── config/                    # AWS/CloudFront configuration files
├── experiment-viewer/         # Web-based experiment result viewer
├── basvuru_formu/             # TUBiTAK 1001 application form
└── archive/                   # Legacy tracker versions
```

## Features Used (9)

| Feature | Description | Importance |
|---------|-------------|------------|
| aspect_ratio | a/c ratio | 0.263 |
| a | Length (mm) | 0.162 |
| b | Width (mm) | 0.115 |
| vol_surf_ratio | Volume/Surface area | 0.099 |
| density | Particle density (kg/m³) | - |
| shape_enc | Shape encoding (0-6) | - |
| c | Height (mm) | - |
| volume | Particle volume (mm³) | - |
| surface_area | Surface area (mm²) | - |

**Shape types:** Cylinder, Half Cylinder, Cube, Wedge, Box, Sphere, Elliptic

## Tracking System

- **Method:** Background subtraction + Centroid tracking
- **Camera FPS:** 50
- **Calibration:** 28.5 cm column height
- **Output:** Frame-by-frame displacement vectors (dX, dY, Magnitude)

## Quick Start

### Train the model
```bash
cd ml_models
python train_particle_avg.py
```

### Run batch tracking
```bash
cd trackers
python batch_processor_full.py
```

### Make predictions
```bash
cd ml_models
python predict.py
```

## ML Experiment History

| Experiment | Description | R² |
|-----------|-------------|-----|
| 1 | Baseline (5 features) | 0.77 |
| 3 | Remove high variance | 0.81 |
| 5 | Particle averaging | 0.83 |
| 9 | Hyperparameter tuning | 0.84 |
| 10 | New data + tuning (240 particles) | **0.859** |

See [ML_EXPERIMENT_LOG.md](ml_models/ML_EXPERIMENT_LOG.md) for details.
