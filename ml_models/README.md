# ML Models - Mikroplastik Settling Velocity

## En İyi Model ⭐

**Ensemble (RF + NN)** - CV R² = **0.84**
- Random Forest: CV R² = 0.844
- Neural Network: CV R² = 0.785
- Veri: 966 deney (RESIN outlier çıkarılmış)
- RMSE: ~0.89 cm/s
- Teorik limitin %95.6'sı

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `ensemble_rf.joblib` | Random Forest model ⭐ |
| `ensemble_nn.pth` | Neural Network model ⭐ |
| `ensemble_meta.joblib` | Meta bilgiler |
| `predict.py` | **Kolay tahmin scripti** ⭐ |
| `ensemble_model.py` | Ensemble eğitim scripti |
| `settling_velocity_nn.pth` | Sadece NN (CV R²=0.80) |
| `ml_data_full_v2.csv` | Eğitim verisi (1078 satır) |

## Model Mimarisi

```
Input (22 features)
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(64) → BatchNorm → ReLU
    ↓
Dense(1) → Output (velocity m/s)
```

## Features (22)

**Boyutlar:** a, b, c, d_eq, L, I, S
**Hacim/Yüzey:** volume, surface_area
**Şekil:** sphericity, aspect_ratio, CSF, flatness, elongation
**Yoğunluk:** density, delta_rho, rho_relative
**Physics:** D_star, v_stokes
**Kategorik:** cat_enc, mat_enc, view_enc

## Kullanım

### Basit Kullanım
```python
from predict import predict_velocity

# Tek parçacık tahmini (cm/s döner)
v = predict_velocity(a=3, b=3, c=3, density=1050)
print(f"Hız: {v:.2f} cm/s")
```

### Detaylı Kullanım
```python
from predict import EnsemblePredictor

predictor = EnsemblePredictor()
predictor.load()

# DataFrame ile tahmin
import pandas as pd
df = pd.DataFrame([...])
velocities = predictor.predict(df, method='ensemble')  # veya 'rf', 'nn'
```

## Notlar

- RESIN (A=9 R=4.5) verileri çıkarıldı (outlier)
- V2/V3 verileri dahil edilmedi (kalite sorunu)
- Teorik maksimum R² ≈ 0.87 (ölçüm belirsizliği nedeniyle)
- Model teorik limitin %92'sini yakalıyor
