# ML Models - Mikroplastik Settling Velocity

## En İyi Model ⭐

**`settling_velocity_nn.pth`** - BPNN (Neural Network)
- CV R² = **0.80**
- Veri: 966 deney (RESIN A=9 çıkarılmış)
- RMSE: ~1 cm/s

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `settling_velocity_nn.pth` | **En iyi model** (CV R²=0.80) ⭐ |
| `settling_velocity_nn_v2v3.pth` | V2/V3 eklendi (CV R²=0.64) - kullanma |
| `ml_neural_network.py` | BPNN eğitim scripti |
| `ml_full_model_v2.py` | RF + XGBoost baseline |
| `train_with_v2v3.py` | V2/V3 denemesi (başarısız) |
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

```python
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

# Model yükle
checkpoint = torch.load('settling_velocity_nn.pth')
model.load_state_dict(checkpoint['model_state'])

# Scaler
scaler = StandardScaler()
scaler.mean_ = checkpoint['scaler_mean']
scaler.scale_ = checkpoint['scaler_scale']

# Tahmin
X_scaled = scaler.transform(X_new)
velocity = model(torch.FloatTensor(X_scaled)).squeeze().detach().numpy()
```

## Notlar

- RESIN (A=9 R=4.5) verileri çıkarıldı (outlier)
- V2/V3 verileri dahil edilmedi (kalite sorunu)
- Teorik maksimum R² ≈ 0.87 (ölçüm belirsizliği nedeniyle)
- Model teorik limitin %92'sini yakalıyor
