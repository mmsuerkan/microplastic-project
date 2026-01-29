# Microplastic Settling Velocity Project

## Proje Durumu (2025-01-29)

### En İyi Model
**Balanced Ensemble (RF + NN) - R² = 0.859 ± 0.04**

```
Başlangıç: 0.77 → Final: 0.859 (+12% iyileşme)
Train-Test Gap: 0.06 (düşük overfitting riski)
```

**Model Dosyaları:** `ml_models/best_model/`

### Veri Kaynakları
- **Ana Excel:** `c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS-updated (1).xlsx`
- **WSP density:** `c:\Users\mmert\Downloads\ALL PARTICLES MEASUREMENTS-updated (3).xlsx`
- **Training Data:** `data/training_data_particle_avg.csv` (240 parçacık, ortalama alınmış)
- **Ham Data:** `data/training_data_v2_features.csv` (879 ölçüm)

### ML Deney Özeti

| Deney | Açıklama | R² | Sonuç |
|-------|----------|-----|-------|
| 1 | Baseline (5 feature) | 0.77 | - |
| 3 | Yüksek varyans çıkar | 0.81 | +0.04 ✓ |
| 4 | Volume/aspect_ratio ekle | 0.81 | aynı |
| 5 | Parçacık ortalaması | 0.83 | +0.02 ✓ |
| 6 | XGBoost/LightGBM | 0.78 | ❌ kötü |
| 7 | Stokes feature | 0.83 | aynı |
| 8 | Sample weighting | 0.83 | aynı |
| 9 | Hyperparameter tuning | 0.84 | +0.01 ✓ |
| 10 | Yeni veri + tuning (240 parçacık) | **0.858** | +0.02 ✓ |

### Final Model Parametreleri
**Ensemble:** RF 70% + NN 30%

**Random Forest (regularized):**
```python
n_estimators=100, max_depth=10, min_samples_leaf=4
```

**Neural Network (regularized):**
```python
hidden_layer_sizes=(128,), alpha=0.01,
max_iter=2000, early_stopping=True
```

**Model Dosyaları:** `ml_models/best_model/`
- `rf_model.joblib` - Random Forest model
- `nn_model.joblib` - Neural Network model
- `scaler.joblib` - StandardScaler
- `model_params.json` - Tüm parametreler

### Feature'lar (9 adet)
1. a, b, c (boyutlar mm)
2. density (kg/m³)
3. shape_enc (0-6: Cylinder, Half Cylinder, Cube, Wedge, Box, Sphere, Elliptic)
4. volume, surface_area, aspect_ratio, vol_surf_ratio

### Feature Importance
1. aspect_ratio: 0.263
2. a: 0.162
3. b: 0.115
4. vol_surf_ratio: 0.099

### Çıkarılan Parçacıklar (std > 3 cm/s)
- RESIN (a=6 r=3): C-6, SP-5, SP-6, SP-7
- RESIN (a=9 r=4.5): CUBE-1, SP-1, SP-6, SP-8

### Bilinen Limitasyonlar
- RESIN parçacıkları (büyük boyut: 7-9mm) zor tahmin ediliyor
- Model küçük parçacıkları (4mm) iyi öğrenmiş, büyüklere extrapolate zorlanıyor
- Çözüm: Daha fazla büyük parçacık verisi gerekli

### Önemli Dosyalar
- `ml_models/ML_EXPERIMENT_LOG.md` - Detaylı deney raporu
- `ml_models/best_hyperparameters.json` - En iyi parametreler
- `ml_models/train_particle_avg.py` - Ana eğitim scripti
- `ml_models/hyperparameter_tuning.py` - Parametre optimizasyonu
- `ml_models/error_analysis.py` - Hata analizi (Andrew Ng yaklaşımı)

### GitHub
- experiments.json, path görüntüleri (1241), tracking CSV'leri push edildi

---

## Veri İşleme Pipeline (2025-01-29)

### Klasör Yapısı
```
source_videos/                    # Ham videolar
├── IC CAPTURE 01.11.23(MAK)/
│   ├── FIRST/
│   │   ├── BSP/BSP-1/output_video.mp4
│   │   ├── C/C-2/output_video.mp4
│   │   └── ...
│   ├── SECOND/
│   └── THIRD/
└── ...

processed_results/                # İşlenmiş sonuçlar
├── success/                      # Başarılı deneyler
│   └── {tarih}/{view}/{repeat}/{kategori}/{kod}/
├── fail/                         # Başarısız deneyler
│   └── {sebep}/{tarih}/{view}/{repeat}/{kategori}/{kod}/
└── experiments.json              # Tüm deneylerin JSON kaydı
```

### Video İşleme Scriptleri
- `trackers/batch_processor_full.py` - Ana batch işleyici
- `trackers/auto_particle_tracker.py` - Parçacık takip algoritması
- `trackers/auto_particle_tracker_v3.py` - V3 (aralıklı frame tarama)
- `trackers/batch_tracker_v4.py` - Fail deneyleri yeniden işle

### PMMA Veri Durumu

**Excel vs Experiments.json Eşleştirme:**
- Excel'de PMMA: 200 parçacık (BSP:50, C:50, HC:50, WSP:50)
- Experiments.json kategori isimleri: BSP, C, HC, WSP (PMMA için)

**Mevcut Durum (2025-01-29):**
| Kategori | Excel | Success | İşlenmemiş Video |
|----------|-------|---------|------------------|
| BSP | 50 | 21 | 49 deney |
| C | 50 | 15 | 36 deney |
| HC | 50 | 17 | 54 deney |
| WSP | 50 | 20 | 56 deney |
| **TOPLAM** | **200** | **73** | **195 deney** |

### Eksik Veri Listesi
- `data/eksik_pmma_velocity_listesi.xlsx` - Lab için eksik parçacık listesi
- `data/eksik_pmma_velocity_listesi.csv` - CSV formatı
- `ml_models/export_missing_pmma.py` - Liste oluşturma scripti

### Batch Processing Çalıştırma
```bash
cd trackers
python batch_processor_full.py
```
- Otomatik olarak işlenmemiş videoları bulur
- Success/fail olarak sınıflandırır
- `processed_results/` altına kaydeder
- **NOT:** S3 upload ve H.264 dönüşüm hata verebilir (AWS yoksa), ana tracking çalışır

### Experiments.json Güncelleme
Batch processor sonrası experiments.json'a yeni deneyler eklenmeli:
- `ml_models/prepare_training_data.py` - Training data hazırla
- `ml_models/train_particle_avg.py` - Model eğit (parçacık ortalaması ile)

---

## TÜBİTAK 1001 Analizi (2025-01-29)

### Proje Dokümanı
- `basvuru_formu/1001_basvuru_formu-Genc-TEDU-Eylul-2022 (8).docx`

### İş Paketleri Durumu

| İP | Hedef | Mevcut Durum | Durum |
|----|-------|--------------|-------|
| İP-1 | MP parçacık hazırlama | 240 parçacık, 7 şekil | ✅ |
| İP-2 | EPE < 3-4 piksel | **0.47 piksel** | ✅ |
| İP-2 | CD-Re grafikleri | Yapıldı | ✅ |
| İP-3 | CFD simülasyonları | Kod yok | ❌ |
| İP-4 | RMSE < %0.2 | %13 (hedef çok agresif) | ⚠️ |
| İP-4 | MVAF 1.0±5% | 1.03 (3% hata) | ✅ |
| İP-4 | TVE < %5 | %17 | ❌ |

### EPE (End Point Error) Analizi
- **Klasör:** `epe_analysis/`
- **Sonuç:** Ortalama EPE = 0.47 piksel (Hedef: 3-4 piksel) ✅
- **Yöntem:** Centroid tracking ile displacement vektörleri (dX, dY)
- Magnitude = √(dX² + dY²) → "optik akış yoğunluğu" olarak kullanılabilir

**Dosyalar:**
- `epe_analysis/epe_analysis_particle.csv` - 240 parçacık EPE
- `epe_analysis/epe_analysis_visualization.png` - EPE grafikleri

### Re-CD Analizi
**Formüller:**
```python
# Reynolds sayısı
Re = ρf × V × d_eq / μ
# d_eq = (a × b × c)^(1/3)  # geometric mean

# Sürükleme katsayısı
CD = 4/3 × (ρp - ρf) × g × d / (ρf × V²)

# Sabitler
ρf = 998 kg/m³, μ = 0.001 Pa.s, g = 9.81 m/s²
```

**Sonuçlar:**
- Re aralığı: 41 - 965
- CD aralığı: 0.10 - 21.82
- Dosya: `epe_analysis/re_cd_analysis.csv`

### Göğüş et al. (2001) Şekil Faktörü
**Referans:** ASCE J. Hydraulic Engineering, Vol. 127, No. 10

**Formüller (Table 3):**
```python
# Transformed dimensions
Cube: a1=√2×l, b1=√2×l, c1=l
Cylinder: a1=√2×d, b1=√2×d, c1=h
Wedge: a1=√3×l, b1=2√(2/3)×l, c1=l/√2
Box: a1=√(l1²+l2²), b1=2l1l2/√(l1²+l2²), c1=l3

# Şekil faktörü (Eq. 8)
Ψ = ((a1+b1)/(2c1)) × (a1×b1×c1/V)

# Karakteristik uzunluk
L = √(a1² + b1²)

# Modified Reynolds & Drag Coefficient
R* = w × ρf × L / μf
CD* = 2×c1×(s-1)×g / w²

# Ampirik bağıntı (Eq. 12, Table 4)
Ψ×CD* = α × (R*)^β
```

**Şekil Faktörleri (Ψ):**
| Şekil | Bizim | Makale |
|-------|-------|--------|
| Cube | 3.34 | 2.83 |
| Cylinder | 4.90 | 3.60 |
| Wedge | 16.90 | 9.51 |
| Box | 5.25 | 4-8 |

**Dosya:** `epe_analysis/gogus_fig7_comparison.png`

### Tracking Sistemi
- **Yöntem:** Background subtraction + Centroid tracking
- **FPS:** 50
- **Kalibrasyon:** 28.5 cm kolon yüksekliği
- **Çıktılar:** dX, dY, Magnitude (px/frame)
- **NOT:** RAFT optik akış implement edilmemiş (TÜBİTAK planı), ama mevcut yöntem EPE hedefini karşılıyor
