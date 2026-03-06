# ML Model Deney Raporu
**Proje:** Microplastic Settling Velocity Prediction
**Tarih:** 2025-01-23

---

## Veri Seti Bilgisi
- **Kaynak:** ALL PARTICLES MEASUREMENTS-updated (1).xlsx + WSP density'leri (3).xlsx
- **Toplam Satır:** 859
- **Unique Parçacık:** 236
- **Şekiller:** 7 kategori

| Şekil | Adet | Velocity (cm/s) |
|-------|------|-----------------|
| Cylinder | 273 | 6.04 ± 2.02 |
| Cube | 195 | 5.85 ± 1.99 |
| Half Cylinder | 113 | 3.94 ± 1.08 |
| Box Shape Prism | 86 | 5.42 ± 1.12 |
| Elliptic Cylinder | 76 | 3.36 ± 0.65 |
| Sphere | 60 | 11.60 ± 2.81 |
| Wedge Shape Prism | 56 | 3.67 ± 0.90 |

---

## Deney 1: Baseline Model (5 Feature)
**Tarih:** 2025-01-23
**Features:** a, b, c, density, shape_enc
**Veri:** 859 satır (PA6 BSP-1 density düzeltildi: 11465 → 1146.52)

| Model | CV R² | Test R² | RMSE |
|-------|-------|---------|------|
| Random Forest | 0.77 ± 0.04 | 0.77 | - |
| Neural Network | 0.75 ± 0.03 | 0.76 | - |
| Ensemble | 0.76 ± 0.03 | 0.77 | 1.21 cm/s |

**Notlar:**
- Önceki model 0.84 R² skoruna ulaşmıştı
- Feature korelasyonları düşük (a: +0.41, density: +0.11)
- RESIN Sphere'lerde yüksek varyans (std > 5 cm/s)

---

## Deney 2: Sphere Çıkarıldı
**Tarih:** 2025-01-23
**Veri:** 793 satır (Sphere hariç)

| Model | CV R² | Test R² | RMSE |
|-------|-------|---------|------|
| Random Forest | 0.72 ± 0.04 | 0.71 | - |
| Ensemble | 0.70 ± 0.04 | 0.70 | 1.07 cm/s |

**Sonuç:** Performans DÜŞTÜ. Sphere'ler modele katkı sağlıyor.

---

## Veri Analizi Bulguları

### Feature Korelasyonları (velocity ile)
- a: +0.407 (en iyi)
- b: -0.113
- c: +0.073
- density: +0.107
- shape_enc: +0.046

### Yüksek Varyanslı Parçacıklar (std > 1.5 cm/s)
22 parçacık - çoğu RESIN Sphere:
- RESIN SP-6: 12.68 ± 5.71 cm/s
- RESIN CUBE-1: 7.88 ± 4.54 cm/s
- RESIN SP-1: 10.88 ± 4.16 cm/s

### Outlier'lar
48 satır (5.6%) - velocity > 11 cm/s (çoğu RESIN Sphere)

---

## Deney 3: Yüksek Varyanslı Parçacıklar Çıkarıldı
**Tarih:** 2025-01-23
**Veri:** 827 satır (std > 3 cm/s olan 8 parçacık / 32 satır çıkarıldı)

**Çıkarılan parçacıklar:**
- RESIN (a=6 r=3): C-6, SP-5, SP-6, SP-7
- RESIN (a=9 r=4.5): CUBE-1, SP-1, SP-6, SP-8

| Model | CV R² | Değişim |
|-------|-------|---------|
| Random Forest | 0.81 ± 0.05 | +0.04 |
| Neural Network | 0.78 ± 0.05 | +0.03 |
| Ensemble | 0.80 ± 0.04 | +0.04 |

**Sonuç:** Performans ARTTI! Yüksek varyanslı veriler modeli bozuyordu.

---

## Deney 4: Ek Feature'lar Eklendi
**Tarih:** 2025-01-23
**Features (9):** a, b, c, density, shape_enc, volume, surface_area, aspect_ratio, vol_surf_ratio

**Yeni feature korelasyonları:**
- aspect_ratio: +0.60 (en iyi)
- surface_area: +0.47
- volume: +0.46
- vol_surf_ratio: +0.43

| Model | CV R² | Değişim |
|-------|-------|---------|
| Random Forest | 0.81 ± 0.05 | aynı |
| Neural Network | 0.78 ± 0.05 | aynı |
| Ensemble | 0.80 ± 0.05 | aynı |

**Feature Importance (RF):**
1. aspect_ratio: 0.263
2. b: 0.159
3. shape_enc: 0.128
4. density: 0.087
5. volume: 0.087

**Sonuç:** Skor artmadı ama aspect_ratio en önemli feature oldu.

---

## Deney 5: Parçacık Bazlı Ortalama Velocity
**Tarih:** 2025-01-23
**Veri:** 228 unique parçacık (her parçacığın ölçüm ortalaması alındı)
**Features (9):** a, b, c, density, shape_enc, volume, surface_area, aspect_ratio, vol_surf_ratio

**Veri dönüşümü:**
- 827 satır → 228 parçacık
- Ortalama ölçüm sayısı: 3.6 ölçüm/parçacık
- Tekrarlı ölçümler tek velocity değerine indirgendi

**Şekil dağılımı (parçacık bazlı):**
| Şekil | Parçacık | Ortalama Velocity |
|-------|----------|-------------------|
| Cylinder | 74 | 5.81 cm/s |
| Cube | 47 | 5.97 cm/s |
| Half Cylinder | 36 | 3.93 cm/s |
| Box Shape Prism | 25 | 5.21 cm/s |
| Wedge Shape Prism | 20 | 3.45 cm/s |
| Elliptic Cylinder | 16 | 3.33 cm/s |
| Sphere | 10 | 12.93 cm/s |

| Model | CV R² | Değişim |
|-------|-------|---------|
| Random Forest | 0.81 ± 0.09 | aynı |
| Neural Network | 0.83 ± 0.05 | +0.05 |
| **Ensemble** | **0.83 ± 0.06** | **+0.03** |

**Fold bazlı sonuçlar:**
| Fold | RF | NN | Ensemble |
|------|-----|-----|----------|
| 1 | 0.84 | 0.77 | 0.84 |
| 2 | 0.66 | 0.79 | 0.73 |
| 3 | 0.87 | 0.86 | 0.87 |
| 4 | 0.91 | 0.91 | 0.92 |
| 5 | 0.76 | 0.83 | 0.81 |

**Feature Importance (RF):**
1. aspect_ratio: 0.263
2. a: 0.162
3. b: 0.115
4. vol_surf_ratio: 0.099
5. surface_area: 0.084

**Sonuç:**
- Veri 3.6x azaldı (827 → 228) ama performans korundu
- Ensemble 0.80 → 0.83 (+0.03 artış!)
- NN özellikle iyi performans gösterdi (0.78 → 0.83)
- Veri: `data/training_data_particle_avg.csv`

---

## Deney 6: XGBoost ve LightGBM
**Tarih:** 2025-01-23
**Veri:** 228 parçacık (parçacık bazlı ortalama)
**Features (9):** a, b, c, density, shape_enc, volume, surface_area, aspect_ratio, vol_surf_ratio

| Model | CV R² | Notlar |
|-------|-------|--------|
| Random Forest | 0.81 ± 0.09 | Baseline |
| XGBoost (d=5) | 0.77 ± 0.08 | Default |
| XGBoost (d=3) | 0.79 ± 0.09 | Tuned (en iyi) |
| LightGBM | 0.78 ± 0.07 | Default |

**XGBoost Hiperparametre Denemeleri:**
| n_est | depth | lr | CV R² |
|-------|-------|-----|-------|
| 100 | 3 | 0.1 | **0.79** |
| 100 | 5 | 0.1 | 0.77 |
| 100 | 7 | 0.1 | 0.72 |
| 200 | 5 | 0.05 | 0.76 |

**Feature Importance (XGBoost):**
1. b: 0.273
2. shape_enc: 0.203
3. volume: 0.195
4. vol_surf_ratio: 0.106

**Sonuç:**
- XGBoost/LightGBM, Random Forest'tan **KÖTÜ** performans gösterdi
- Veri az (228 satır) → gradient boosting overfitting yapıyor
- RF'nin bagging yaklaşımı küçük veri setleri için daha uygun
- **En iyi model hala Deney 5 Ensemble (0.83)**

---

## Deney 7: Stokes-Inspired Feature'lar
**Tarih:** 2025-01-23
**Amaç:** Fiziksel bilgiyi (Stokes Law) feature olarak eklemek

**Eklenen Feature'lar:**
| Feature | Formül | Korelasyon |
|---------|--------|------------|
| stokes_factor | (ρ - 1000) × a² | +0.52 |
| equiv_diameter | Eşdeğer çap | +0.54 |
| buoyancy_factor | (ρ - 1000) / 1000 | +0.09 |
| size_squared | a² | +0.50 |

**Sonuçlar:**
| Feature Set | Ensemble R² | Değişim |
|-------------|-------------|---------|
| Baseline (9 feat) | **0.83** | - |
| Stokes Added (11 feat) | 0.83 | -0.003 |
| Stokes Only (6 feat) | 0.67 | -0.16 |
| Best Physics (8 feat) | 0.79 | -0.04 |

**Sonuç:** Stokes feature'ları **İYİLEŞTİRME SAĞLAMADI**

**Neden?**
- RF zaten `a` ve `density` ile non-linear ilişkileri öğrenmiş
- `stokes_factor = f(a, density)` → redundant bilgi
- Andrew Ng: "Model zaten öğrenmişse, explicit feature gereksiz"

---

## Error Analysis (Andrew Ng Yaklaşımı)
**Tarih:** 2025-01-23

### Bulgular:
1. **RESIN parçacıkları en zor** (MAE: 2.82 cm/s)
2. **Sphere şekli en zor** (MAE: 3.82 cm/s)
3. **Box Shape Prism en kolay** (MAE: 0.50 cm/s)

### RESIN Analizi:
- RESIN boyutu: 7.6 mm (diğerleri: 4.3 mm)
- Büyük boyut → Yüksek velocity (Stokes Law)
- **ÖLÇÜM HATASI DEĞİL** - Fiziksel olarak tutarlı

### Problem:
- Model küçük parçacıkları (4mm) öğrenmiş
- Büyük parçacıklara (7-9mm) extrapolate edemiyor
- Çözüm: Daha fazla büyük parçacık verisi gerekli

---

## Deney 8: Sample Weighting
**Tarih:** 2025-01-23
**Amaç:** RESIN parçacıklarına daha fazla ağırlık vererek performans artırmak

| RESIN Weight | CV R² | RESIN MAE | Diğer MAE |
|--------------|-------|-----------|-----------|
| 1.0x | 0.807 | 0.69 | 0.74 |
| 1.5x | 0.805 | 0.68 | 0.75 |
| 2.0x | 0.805 | 0.69 | 0.75 |
| 5.0x | 0.807 | 0.69 | 0.75 |

**Sonuç:** İYİLEŞME YOK - Model zaten elinden gelenin en iyisini yapıyor.
Sample weighting, veri eksikliğini çözmez.

---

## Deney 9: Hyperparameter Tuning
**Tarih:** 2025-01-23
**Amaç:** RF ve NN parametrelerini optimize etmek

**En İyi RF Parametreleri:**
```
n_estimators=200, max_depth=20, min_samples_split=10,
min_samples_leaf=2, max_features='log2'
```

**En İyi NN Parametreleri:**
```
hidden_layer_sizes=(64, 32), alpha=0.001,
learning_rate_init=0.001, activation='relu'
```

| Model | Baseline | Tuned | Değişim |
|-------|----------|-------|---------|
| Ensemble | 0.8336 | **0.8415** | **+0.8%** |

**En iyi ensemble ağırlığı:** RF 50% + NN 50%

**Sonuç:** +0.8% iyileşme sağlandı ✓
Parametreler: `ml_models/best_hyperparameters.json`

---

## Sonraki Adımlar (TODO)
- [x] Deney 5: Parçacık bazlı ortalama velocity ✓
- [x] Deney 6: XGBoost/LightGBM dene ✓
- [x] Deney 7: Stokes feature'lar ✓ (iyileşme yok)
- [x] Deney 8: Sample weighting ✓ (iyileşme yok)
- [x] Deney 9: Hyperparameter tuning ✓ (+0.8%)
- [x] Deney 3: Yüksek varyanslı parçacıkları çıkar ✓
- [x] Deney 4: Volume feature ekle ✓

---

## En İyi Model
**Şu anki en iyi:** Deney 9 - Tuned Ensemble (R² = 0.84 ± 0.04)
- Parçacık bazlı ortalama ile 228 satır
- 9 feature: a, b, c, density, shape_enc, volume, surface_area, aspect_ratio, vol_surf_ratio
- **RF 50% + NN 50%** ağırlık
- RF: n_estimators=200, max_depth=20, min_samples_split=10
- NN: hidden_layers=(64,32), alpha=0.001
