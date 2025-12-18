# ML Model ve Teorik Karşılaştırma Raporu

**Proje:** TÜBİTAK 1001 - Mikroplastik Çökelme Hızı Tahmini (İP-4)
**Tarih:** 2025-12-15
**Hazırlayan:** Claude Code (Otomatik Analiz)

---

## 1. Özet

Bu rapor, mikroplastik parçacıkların çökelme hızı tahmininde iki farklı yaklaşımı değerlendirmektedir:

1. **Makine Öğrenmesi Modeli**: Parçacık özelliklerinden (boyut, şekil, yoğunluk) hız tahmini
2. **Teorik Karşılaştırma**: Ölçülen hızlar ile İpek'in teorik hesaplamalarının karşılaştırması

### Temel Bulgular

| Metrik | Değer |
|--------|-------|
| ML Model CV R² | **0.76** |
| Teorik Korelasyon (Yeni Tracker) | **0.91** |
| Toplam Deney Sayısı | 1078 |
| Benzersiz Parçacık | ~200 |
| Malzeme Çeşidi | 10 |

---

## 2. Makine Öğrenmesi Modeli

### 2.1 Veri Hazırlığı

**Veri Kaynakları:**
- `ALL PARTICLES MEASUREMENTS.xlsx`: Ölçülmüş parçacık boyutları ve yoğunlukları
- `processed_results/success/*/summary.csv`: Tracker'dan hesaplanan hızlar
- `Video_Boyut_Eslestirme_FINAL.xlsx`: Video-boyut eşleştirmeleri

**Veri İstatistikleri:**
```
Toplam Deney: 1078 (başarılı eşleşme: %87)
Hız Aralığı: 0.0001 - 0.25 m/s
Ortalama Hız: 0.042 ± 0.027 m/s
Yoğunluk Aralığı: 1000 - 1400 kg/m³
```

**Malzeme Dağılımı:**
| Malzeme | Deney Sayısı |
|---------|--------------|
| Plexiglass | 358 |
| ABS | 281 |
| PLA | 234 |
| PS | 62 |
| RESIN | 88 |
| PA 6 | 55 |

### 2.2 Feature Engineering

**Temel Boyutlar:**
- `a`, `b`, `c`: Ham boyutlar (mm)
- `L`, `I`, `S`: Sıralı boyutlar (L > I > S)

**Türetilen Özellikler:**
- `d_eq`: Eşdeğer çap = (a × b × c)^(1/3)
- `volume`: Hacim = a × b × c
- `surface_area`: Yüzey alanı = 2(ab + bc + ac)
- `sphericity`: Küresellik = π^(1/3) × (6V)^(2/3) / A
- `aspect_ratio`: En/boy oranı = L / S
- `CSF`: Corey Shape Factor = S / √(L × I)
- `delta_rho`: Yoğunluk farkı = ρ_parçacık - ρ_su

**Kategorik Kodlamalar:**
- `cat_enc`: Kategori (şekil) encoding
- `mat_enc`: Malzeme encoding
- `view_enc`: Görüş açısı encoding (MAK/ANG)

### 2.3 Model Sonuçları

#### Random Forest
```
Parametreler: n_estimators=200, max_depth=15
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test R²:     0.78
CV R²:       0.76 ± 0.03
RMSE:        0.013 m/s (1.3 cm/s)
MAE:         0.009 m/s (0.9 cm/s)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### XGBoost
```
Parametreler: n_estimators=200, max_depth=6, learning_rate=0.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test R²:     0.77
CV R²:       0.75 ± 0.04
RMSE:        0.014 m/s (1.4 cm/s)
MAE:         0.010 m/s (1.0 cm/s)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 2.4 Feature Importance (Random Forest)

| Özellik | Önem | Açıklama |
|---------|------|----------|
| density | 0.18 | Parçacık yoğunluğu |
| delta_rho | 0.15 | Yoğunluk farkı (batma kuvveti) |
| d_eq | 0.12 | Eşdeğer çap |
| volume | 0.11 | Hacim |
| L | 0.09 | En büyük boyut |
| sphericity | 0.08 | Küresellik |
| CSF | 0.07 | Corey Shape Factor |
| cat_enc | 0.06 | Kategori (şekil) |
| aspect_ratio | 0.05 | En/boy oranı |
| surface_area | 0.04 | Yüzey alanı |

**Yorumlar:**
- Yoğunluk ve yoğunluk farkı en önemli özellikler (%33 toplam)
- Boyut özellikleri (d_eq, volume, L) ikinci sırada (%32 toplam)
- Şekil faktörleri (sphericity, CSF, aspect_ratio) anlamlı katkı yapıyor (%20 toplam)
- Fiziksel beklentilerle uyumlu sonuçlar

### 2.5 Model Değerlendirmesi

**R² = 0.76 Ne Anlama Geliyor?**
- Varyansın %76'sı model tarafından açıklanıyor
- Geriye kalan %24 açıklanamayan varyans
- Mikroplastik çökelme için iyi bir değer

**Domain Karşılaştırması:**
| Alan | Tipik R² |
|------|----------|
| Fizik/mühendislik | 0.90+ |
| Biyoloji | 0.60-0.80 |
| Sosyal bilimler | 0.30-0.50 |
| **Bu çalışma** | **0.76** |

**Limitasyonlar:**
- ABS CUBE ve ABS EC kategorilerinde 3. boyut eksik
- %13 veri eşleştirilemedi (163 deney)
- Bazı kategorilerde az sayıda deney

---

## 3. Teorik Karşılaştırma

### 3.1 Yöntem

İpek'in teorik hesaplamaları ile ölçülen hızların karşılaştırılması. Teorik hız, Stokes kanunu ve modifiye formülasyonlar kullanılarak hesaplanmıştır.

**Karşılaştırılan Veriler:**
- Eski tracker hızları (V1 - dar arama alanı)
- Yeni tracker hızları (V3 - genişletilmiş arama + interval tarama)
- İpek teorik hızlar

### 3.2 Korelasyon Sonuçları

| Karşılaştırma | Pearson r | Eğim | Sapma |
|---------------|-----------|------|-------|
| Eski vs Teorik | 0.78 | 0.65 | -42% |
| **Yeni vs Teorik** | **0.91** | **0.66** | **-34%** |

### 3.3 Kategori Bazlı Analiz

| Kategori | Deney | Eski r | Yeni r | İyileşme |
|----------|-------|--------|--------|----------|
| BSP | 45 | 0.72 | 0.89 | +24% |
| C | 67 | 0.81 | 0.93 | +15% |
| HC | 52 | 0.75 | 0.90 | +20% |
| WSP | 38 | 0.69 | 0.87 | +26% |
| CUBE | 41 | 0.77 | 0.91 | +18% |
| EC | 29 | 0.73 | 0.88 | +21% |

### 3.4 Sistematik Sapma Analizi

**Gözlem:** Ölçülen hızlar, teorik hızlardan sistematik olarak düşük.

**Olası Nedenler:**
1. **Duvar etkisi**: Silindir çapının sonlu olması
2. **Sıvı viskozitesi**: Gerçek viskozite varsayılandan farklı olabilir
3. **Parçacık şekli**: Teorik hesap küresel parçacık varsayıyor
4. **Başlangıç ivmesi**: Terminal hıza ulaşma süresi

**Düzeltme Faktörü:**
```
v_teorik_düzeltilmiş = v_teorik × 0.66
```

### 3.5 Tracker Versiyon Karşılaştırması

| Versiyon | Teknik Değişiklik | Kurtarılan Deney |
|----------|-------------------|------------------|
| V1 → V2 | Arama alanı: %25-%75 → %10-%90 | 28/115 not_found |
| V2 → V3 | Interval frame tarama eklendi | 163/323 insufficient_movement |

**V3 İyileştirmeleri:**
- Toplam başarılı deney: 1078 → 1241+ (potansiyel)
- Korelasyon artışı: 0.78 → 0.91

---

## 4. Sonuçlar ve Öneriler

### 4.1 Temel Çıkarımlar

1. **ML Modeli Başarılı**: R² = 0.76 ile parçacık özelliklerinden hız tahmini yapılabiliyor
2. **Fiziksel Tutarlılık**: Feature importance fiziksel beklentilerle uyumlu
3. **Tracker İyileştirmesi**: V3 tracker ile korelasyon %78'den %91'e çıktı
4. **Sistematik Sapma**: Teorik hesaplardan %34 düşük ölçüm (düzeltilebilir)

### 4.2 Güçlü Yönler

- ✅ Çoklu malzeme desteği (10 farklı plastik)
- ✅ Ölçülmüş yoğunluk değerleri kullanıldı
- ✅ Fiziksel şekil faktörleri dahil edildi
- ✅ Cross-validation ile güvenilir metrikler
- ✅ Teorik hesaplarla yüksek korelasyon

### 4.3 Geliştirilecek Alanlar

- ⚠️ ABS CUBE ve ABS EC için 3. boyut eksik
- ⚠️ Bazı kategorilerde az veri
- ⚠️ Sistematik sapma düzeltmesi gerekli
- ⚠️ Daha fazla parçacık çeşitliliği yararlı olabilir

### 4.4 Sonraki Adımlar

1. **Eksik boyut verilerinin tamamlanması** (+%13 veri)
2. **Sistematik sapma düzeltme faktörünün validasyonu**
3. **Neural network modeli denemesi** (potansiyel iyileştirme)
4. **Feature selection optimizasyonu**
5. **Daha fazla parçacık ile veri artırma**

---

## 5. Dosya Referansları

| Dosya | Açıklama |
|-------|----------|
| `ml_full_model_v2.py` | ML model eğitim scripti |
| `ml_data_full_v2.csv` | Eğitim verisi (1078 satır) |
| `Hiz_Karsilastirma_Raporu.xlsx` | Teorik karşılaştırma Excel raporu |
| `ALL PARTICLES MEASUREMENTS.xlsx` | Parçacık ölçümleri |
| `auto_particle_tracker_v3.py` | Son tracker versiyonu |

---

*Rapor otomatik olarak oluşturulmuştur.*
*Claude Code - TÜBİTAK 1001 Mikroplastik Projesi*
