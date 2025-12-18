# ML Model Hazırlık - Veri Soruları

## Proje Özeti
**Amaç:** Parçacık özelliklerinden (boyut, şekil, malzeme) çökelme hızı tahmini
**Veri:** 1503 deney, 202 benzersiz parçacık
**Model:** XGBoost / Random Forest (regresyon)

---

## 1. Malzeme Yoğunlukları

Modelin fiziksel anlam taşıması için doğru yoğunluk değerleri gerekli.

| Malzeme | Tahmin (g/cm³) | Gerçek Değer | Onay |
|---------|----------------|--------------|------|
| Plexiglass (PMMA) | 1.18 | | [ ] |
| ABS | 1.04 | | [ ] |
| PLA | 1.24 | | [ ] |
| PS | 1.05 | | [ ] |
| RESIN (A=9 R=4.5) | 1.10 | | [ ] |
| Su (ortam) | 1.00 | | [ ] |

**Not:** Parçacık yoğunluğu > Su yoğunluğu olmalı (batması için)

---

## 2. Eksik Boyut 3 Verisi

634 satırda (toplam 1503) Boyut 3 değeri yok.

**Etkilenen kategoriler:**
- PLA CUBE: 167
- RESIN (a=9 r=4.5): 116
- RESIN (a=6 r=3): 115
- ABS CUBE: 56
- PLA C: 55
- ABS EC: 48
- PS: 31
- C: 29
- ABS C: 11
- PA 6: 6

**Soru:** Bu parçacıklar için Boyut 3 neden yok?

- [ ] 2D parçacık (film/plaka) - çok ince, ölçülmedi
- [ ] Küresel parçacık - a=b=c varsayılabilir
- [ ] Silindirik parçacık - c = çap olarak alınabilir
- [ ] Ölçüm yapılmadı - tahmin gerekli
- [ ] Diğer: _______________

**Önerilen çözüm:**
```
Eğer küp/küre ise: Boyut 3 = Boyut 1
Eğer silindir ise: Boyut 3 = Boyut 2 (çap)
Eğer plaka ise: Boyut 3 = 0.1 mm (varsayılan kalınlık)
```

---

## 3. Outlier Hız Değerleri

3 satırda anormal yüksek hız var (diğerleri max 0.25 m/s):

| # | Kategori | Kod | Hız (m/s) | Durum |
|---|----------|-----|-----------|-------|
| 1 | HC | HC-6 | 14.25 | [ ] Hata / [ ] Doğru |
| 2 | PLA HC | HC-9 | 1.58 | [ ] Hata / [ ] Doğru |
| 3 | RESIN (a=9 r=4.5) | CUBE-5 | 14.25 | [ ] Hata / [ ] Doğru |

**Soru:** Bu değerler:
- [ ] Ölçüm/hesaplama hatası - silinecek
- [ ] Birim hatası (cm/s yerine m/s yazılmış)
- [ ] Gerçek değer - kalacak

**Not:** 14.25 m/s = 51 km/h - mikroplastik için imkansız

---

## 4. Kategori/Şekil Açıklamaları

Model için şekil kategorilerinin ne anlama geldiğini bilmem gerekiyor.

| Kod | Açıklama | Şekil Tipi | Onay |
|-----|----------|------------|------|
| BSP | ? | ? | [ ] |
| WSP | ? | ? | [ ] |
| HC | ? | ? | [ ] |
| C | ? | ? | [ ] |
| EC | ? | ? | [ ] |
| CUBE | Küp | Düzenli | [ ] |
| PA 6 | Polyamide 6? | ? | [ ] |

**Şekil tipleri için öneriler:**
- Küre (Sphere)
- Küp (Cube)
- Silindir (Cylinder)
- Disk/Plaka (Disk/Plate)
- Fiber/Çubuk (Fiber/Rod)
- Düzensiz (Irregular)

---

## 5. Ek Sorular

### 5.1 Deney Ortamı
- [ ] Sıvı: Saf su mu?
- [ ] Sıcaklık: Oda sıcaklığı (~20°C)?
- [ ] Kolon yüksekliği: 28.5 cm (kod'dan)

### 5.2 Ölçüm Yöntemi
- [ ] Boyutlar nasıl ölçüldü? (Kumpas, mikroskop, 3D tarama?)
- [ ] Hassasiyet nedir? (±0.01 mm?)

### 5.3 Tekrar Deneyleri
- Her parçacık için 3 tekrar var (FIRST, SECOND, THIRD)
- [ ] Modelde ortalama mı kullanılsın?
- [ ] Her tekrar ayrı veri noktası mı olsun?

---

## Yanıtlar

_Bu bölümü doldurun:_

### 1. Yoğunluklar
```
Plexiglass:
ABS:
PLA:
PS:
RESIN:
```

### 2. Boyut 3 Eksikliği
```
Açıklama:
Çözüm önerisi:
```

### 3. Outlier'lar
```
HC-6 (14.25):
HC-9 (1.58):
CUBE-5 (14.25):
```

### 4. Kategori Açıklamaları
```
BSP:
WSP:
HC:
C:
EC:
PA 6:
```

### 5. Ek Bilgiler
```
Sıvı:
Sıcaklık:
Ölçüm yöntemi:
Tekrarlar:
```

---

## Sonraki Adımlar

Bu sorular yanıtlandıktan sonra:

1. [ ] Veri temizleme (outlier, eksik veri)
2. [ ] Feature engineering (shape factors)
3. [ ] Model eğitimi (XGBoost baseline)
4. [ ] Değerlendirme (RMSE, R², cross-validation)
5. [ ] Feature importance analizi

---

*Oluşturulma: 2025-12-15*
*Proje: TÜBİTAK 1001 - Mikroplastik Çökelme Hızı*
