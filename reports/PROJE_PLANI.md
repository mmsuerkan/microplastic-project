# TÜBİTAK 1001 Projesi - Mikroplastik Çökelme Hızı Araştırması

## Proje Özeti
**Başlık:** Düzenli ve Düzensiz Şekilli Mikroplastik Parçacıkların Çökelme Hızlarının Deneysel, Sayısal ve Derin Öğrenmeye Dayanan Yöntemlerle Araştırılması

**Yürütücü:** Doç.Dr. Aslı Numanoğlu Genç (TED Üniversitesi)

## İş Paketleri ve Durum

### İP-1: Deney Numunelerinin Hazırlanması (20%) - TAMAMLANDI
- Polimer tabakalardan MP üretimi
- Malzemeler: ABS, RESIN, BSP, HC, vs.
- Şekiller: C (silindir), CUBE (küp), SP (küre)

### İP-2: Çökelme Deneyleri (25%) - TAMAMLANDI
- 1500+ deney videosu kaydedildi
- Parçacık takibi ile hız, salınım, yörünge analizi
- MAK (makro) ve ANG (açılı) görünümler
- 3 tekrar: FIRST, SECOND, THIRD
- CD-Re grafikleri oluşturulacak

### İP-3: CFD Modelleme (25%) - YAPILACAK
- StarCCM+ veya ANSYS Fluent ile sayısal simülasyon
- Sorumlu: Onur Baş

### İP-4: Derin Öğrenme (30%) - YAPILACAK
- RAFT modeli + LSTM hibrid yapı
- Evrensel şekil faktörü geliştirme
- Parçacık yörüngesi ve eksen hareketi analizi
- Başarı ölçütü: RMSE/MAE %0.2, MVAF/TVE %5

## Mevcut Durum (Aralık 2024)

### Tamamlanan İşler
1. Tüm deneyler işlendi (1318 lokal + 226 S3)
2. Parçacık takip algoritması çalışıyor (auto_particle_tracker.py)
3. FFT analizi ile salınım frekansı hesaplanıyor
4. CLAHE ön işleme eklendi
5. S3'e yükleme devam ediyor (1301 video)

### Sonraki Adımlar
1. S3 yüklemesi tamamlansın (yarın öğlene kadar)
2. experiments.json güncelle - tüm S3 verilerini listele
3. Web app'i geliştir - filtreler ekle:
   - Tarih, Görünüm (MAK/ANG), Tekrar, Kategori
   - Durum (success/fail), Fail nedeni
   - Frekans aralığı, Salınım genliği
   - Dikey hareket, Yatay sapma
4. Fail olanları analiz et - algoritmaları iyileştir
5. Veri temizliği yap
6. LSTM modeli için veri hazırla
7. PyTorch ile model eğit

## Teknik Detaylar

### Klasör Yapısı
```
processed_results/
├── success/
│   └── TARIH/VIEW/REPEAT/CATEGORY/CODE/
│       ├── auto_tracking_results.csv
│       ├── auto_tracked_path.jpg
│       └── output_video.mp4
└── fail/
    └── REASON/TARIH/VIEW/REPEAT/CATEGORY/CODE/
        └── ...
```

### Fail Nedenleri
- video_error: Video açılamadı
- not_found: Parçacık bulunamadı
- lost_early: Erken kayboldu (<50 frame)
- insufficient_movement: Yetersiz dikey hareket (<400px)
- horizontal_drift: Çok fazla yatay sapma (>50%)

### S3 Bucket
- Bucket: microplastic-experiments
- CloudFront ile web erişimi

### CSV Çıktıları
auto_tracking_results.csv içeriği:
- frame, x, y koordinatları
- vertical_pixels, horizontal_drift
- dominant_frequency_hz, oscillation_amplitude
- oscillation_detected

## Notlar
- Videolar SSD'de: C:/Users/mmert/PycharmProjects/ObjectTrackingProject/source_videos
- D: diskindeki orijinaller yedek olarak saklanıyor
- Her deney 3 kez tekrarlandı (FIRST, SECOND, THIRD)
