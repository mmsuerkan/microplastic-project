import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import time
import os

# Çıktı klasörü
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)

# Toplam mesafe sabit (örnek: 28.5 cm = 0.285 m)
total_distance_meters = 0.285

# Video yükle
cap = cv2.VideoCapture("output.mp4")
ret, frame = cap.read()
if not ret:
    print("Video açılamadı")
    exit()

# CLAHE (Contrast Limited Adaptive Histogram Equalization) oluştur
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Ön işleme fonksiyonu
def preprocess_frame(frame, use_clahe=True, use_edge=False):
    """
    Frame'i ön işleme tabi tut
    
    Args:
        frame: Giriş frame
        use_clahe: CLAHE kullan
        use_edge: Edge detection kullan
    
    Returns:
        processed_frame: İşlenmiş frame
    """
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if use_clahe:
        # CLAHE uygula
        gray = clahe.apply(gray)
    
    if use_edge:
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        # Edge'leri orijinal frame'e ekle
        frame_edge = frame.copy()
        frame_edge[edges > 0] = [0, 255, 0]  # Yeşil kenarlar
        return frame_edge
    
    # CLAHE uygulanmış frame'i BGR'ye geri çevir
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Ön işleme parametreleri (True/False yaparak açıp kapatabilirsiniz)
USE_CLAHE = True
USE_EDGE = False

# ROI seçimi (sadece orijinal frame'i göster)
cv2.namedWindow("ROI Selection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ROI Selection", 1280, 720)
roi = cv2.selectROI("ROI Selection", frame, False)
cv2.destroyWindow("ROI Selection")

# -----------------------------
# 1. CSRT TRACKER BAŞLANGICI
# -----------------------------
# İlk frame'i de ön işleme tabi tut
init_frame = preprocess_frame(frame, USE_CLAHE, USE_EDGE) if (USE_CLAHE or USE_EDGE) else frame

csrt_tracker = cv2.TrackerCSRT_create()
csrt_tracker.init(init_frame, roi)

csrt_frame_count = 0
csrt_prev_bbox = roi
csrt_tracking_start_time = None
csrt_tracking_end_time = None
csrt_no_movement_start_time = None
csrt_path_points = []
csrt_last_valid_frame = None

csrt_csv_path = os.path.join(output_dir, 'csrt_tracking_coordinates.csv')
csrt_csv = open(csrt_csv_path, 'w', newline='')
csrt_writer = csv.writer(csrt_csv)
csrt_writer.writerow(['Frame', 'X', 'Y', 'Time'])

# -----------------------------
# 2. OPTICAL FLOW BAŞLANGICI
# -----------------------------
# Optical flow için gray scale (CLAHE uygulanmış olabilir)
if USE_CLAHE:
    old_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
else:
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

roi_x, roi_y, roi_w, roi_h = roi
mask = np.zeros_like(old_gray)
mask[int(roi_y):int(roi_y + roi_h), int(roi_x):int(roi_x + roi_w)] = 255
p0 = cv2.goodFeaturesToTrack(old_gray, 25, 0.3, 7, mask=mask)
vector_data = []

# -----------------------------
# DÖNGÜ
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Frame'i ön işleme tabi tut (tracker için)
    processed_frame = preprocess_frame(frame, USE_CLAHE, USE_EDGE) if (USE_CLAHE or USE_EDGE) else frame

    # CSRT Takibi (işlenmiş frame üzerinde)
    success, bbox = csrt_tracker.update(processed_frame)
    if success:
        if csrt_tracking_start_time is None:
            csrt_tracking_start_time = time.time()
        csrt_tracking_end_time = time.time()

        csrt_last_valid_frame = frame.copy()
        center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
        csrt_path_points.append(center)

        elapsed_time = csrt_tracking_end_time - csrt_tracking_start_time
        csrt_writer.writerow([csrt_frame_count, center[0], center[1], elapsed_time])

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        if int(bbox[0]) == int(csrt_prev_bbox[0]) and int(bbox[1]) == int(csrt_prev_bbox[1]):
            if csrt_no_movement_start_time is None:
                csrt_no_movement_start_time = time.time()
            elif time.time() - csrt_no_movement_start_time > 2:
                print("CSRT: Nesne 2 saniye boyunca aynı konumda kaldı.")
                break
        else:
            csrt_no_movement_start_time = None

        csrt_prev_bbox = bbox
    else:
        print("CSRT: Takip başarısız oldu.")
        break

    # Optical Flow
    frame_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    if p0 is not None:
        p1_optical, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
        if p1_optical is not None:
            good_new = p1_optical[st == 1]
            good_old = p0[st == 1]

            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                vector_data.append([c, d, a - c, b - d])
                cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 1, tipLength=0.3)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    for pt in csrt_path_points:
        cv2.circle(frame, pt, 2, (0, 0, 255), -1)

    # Sadece orijinal frame'i göster (tracker arka planda işlenmiş olanı kullanıyor)
    cv2.imshow("Tracking", frame)
    csrt_frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# KAPAT
# -----------------------------
csrt_csv.close()
cap.release()
cv2.destroyAllWindows()

# CSRT Görselleştirme
if csrt_last_valid_frame is not None:
    for pt in csrt_path_points:
        cv2.circle(csrt_last_valid_frame, pt, 2, (0, 0, 255), -1)
    csrt_img_path = os.path.join(output_dir, "csrt_tracked_path.jpg")
    cv2.imwrite(csrt_img_path, csrt_last_valid_frame)

# CSRT Ortalama Hız Hesabı
if csrt_tracking_start_time and csrt_tracking_end_time:
    total_time = csrt_tracking_end_time - csrt_tracking_start_time
    avg_speed = total_distance_meters / total_time
    print(f"CSRT Takip Süresi: {total_time:.2f} s")
    print(f"Ortalama Hız: {avg_speed:.2f} m/s")
else:
    print("CSRT: Süre hesaplanamadı.")

# Optical Flow Görselleştirme
if vector_data:
    vector_data = np.array(vector_data)
    magnitudes = np.sqrt(vector_data[:, 2]**2 + vector_data[:, 3]**2)
    mean_magnitude = np.mean(magnitudes)
    print(f"Optical Flow Ortalama Hareket Yoğunluğu: {mean_magnitude:.4f}")
    
    # FFT ile Dominant Genlik Analizi
    if len(magnitudes) > 30:  # En az 1 saniyelik veri (30 fps için)
        # FFT hesaplama
        fft_result = np.fft.fft(magnitudes - np.mean(magnitudes))
        fft_magnitude = np.abs(fft_result)
        frequencies = np.fft.fftfreq(len(magnitudes), 1/30)  # 30 fps varsayımı
        
        # Pozitif frekanslar
        positive_idx = frequencies > 0
        pos_freq = frequencies[positive_idx]
        pos_magnitude = fft_magnitude[positive_idx]
        
        # İlk iki dominant frekans ve genlik
        # Genlik değerlerine göre sırala
        sorted_indices = np.argsort(pos_magnitude)[::-1]  # Büyükten küçüğe
        
        # 1. Dominant
        dominant1_idx = sorted_indices[0]
        dominant1_freq = pos_freq[dominant1_idx]
        dominant1_amplitude = pos_magnitude[dominant1_idx]
        
        # 2. Dominant (varsa)
        dominant2_freq = None
        dominant2_amplitude = None
        if len(sorted_indices) > 1:
            dominant2_idx = sorted_indices[1]
            dominant2_freq = pos_freq[dominant2_idx]
            dominant2_amplitude = pos_magnitude[dominant2_idx]
        
        print(f"\n=== SALIM HAREKETİ ANALİZİ ===")
        print(f"1. Dominant Frekans: {dominant1_freq:.2f} Hz | Genlik: {dominant1_amplitude:.2f}")
        if dominant1_freq > 0:
            period1 = 1 / dominant1_freq
            print(f"   Periyot: {period1:.2f} saniye")
        
        if dominant2_freq is not None:
            print(f"2. Dominant Frekans: {dominant2_freq:.2f} Hz | Genlik: {dominant2_amplitude:.2f}")
            if dominant2_freq > 0:
                period2 = 1 / dominant2_freq
                print(f"   Periyot: {period2:.2f} saniye")
            
            # İki frekansın oranı (harmonik analizi)
            if dominant1_freq > 0 and dominant2_freq > 0:
                freq_ratio = dominant2_freq / dominant1_freq
                if abs(freq_ratio - round(freq_ratio)) < 0.1:
                    print(f"   Harmonik İlişki: {round(freq_ratio)}. harmonik")
        
        # Salınım tipi belirleme
        total_energy = np.sum(pos_magnitude)
        dom1_ratio = dominant1_amplitude / total_energy
        dom2_ratio = dominant2_amplitude / total_energy if dominant2_amplitude else 0
        
        if dom1_ratio > 0.7:
            motion_type = "Basit Harmonik Salınım"
        elif dom1_ratio + dom2_ratio > 0.6:
            motion_type = "Karmaşık Salınım (Çift Frekanslı)"
        elif dom1_ratio > 0.3:
            motion_type = "Gürültülü Salınım"
        else:
            motion_type = "Düzensiz Hareket"
        
        print(f"\nHareket Tipi: {motion_type}")
        print(f"Dominant Enerji Oranları: 1. %{dom1_ratio*100:.1f}, 2. %{dom2_ratio*100:.1f}")
        
        # FFT spektrum görselleştirme
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(magnitudes))/30, magnitudes)
        plt.xlabel('Zaman (s)')
        plt.ylabel('Hareket Genliği')
        plt.title('Hareket Genliği Zaman Serisi')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.stem(pos_freq[:50], pos_magnitude[:50], basefmt=' ')
        plt.axvline(dominant1_freq, color='r', linestyle='--', 
                   label=f'1. Dominant: {dominant1_freq:.2f} Hz', linewidth=2)
        if dominant2_freq is not None:
            plt.axvline(dominant2_freq, color='orange', linestyle='--', 
                       label=f'2. Dominant: {dominant2_freq:.2f} Hz', linewidth=1.5)
        plt.xlabel('Frekans (Hz)')
        plt.ylabel('Genlik')
        plt.title('FFT Frekans Spektrumu - Salınım Analizi')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fft_img_path = os.path.join(output_dir, 'fft_analysis.png')
        plt.savefig(fft_img_path)
        plt.close()
        print(f"FFT analizi kaydedildi: {fft_img_path}")

    # CSV kaydet
    vector_csv_path = os.path.join(output_dir, 'optical_vector_data.csv')
    with open(vector_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Start_X', 'Start_Y', 'Vector_X', 'Vector_Y'])
        writer.writerows(vector_data)

    # Quiver görseli
    plt.figure(figsize=(10, 10))
    plt.quiver(vector_data[:, 0], vector_data[:, 1],
               vector_data[:, 2], vector_data[:, 3],
               angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.003)
    plt.gca().invert_yaxis()
    plt.title('Optical Flow Motion Vectors')
    vector_img_path = os.path.join(output_dir, 'optical_flow_vectors.png')
    plt.savefig(vector_img_path)
    plt.close()
else:
    print("Optical Flow: Vektör verisi yok.")
