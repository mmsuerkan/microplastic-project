import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import time
import os

# =============================================
# ŞEFFAF PARÇACIK TAKİP SİSTEMİ v2.0
# Background Subtraction + Kalman Filter
# =============================================

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

# Video özellikleri
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 50  # Varsayılan FPS (images_combine.py'den)
print(f"Video FPS: {fps}")

# =============================================
# PARAMETRELER (Ayarlanabilir)
# =============================================
# Background Subtraction parametreleri
BG_HISTORY = 500           # Arka plan modelinin geçmiş frame sayısı
BG_VAR_THRESHOLD = 16      # Varyans eşiği (düşük = daha hassas)
BG_DETECT_SHADOWS = False  # Gölge tespiti (şeffaf için kapalı)

# Morphological operations
MORPH_KERNEL_SIZE = 3      # Kernel boyutu
MORPH_ITERATIONS = 2       # İterasyon sayısı

# Blob/Contour filtreleme
MIN_CONTOUR_AREA = 10      # Minimum kontur alanı (piksel^2)
MAX_CONTOUR_AREA = 5000    # Maksimum kontur alanı

# Takip parametreleri
SEARCH_RADIUS = 150        # Kayıp parçacık arama yarıçapı (artırıldı)
NO_MOVEMENT_TIMEOUT = 3    # Hareketsizlik timeout (saniye)
LOST_FRAME_THRESHOLD = 30  # Kaç frame kayıp olursa durdur (artırıldı)

# Kalman Filter parametreleri
KALMAN_PROCESS_NOISE = 1e-2    # Hareket belirsizliği
KALMAN_MEASUREMENT_NOISE = 1e-1  # Ölçüm belirsizliği

# =============================================
# BACKGROUND SUBTRACTOR OLUŞTUR
# =============================================
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=BG_HISTORY,
    varThreshold=BG_VAR_THRESHOLD,
    detectShadows=BG_DETECT_SHADOWS
)

# Alternatif: KNN (daha iyi sonuç verebilir)
# bg_subtractor = cv2.createBackgroundSubtractorKNN(
#     history=BG_HISTORY,
#     dist2Threshold=400.0,
#     detectShadows=BG_DETECT_SHADOWS
# )

# CLAHE (kontrast iyileştirme)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Morphological kernel
morph_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
)

# =============================================
# ROI SEÇİMİ
# =============================================
cv2.namedWindow("ROI Selection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ROI Selection", 1280, 720)
roi = cv2.selectROI("ROI Selection", frame, False)
cv2.destroyWindow("ROI Selection")

roi_x, roi_y, roi_w, roi_h = [int(v) for v in roi]
print(f"Seçilen ROI: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

# İlk merkez nokta
current_center = (roi_x + roi_w // 2, roi_y + roi_h // 2)
prev_center = current_center

# =============================================
# KALMAN FİLTER OLUŞTUR
# =============================================
# 4 durum değişkeni: x, y, vx, vy (pozisyon ve hız)
# 2 ölçüm: x, y
kalman = cv2.KalmanFilter(4, 2)

# Geçiş matrisi (pozisyon + hız modeli)
kalman.transitionMatrix = np.array([
    [1, 0, 1, 0],  # x = x + vx
    [0, 1, 0, 1],  # y = y + vy
    [0, 0, 1, 0],  # vx = vx
    [0, 0, 0, 1]   # vy = vy
], dtype=np.float32)

# Ölçüm matrisi (sadece x, y ölçüyoruz)
kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=np.float32)

# Süreç gürültüsü (hareket belirsizliği)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE

# Ölçüm gürültüsü
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE

# Başlangıç durumu
kalman.statePre = np.array([
    [current_center[0]],
    [current_center[1]],
    [0],  # başlangıç vx = 0
    [5]   # başlangıç vy = 5 (aşağı düşüyor)
], dtype=np.float32)

kalman.statePost = kalman.statePre.copy()

# =============================================
# TAKİP DEĞİŞKENLERİ
# =============================================
frame_count = 0
tracking_start_time = None
tracking_end_time = None
no_movement_start_time = None
path_points = []
last_valid_frame = None
lost_frames = 0

# CSV dosyası
csv_path = os.path.join(output_dir, 'transparent_tracking_coordinates.csv')
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'X', 'Y', 'Time', 'Area', 'Method'])

# Optical flow için veri
vector_data = []
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
old_gray = clahe.apply(old_gray)

# =============================================
# YARDIMCI FONKSİYONLAR
# =============================================
def find_particle_in_region(fg_mask, search_center, search_radius):
    """
    Belirli bir bölgede parçacık ara
    """
    h, w = fg_mask.shape
    x, y = search_center

    # Arama bölgesi sınırları
    x1 = max(0, x - search_radius)
    y1 = max(0, y - search_radius)
    x2 = min(w, x + search_radius)
    y2 = min(h, y + search_radius)

    # Bölgeyi kes
    region = fg_mask[y1:y2, x1:x2]

    # Konturları bul
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, 0

    # En uygun konturu bul (merkeze en yakın ve boyut filtreli)
    best_contour = None
    best_distance = float('inf')
    best_area = 0

    region_center = (search_radius, search_radius)

    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Merkeze uzaklık
                dist = np.sqrt((cx - region_center[0])**2 + (cy - region_center[1])**2)

                if dist < best_distance:
                    best_distance = dist
                    best_contour = contour
                    best_area = area

    if best_contour is None:
        return None, None, 0

    # Global koordinatlara çevir
    M = cv2.moments(best_contour)
    cx = int(M["m10"] / M["m00"]) + x1
    cy = int(M["m01"] / M["m00"]) + y1

    # Bounding box
    bx, by, bw, bh = cv2.boundingRect(best_contour)
    bbox = (bx + x1, by + y1, bw, bh)

    return (cx, cy), bbox, best_area


def preprocess_frame(frame):
    """
    Frame ön işleme
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    return gray


# =============================================
# İLK FRAME'LERİ ARKA PLAN MODELİNE EKLE
# =============================================
print("Arka plan modeli oluşturuluyor...")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# İlk birkaç frame'i arka plan modeline ekle
for i in range(min(30, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
    ret, init_frame = cap.read()
    if ret:
        gray = preprocess_frame(init_frame)
        bg_subtractor.apply(gray, learningRate=0.1)

# Video başına dön
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame = cap.read()

print("Takip başlıyor...")

# =============================================
# ANA DÖNGÜ
# =============================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ön işleme
    gray = preprocess_frame(frame)

    # Kalman tahmin (ölçümden önce)
    prediction = kalman.predict()
    predicted_center = (int(prediction[0, 0]), int(prediction[1, 0]))

    # Background subtraction
    fg_mask = bg_subtractor.apply(gray, learningRate=0.001)

    # Morphological operations (gürültü temizleme)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, morph_kernel, iterations=MORPH_ITERATIONS)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=MORPH_ITERATIONS)

    # Parçacık ara (Kalman tahmininin çevresinde)
    center, bbox, area = find_particle_in_region(fg_mask, predicted_center, SEARCH_RADIUS)

    detection_method = "background_subtraction"

    # Eğer bulunamadıysa, frame farkı dene
    if center is None:
        frame_diff = cv2.absdiff(old_gray, gray)
        _, diff_thresh = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
        diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, morph_kernel)

        center, bbox, area = find_particle_in_region(diff_thresh, predicted_center, SEARCH_RADIUS)
        detection_method = "frame_difference"

    # Eğer hala bulunamadıysa, adaptive threshold dene
    if center is None:
        # Lokal bölgede adaptive threshold
        h, w = gray.shape
        px, py = predicted_center
        x1, y1 = max(0, px - SEARCH_RADIUS), max(0, py - SEARCH_RADIUS)
        x2, y2 = min(w, px + SEARCH_RADIUS), min(h, py + SEARCH_RADIUS)

        local_region = gray[y1:y2, x1:x2]
        if local_region.size > 0:
            local_thresh = cv2.adaptiveThreshold(
                local_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            local_thresh = cv2.morphologyEx(local_thresh, cv2.MORPH_OPEN, morph_kernel)

            # Konturu bul
            contours, _ = cv2.findContours(local_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Merkeze en yakın ve uygun boyuttaki konturu seç
                region_center = (SEARCH_RADIUS, SEARCH_RADIUS)
                best_contour = None
                best_dist = float('inf')

                for cnt in contours:
                    cnt_area = cv2.contourArea(cnt)
                    if MIN_CONTOUR_AREA <= cnt_area <= MAX_CONTOUR_AREA:
                        M = cv2.moments(cnt)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            dist = np.sqrt((cx - region_center[0])**2 + (cy - region_center[1])**2)
                            if dist < best_dist:
                                best_dist = dist
                                best_contour = cnt

                if best_contour is not None:
                    M = cv2.moments(best_contour)
                    cx = int(M["m10"] / M["m00"]) + x1
                    cy = int(M["m01"] / M["m00"]) + y1
                    center = (cx, cy)
                    bx, by, bw, bh = cv2.boundingRect(best_contour)
                    bbox = (bx + x1, by + y1, bw, bh)
                    area = cv2.contourArea(best_contour)
                    detection_method = "adaptive_threshold"

    # Parçacık bulundu mu?
    if center is not None:
        lost_frames = 0

        # Kalman güncelle
        measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]])
        kalman.correct(measurement)

        if tracking_start_time is None:
            tracking_start_time = time.time()
        tracking_end_time = time.time()

        current_center = center
        last_valid_frame = frame.copy()
        path_points.append(center)

        elapsed_time = tracking_end_time - tracking_start_time
        csv_writer.writerow([frame_count, center[0], center[1], elapsed_time, area, detection_method])

        # Görselleştirme
        if bbox:
            p1 = (bbox[0], bbox[1])
            p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

        cv2.circle(frame, center, 5, (0, 255, 255), -1)  # Sarı: gerçek tespit
        cv2.circle(frame, predicted_center, 5, (255, 0, 255), 2)  # Mor: Kalman tahmin

        # Hareketsizlik kontrolü
        movement = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
        if movement < 2:
            if no_movement_start_time is None:
                no_movement_start_time = time.time()
            elif time.time() - no_movement_start_time > NO_MOVEMENT_TIMEOUT:
                print(f"Parçacık {NO_MOVEMENT_TIMEOUT} saniye boyunca hareketsiz kaldı.")
                break
        else:
            no_movement_start_time = None

        prev_center = center

        # Hareket vektörleri
        if len(path_points) > 1:
            dx = center[0] - path_points[-2][0]
            dy = center[1] - path_points[-2][1]
            if abs(dx) > 0 or abs(dy) > 0:
                vector_data.append([center[0], center[1], dx, dy])
    else:
        lost_frames += 1

        # Kalman tahmini kullan (parçacık bulunamadığında)
        current_center = predicted_center
        path_points.append(predicted_center)

        if tracking_start_time:
            elapsed_time = time.time() - tracking_start_time
            csv_writer.writerow([frame_count, predicted_center[0], predicted_center[1], elapsed_time, 0, "kalman_prediction"])

        # Mor nokta (tahmin)
        cv2.circle(frame, predicted_center, 7, (255, 0, 255), 2)

        print(f"Frame {frame_count}: Parçacık bulunamadı, Kalman tahmini kullanılıyor (kayıp: {lost_frames})")

        if lost_frames > LOST_FRAME_THRESHOLD:
            print(f"Parçacık {LOST_FRAME_THRESHOLD} frame boyunca kayıp. Takip durduruluyor.")
            break

    # Yolu çiz
    for i, pt in enumerate(path_points):
        cv2.circle(frame, pt, 2, (0, 0, 255), -1)
        if i > 0:
            cv2.line(frame, path_points[i-1], pt, (0, 0, 255), 1)

    # Bilgi göster
    info_text = f"Frame: {frame_count} | Method: {detection_method} | Lost: {lost_frames}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Görüntüle
    cv2.imshow("Tracking", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    old_gray = gray.copy()
    frame_count += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Space ile duraklat
        cv2.waitKey(0)

# =============================================
# KAPAT VE SONUÇLARI KAYDET
# =============================================
csv_file.close()
cap.release()
cv2.destroyAllWindows()

# Yol görselleştirme
if last_valid_frame is not None and path_points:
    for i, pt in enumerate(path_points):
        cv2.circle(last_valid_frame, pt, 2, (0, 0, 255), -1)
        if i > 0:
            cv2.line(last_valid_frame, path_points[i-1], pt, (0, 0, 255), 1)

    img_path = os.path.join(output_dir, "transparent_tracked_path.jpg")
    cv2.imwrite(img_path, last_valid_frame)
    print(f"Yol görüntüsü kaydedildi: {img_path}")

# Hız hesabı
if tracking_start_time and tracking_end_time:
    total_time = tracking_end_time - tracking_start_time
    avg_speed = total_distance_meters / total_time
    print(f"\n=== SONUÇLAR ===")
    print(f"Toplam Frame: {frame_count}")
    print(f"Takip Süresi: {total_time:.2f} s")
    print(f"Ortalama Hız: {avg_speed:.4f} m/s ({avg_speed*100:.2f} cm/s)")

    # Piksel cinsinden toplam mesafe
    if len(path_points) > 1:
        pixel_distance = sum(
            np.sqrt((path_points[i][0] - path_points[i-1][0])**2 +
                   (path_points[i][1] - path_points[i-1][1])**2)
            for i in range(1, len(path_points))
        )
        print(f"Piksel Mesafe: {pixel_distance:.2f} px")
        print(f"Piksel/Metre Oranı: {pixel_distance / total_distance_meters:.2f} px/m")
else:
    print("Süre hesaplanamadı.")

# FFT Analizi
if vector_data and len(vector_data) > 30:
    vector_data = np.array(vector_data)
    magnitudes = np.sqrt(vector_data[:, 2]**2 + vector_data[:, 3]**2)

    print(f"\n=== HAREKET ANALİZİ ===")
    print(f"Ortalama Hareket: {np.mean(magnitudes):.4f} px/frame")
    print(f"Maksimum Hareket: {np.max(magnitudes):.4f} px/frame")

    # FFT
    fft_result = np.fft.fft(magnitudes - np.mean(magnitudes))
    fft_magnitude = np.abs(fft_result)
    frequencies = np.fft.fftfreq(len(magnitudes), 1/fps)

    positive_idx = frequencies > 0
    pos_freq = frequencies[positive_idx]
    pos_magnitude = fft_magnitude[positive_idx]

    if len(pos_magnitude) > 0:
        sorted_indices = np.argsort(pos_magnitude)[::-1]
        dominant_freq = pos_freq[sorted_indices[0]]
        dominant_amplitude = pos_magnitude[sorted_indices[0]]

        print(f"\n=== SALIM ANALİZİ ===")
        print(f"Dominant Frekans: {dominant_freq:.2f} Hz")
        print(f"Dominant Genlik: {dominant_amplitude:.2f}")
        if dominant_freq > 0:
            print(f"Periyot: {1/dominant_freq:.2f} s")

        # Görselleştirme
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(np.arange(len(path_points)), [p[1] for p in path_points], 'b-')
        plt.xlabel('Frame')
        plt.ylabel('Y Pozisyonu (piksel)')
        plt.title('Dikey Hareket')
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 2)
        plt.plot(np.arange(len(magnitudes))/fps, magnitudes, 'g-')
        plt.xlabel('Zaman (s)')
        plt.ylabel('Hareket Genliği (px)')
        plt.title('Hareket Genliği Zaman Serisi')
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 3)
        plt.stem(pos_freq[:min(50, len(pos_freq))], pos_magnitude[:min(50, len(pos_magnitude))], basefmt=' ')
        plt.axvline(dominant_freq, color='r', linestyle='--', label=f'Dominant: {dominant_freq:.2f} Hz')
        plt.xlabel('Frekans (Hz)')
        plt.ylabel('Genlik')
        plt.title('FFT Frekans Spektrumu')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        fft_path = os.path.join(output_dir, 'transparent_fft_analysis.png')
        plt.savefig(fft_path)
        plt.close()
        print(f"FFT analizi kaydedildi: {fft_path}")

# Vektör verisi kaydet
if vector_data is not None and len(vector_data) > 0:
    vector_csv_path = os.path.join(output_dir, 'transparent_vector_data.csv')
    with open(vector_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['X', 'Y', 'dX', 'dY'])
        writer.writerows(vector_data)
    print(f"Vektör verisi kaydedildi: {vector_csv_path}")

print("\nTakip tamamlandı!")
