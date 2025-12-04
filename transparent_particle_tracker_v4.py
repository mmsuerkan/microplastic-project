import cv2
import numpy as np
import csv
import time
import os

# =============================================
# ŞEFFAF PARÇACIK TAKİP SİSTEMİ v4.0
# Accumulative Frame Difference + Strict Constraints
# =============================================

output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture("output.mp4")
ret, frame = cap.read()
if not ret:
    print("Video acilamadi")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 50
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps}, Toplam Frame: {total_frames}")

# =============================================
# PARAMETRELER
# =============================================
# Hareket kisitlamalari
MAX_HORIZONTAL_PER_FRAME = 12   # Max yatay hareket (piksel/frame)
MIN_VERTICAL_PER_FRAME = 0      # Min dikey hareket (0 = yavas hareket izinli)
MAX_VERTICAL_PER_FRAME = 15     # Max dikey hareket (daha siki)

# Tespit parametreleri
DIFF_THRESHOLD = 20             # Frame farki esigi
MIN_BLOB_AREA = 5               # Minimum blob alani
MAX_BLOB_AREA = 300             # Maximum blob alani (kucuk parcacik)
SEARCH_RADIUS = 40              # Arama yaricapi (daha dar)

# Diger
NO_MOVEMENT_TIMEOUT = 5.0       # Hareketsizlik suresi (arttirildi)
MAX_LOST_FRAMES = 300           # Tahmin izni (cok arttirildi)

# Silindir sinirlari (parcacik bu Y degerine ulasinca dur)
CYLINDER_BOTTOM_Y = 720         # Silindirin alt siniri (piksel)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# =============================================
# ROI SECIMI
# =============================================
cv2.namedWindow("ROI Selection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ROI Selection", 1280, 720)
print("Parcacigi secin...")
roi = cv2.selectROI("ROI Selection", frame, False)
cv2.destroyWindow("ROI Selection")

roi_x, roi_y, roi_w, roi_h = [int(v) for v in roi]
current_pos = (roi_x + roi_w // 2, roi_y + roi_h // 2)
print(f"Baslangic: {current_pos}")

# Reference frame (ilk frame)
gray_ref = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_ref = clahe.apply(gray_ref)
gray_prev = gray_ref.copy()

# =============================================
# DEGISKENLER
# =============================================
path_points = [current_pos]
frame_count = 0
lost_count = 0
tracking_start = None
tracking_end = None
no_move_start = None

# CSV
csv_file = open(os.path.join(output_dir, 'transparent_tracking_coordinates.csv'), 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'X', 'Y', 'Time', 'Area', 'Method'])

# Vektor verisi (optik akis)
vector_data = []

# =============================================
# YARDIMCI FONKSIYONLAR
# =============================================
def find_movement_blob(diff_img, search_center, radius):
    """
    Frame farkinda hareket blobu bul
    """
    h, w = diff_img.shape
    cx, cy = search_center

    # Arama bolgesi
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius)
    y2 = min(h, cy + radius)

    region = diff_img[y1:y2, x1:x2]

    # Threshold
    _, thresh = cv2.threshold(region, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Konturlari bul
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, 0, thresh

    # En uygun konturu sec - MERKEZE EN YAKIN olani tercih et
    best = None
    best_dist = float('inf')
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                local_cx = int(M["m10"] / M["m00"])
                local_cy = int(M["m01"] / M["m00"])

                # Global koordinatlar
                global_cx = local_cx + x1
                global_cy = local_cy + y1

                # Hareket
                dy = global_cy - cy
                dx = global_cx - cx

                # Sadece asagi veya hafif yukari hareket kabul
                if dy < -5:  # cok yukari gitmesin
                    continue

                # Merkeze uzaklik (oncelik bu)
                dist = np.sqrt(dx**2 + dy**2)

                # En yakin olani sec
                if dist < best_dist:
                    best_dist = dist
                    best = (global_cx, global_cy)
                    best_area = area

    return best, best_area, thresh


def validate_movement(old_pos, new_pos):
    """
    Hareketin fiziksel olarak mantikli oldugunu kontrol et
    """
    dx = new_pos[0] - old_pos[0]
    dy = new_pos[1] - old_pos[1]

    # Yatay hareket kontrolu
    if abs(dx) > MAX_HORIZONTAL_PER_FRAME:
        return False, f"horizontal_large ({dx})"

    # Yukari hareket engelle (sadece asagi veya sabit olmali)
    if dy < -3:  # 3 piksel yukari tolerans
        return False, f"upward ({dy})"

    if dy > MAX_VERTICAL_PER_FRAME:
        return False, f"vertical_large ({dy})"

    return True, "ok"


# =============================================
# ANA DONGU
# =============================================
print("Takip basliyor... (Q=cik, SPACE=duraklat)")

# Video basina don
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame = cap.read()

# Ilk birkac frame'i atla (parcacik henuz dusmemis olabilir)
skip_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti.")
        break

    frame_count += 1

    # Grayscale + CLAHE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)

    # Frame farki (onceki frame ile)
    diff = cv2.absdiff(gray_prev, gray)

    # Hareket blobu bul
    new_pos, area, debug_thresh = find_movement_blob(diff, current_pos, SEARCH_RADIUS)

    method = "none"
    detected = False

    if new_pos is not None:
        valid, reason = validate_movement(current_pos, new_pos)

        if valid:
            current_pos = new_pos
            detected = True
            method = "frame_diff"
            lost_count = 0
        else:
            method = f"rejected:{reason}"
            lost_count += 1
    else:
        # Hareket bulunamadi - lineer tahmin
        if len(path_points) >= 2:
            # Son 20 noktadan ortalama hiz (cok stabil)
            n = min(20, len(path_points) - 1)
            if n > 0:
                avg_dx = sum(path_points[-i][0] - path_points[-i-1][0] for i in range(1, n+1)) / n
                avg_dy = sum(path_points[-i][1] - path_points[-i-1][1] for i in range(1, n+1)) / n
            else:
                avg_dx = 0
                avg_dy = 4  # varsayilan asagi hareket

            # Minimum dikey hareket (parcacik dusmaya devam etmeli)
            if avg_dy < 2:
                avg_dy = 3  # en az 3 piksel asagi

            # Yatay hareketi sifirla (duz asagi gitsin)
            pred_x = int(current_pos[0] + avg_dx * 0.1)  # neredeyse sifir yatay
            pred_y = int(current_pos[1] + avg_dy)

            current_pos = (pred_x, pred_y)
            method = "linear_predict"

        lost_count += 1

    # Zamanlama
    if tracking_start is None:
        tracking_start = time.time()
    tracking_end = time.time()

    # Kaydet
    path_points.append(current_pos)
    elapsed = tracking_end - tracking_start
    csv_writer.writerow([frame_count, current_pos[0], current_pos[1], elapsed, area, method])

    # Vektor verisi (optik akis)
    if len(path_points) >= 2:
        dx = path_points[-1][0] - path_points[-2][0]
        dy = path_points[-1][1] - path_points[-2][1]
        vector_data.append([current_pos[0], current_pos[1], dx, dy])

    # Gorsellestirme
    display = frame.copy()

    # Arama bolgesi
    cv2.rectangle(display,
                  (current_pos[0] - SEARCH_RADIUS, current_pos[1] - SEARCH_RADIUS),
                  (current_pos[0] + SEARCH_RADIUS, current_pos[1] + SEARCH_RADIUS),
                  (100, 100, 100), 1)

    # Yolu ciz
    for i in range(1, len(path_points)):
        pt1 = path_points[i-1]
        pt2 = path_points[i]
        cv2.line(display, pt1, pt2, (0, 0, 255), 1)

    # Mevcut konum
    color = (0, 255, 0) if detected else (0, 255, 255)
    cv2.circle(display, current_pos, 6, color, -1)

    # Bilgi
    info = f"F:{frame_count}/{total_frames} | {method} | Lost:{lost_count}"
    cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Tracking", display)

    # Debug: diff goster
    diff_display = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    cv2.circle(diff_display, current_pos, 6, (0, 255, 0), 2)
    cv2.imshow("Difference", diff_display)

    # Onceki frame guncelle
    gray_prev = gray.copy()

    # Hareketsizlik kontrolu
    if len(path_points) >= 2:
        last_move = np.sqrt((path_points[-1][0] - path_points[-2][0])**2 +
                           (path_points[-1][1] - path_points[-2][1])**2)
        if last_move < 2 and detected:
            if no_move_start is None:
                no_move_start = time.time()
            elif time.time() - no_move_start > NO_MOVEMENT_TIMEOUT:
                print("Parcacik durdu.")
                break
        else:
            no_move_start = None

    # Cok uzun kayip
    if lost_count > MAX_LOST_FRAMES:
        print(f"Parcacik {MAX_LOST_FRAMES} frame kayip.")
        break

    # Silindir dibine ulasti mi?
    if current_pos[1] >= CYLINDER_BOTTOM_Y:
        print(f"Parcacik silindir dibine ulasti (Y={current_pos[1]}).")
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)

# =============================================
# SONUCLAR
# =============================================
csv_file.close()
cap.release()
cv2.destroyAllWindows()

# Son frame'e yol ciz
cap = cv2.VideoCapture("output.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
ret, final_frame = cap.read()
cap.release()

if ret and path_points:
    for i in range(1, len(path_points)):
        cv2.line(final_frame, path_points[i-1], path_points[i], (0, 0, 255), 1)
        cv2.circle(final_frame, path_points[i], 2, (0, 0, 255), -1)

    cv2.imwrite(os.path.join(output_dir, "transparent_tracked_path.jpg"), final_frame)
    print("Yol kaydedildi.")

# Istatistikler
if tracking_start and tracking_end and len(path_points) > 1:
    total_time = tracking_end - tracking_start
    start_y = path_points[0][1]
    end_y = path_points[-1][1]
    vertical_px = end_y - start_y

    print(f"\n=== SONUCLAR ===")
    print(f"Toplam Frame: {frame_count}")
    print(f"Sure: {total_time:.2f} s")
    print(f"Dikey Hareket: {vertical_px} piksel")
    print(f"Baslangic Y: {start_y}, Bitis Y: {end_y}")

# Vektor verisini kaydet
if vector_data:
    import matplotlib.pyplot as plt

    vector_csv_path = os.path.join(output_dir, 'transparent_vector_data.csv')
    with open(vector_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['X', 'Y', 'dX', 'dY'])
        writer.writerows(vector_data)
    print(f"Vektor verisi kaydedildi: {vector_csv_path}")

    # Optik akis analizi
    vector_arr = np.array(vector_data)
    magnitudes = np.sqrt(vector_arr[:, 2]**2 + vector_arr[:, 3]**2)

    mean_magnitude = np.mean(magnitudes)
    max_magnitude = np.max(magnitudes)
    mean_dx = np.mean(vector_arr[:, 2])
    mean_dy = np.mean(vector_arr[:, 3])

    print(f"\n=== OPTIK AKIS ANALIZI ===")
    print(f"Ortalama Hareket Yogunlugu: {mean_magnitude:.4f} piksel/frame")
    print(f"Maksimum Hareket: {max_magnitude:.4f} piksel/frame")
    print(f"Ortalama dX (yatay): {mean_dx:.2f} piksel/frame")
    print(f"Ortalama dY (dikey): {mean_dy:.2f} piksel/frame")

    # Gercek hiz tahmini (piksel/saniye)
    px_per_sec = mean_magnitude * fps
    print(f"Hiz: {px_per_sec:.1f} piksel/saniye")

    # Grafik olustur
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    t = np.arange(len(magnitudes)) / fps

    # 1. Hiz buyuklugu
    axes[0, 0].plot(t, magnitudes, 'g-', linewidth=1)
    axes[0, 0].axhline(y=mean_magnitude, color='r', linestyle='--', label=f'Ort: {mean_magnitude:.1f}')
    axes[0, 0].set_xlabel('Zaman (s)')
    axes[0, 0].set_ylabel('Hiz (piksel/frame)')
    axes[0, 0].set_title('Hareket Yogunlugu vs Zaman')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. dX ve dY
    axes[0, 1].plot(t, vector_arr[:, 2], 'b-', label='dX (yatay)', alpha=0.8)
    axes[0, 1].plot(t, vector_arr[:, 3], 'r-', label='dY (dikey)', alpha=0.8)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].set_xlabel('Zaman (s)')
    axes[0, 1].set_ylabel('Hareket (piksel/frame)')
    axes[0, 1].set_title('Yatay ve Dikey Hareket')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Yol
    axes[1, 0].plot(vector_arr[:, 0], vector_arr[:, 1], 'b-', linewidth=1)
    axes[1, 0].scatter(vector_arr[0, 0], vector_arr[0, 1], c='green', s=100, zorder=5, label='Baslangic')
    axes[1, 0].scatter(vector_arr[-1, 0], vector_arr[-1, 1], c='red', s=100, zorder=5, label='Bitis')
    axes[1, 0].set_xlabel('X (piksel)')
    axes[1, 0].set_ylabel('Y (piksel)')
    axes[1, 0].set_title('Parcacik Yolu')
    axes[1, 0].invert_yaxis()
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. FFT (salinim analizi)
    x_pos = vector_arr[:, 0]
    x_detrended = x_pos - np.polyval(np.polyfit(t, x_pos, 1), t)
    n = len(x_detrended)
    fft_result = np.fft.fft(x_detrended)
    fft_magnitude = np.abs(fft_result) / n * 2
    frequencies = np.fft.fftfreq(n, 1/fps)
    pos_mask = frequencies > 0
    pos_freq = frequencies[pos_mask]
    pos_mag = fft_magnitude[pos_mask]

    axes[1, 1].stem(pos_freq[:25], pos_mag[:25], basefmt=' ')
    if len(pos_mag) > 0:
        dominant_idx = np.argmax(pos_mag)
        dominant_freq = pos_freq[dominant_idx]
        axes[1, 1].axvline(x=dominant_freq, color='r', linestyle='--', label=f'Dominant: {dominant_freq:.2f} Hz')
        print(f"Dominant Salinim Frekansi: {dominant_freq:.3f} Hz")
    axes[1, 1].set_xlabel('Frekans (Hz)')
    axes[1, 1].set_ylabel('Genlik')
    axes[1, 1].set_title('FFT Salinim Analizi')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 10)

    plt.tight_layout()
    analysis_path = os.path.join(output_dir, 'transparent_fft_analysis.png')
    plt.savefig(analysis_path, dpi=150)
    plt.close()
    print(f"Analiz grafigi kaydedildi: {analysis_path}")

print("\nTamamlandi!")
