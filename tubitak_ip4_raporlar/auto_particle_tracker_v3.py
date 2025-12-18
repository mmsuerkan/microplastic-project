import cv2
import numpy as np
import csv
import time
import os
import sys

# =============================================
# OTOMATİK PARÇACIK TAKİP SİSTEMİ V3
# Değişiklik: Aralıklı frame tarama (video başı kırpılmamış durumlar için)
# - Silindir bölgesi %10-%90 (V2'den)
# - Her N frame'de bir parçacık arar (SCAN_INTERVAL)
# - Bulunca o noktadan takibe başlar
# =============================================

def auto_track_particle(video_path, output_dir="output_results", show_video=True, debug=False):
    """
    Videodan parçacığı otomatik tespit edip takip eder.

    V3 Değişiklikleri:
    - Aralıklı frame tarama: Video başı kırpılmamışsa parçacık geç görünebilir
    - SCAN_INTERVAL frame'de bir tarama yaparak hızlı tespit
    - Parçacık bulununca o frame'den takibe başlar

    Args:
        video_path: Video dosyası yolu
        output_dir: Çıktı klasörü
        show_video: Görüntüleme açık/kapalı
        debug: Debug modu

    Returns:
        dict: Sonuçlar (hız, süre, koordinatlar, vb.)
    """

    os.makedirs(output_dir, exist_ok=True)

    # Video aç
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"HATA: Video acilamadi: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 50
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"FPS: {fps}, Frames: {total_frames}, Boyut: {width}x{height}")

    # =============================================
    # PARAMETRELER (V3 - Aralıklı tarama)
    # =============================================
    # Silindir bolgesi - V2'den
    CYLINDER_LEFT = int(width * 0.10)
    CYLINDER_RIGHT = int(width * 0.90)
    CYLINDER_TOP = int(height * 0.05)
    CYLINDER_BOTTOM = int(height * 0.90)

    # V3: Aralıklı tarama parametreleri
    SCAN_INTERVAL = 10  # Her 10 frame'de bir tara
    MAX_START_Y = int(height * 0.5)  # Parçacık en fazla ekranın yarısında olmalı (üst yarı)

    # Tespit parametreleri
    DIFF_THRESHOLD = 25
    MIN_BLOB_AREA = 10
    MAX_BLOB_AREA = 2000

    # Takip parametreleri
    SEARCH_RADIUS = 50
    MAX_HORIZONTAL_MOVE = 15
    MAX_VERTICAL_MOVE = 20
    MIN_VERTICAL_MOVE = -5

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # =============================================
    # ARKA PLAN MODELI OLUSTUR (V3: Farklı bölgelerden örnek al)
    # =============================================
    print("Arka plan modeli olusturuluyor...")

    # Video boyunca farklı noktalardan frame al (daha robust arka plan)
    bg_frames = []
    sample_points = [0, total_frames//4, total_frames//2, 3*total_frames//4]

    for start_frame in sample_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(3):  # Her noktadan 3 frame
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = clahe.apply(gray)
                bg_frames.append(gray.astype(np.float32))

    if not bg_frames:
        print("HATA: Frame okunamadi")
        return None

    background = np.median(bg_frames, axis=0).astype(np.uint8)  # Median daha robust

    # =============================================
    # V3: ARALIKLI PARCACIK TESPITI
    # Video başı kırpılmamış olabilir, aralıklı tara
    # =============================================
    print(f"Parcacik araniyor (aralik: {SCAN_INTERVAL} frame)...")

    particle_found = False
    initial_pos = None
    detection_frame = 0
    best_candidate = None
    best_y = float('inf')  # En üstteki (en küçük Y) parçacığı bul

    # Aralıklı tarama - her SCAN_INTERVAL frame'de bir bak
    scan_frames = list(range(0, total_frames, SCAN_INTERVAL))

    for frame_idx in scan_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        # Arka plan farki
        diff = cv2.absdiff(background, gray)
        _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Morfoloji
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Sadece silindir bolgesinde ara
        mask = np.zeros_like(thresh)
        mask[CYLINDER_TOP:CYLINDER_BOTTOM, CYLINDER_LEFT:CYLINDER_RIGHT] = 255
        thresh = cv2.bitwise_and(thresh, mask)

        # Konturlari bul
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Uygun boyutta blob ara
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Üst yarıda mı ve şu ana kadar bulunanlardan daha yukarıda mı?
                    if cy < MAX_START_Y and cy < best_y:
                        best_candidate = (cx, cy)
                        best_y = cy
                        detection_frame = frame_idx
                        particle_found = True

    if particle_found:
        initial_pos = best_candidate
        print(f"Parcacik bulundu! Frame {detection_frame}, Konum: {initial_pos}")

    if not particle_found:
        print("HATA: Parcacik tespit edilemedi!")
        cap.release()
        return None

    print(f"Takip baslangici: Frame {detection_frame}, Konum: {initial_pos}")

    # =============================================
    # TAKİP BASLAT
    # =============================================
    print("Takip basliyor...")

    cap.set(cv2.CAP_PROP_POS_FRAMES, detection_frame)

    current_pos = initial_pos
    path_points = [current_pos]
    vector_data = []
    lost_frames = 0
    tracking_start = time.time()

    prev_gray = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        # Arka plan farki
        diff = cv2.absdiff(background, gray)
        _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Arama bolgesi (onceki konumun cevresi)
        search_mask = np.zeros_like(thresh)
        cv2.circle(search_mask, current_pos, SEARCH_RADIUS, 255, -1)
        search_region = cv2.bitwise_and(thresh, search_mask)

        # Kontur bul
        contours, _ = cv2.findContours(search_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        new_pos = None
        best_dist = float('inf')

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Hareket kontrolu
                    dx = cx - current_pos[0]
                    dy = cy - current_pos[1]

                    # Sadece asagi veya hafif yukari hareket kabul
                    if dy >= MIN_VERTICAL_MOVE and dy <= MAX_VERTICAL_MOVE:
                        if abs(dx) <= MAX_HORIZONTAL_MOVE:
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist < best_dist:
                                best_dist = dist
                                new_pos = (cx, cy)

        if new_pos is not None:
            # Vektor verisi
            dx = new_pos[0] - current_pos[0]
            dy = new_pos[1] - current_pos[1]
            vector_data.append([new_pos[0], new_pos[1], dx, dy])

            current_pos = new_pos
            path_points.append(current_pos)
            lost_frames = 0
        else:
            # Tahmin kullan
            if len(vector_data) >= 3:
                avg_dx = np.mean([v[2] for v in vector_data[-5:]])
                avg_dy = np.mean([v[3] for v in vector_data[-5:]])
                if avg_dy < 2:
                    avg_dy = 3
                pred_x = int(current_pos[0] + avg_dx * 0.2)
                pred_y = int(current_pos[1] + avg_dy)
                current_pos = (pred_x, pred_y)
                path_points.append(current_pos)
                vector_data.append([current_pos[0], current_pos[1], avg_dx * 0.2, avg_dy])

            lost_frames += 1

        # Gorsellestirme
        if show_video:
            display = frame.copy()

            # Yol ciz
            for i in range(1, len(path_points)):
                cv2.line(display, path_points[i-1], path_points[i], (0, 0, 255), 1)

            cv2.circle(display, current_pos, 5, (0, 255, 0), -1)

            info = f"Frame: {frame_count} | Pos: {current_pos} | Lost: {lost_frames}"
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Auto Tracking V3", display)

            if debug:
                cv2.imshow("Threshold", thresh)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        frame_count += 1
        prev_gray = gray.copy()

        # Bitis kosullari
        if current_pos[1] >= CYLINDER_BOTTOM:
            print("Parcacik silindir dibine ulasti.")
            break

        if lost_frames > 50:
            print("Parcacik kayboldu.")
            break

    tracking_end = time.time()
    cap.release()
    if show_video:
        cv2.destroyAllWindows()

    # =============================================
    # SONUCLARI HESAPLA
    # =============================================
    if len(path_points) < 2:
        print("HATA: Yeterli veri yok")
        return None

    total_time = tracking_end - tracking_start
    start_y = path_points[0][1]
    end_y = path_points[-1][1]
    vertical_pixels = end_y - start_y

    # Piksel/metre orani - kolon yuksekligi 28.5 cm, goruntuDeki piksel yuksekligi
    COLUMN_HEIGHT_M = 0.285  # 28.5 cm
    FRAME_HEIGHT_PX = height  # Video frame yuksekligi (piksel)
    PIXELS_PER_METER = FRAME_HEIGHT_PX / COLUMN_HEIGHT_M

    # Optik akis analizi
    if vector_data:
        vector_arr = np.array(vector_data)
        magnitudes = np.sqrt(vector_arr[:, 2]**2 + vector_arr[:, 3]**2)
        mean_magnitude = np.mean(magnitudes)
        mean_dy = np.mean(vector_arr[:, 3])
    else:
        mean_magnitude = 0
        mean_dy = 0

    # Hiz hesabi - gercek kat edilen mesafe ve gercek takip suresi
    video_duration = total_frames / fps
    tracking_frames = len(path_points)
    tracking_duration = tracking_frames / fps  # Parcacigin takip edildigi sure

    # Gercek kat edilen dikey mesafe (metre)
    actual_distance_m = abs(vertical_pixels) / PIXELS_PER_METER

    # Hiz = kat edilen mesafe / takip suresi
    speed_mps = actual_distance_m / tracking_duration if tracking_duration > 0 else 0

    results = {
        'video_path': video_path,
        'total_frames': frame_count,
        'tracking_time': total_time,
        'video_duration': video_duration,
        'tracking_duration': tracking_duration,
        'start_pos': path_points[0],
        'end_pos': path_points[-1],
        'vertical_pixels': vertical_pixels,
        'actual_distance_m': actual_distance_m,
        'horizontal_drift': path_points[-1][0] - path_points[0][0],
        'speed_mps': speed_mps,
        'mean_magnitude': mean_magnitude,
        'mean_dy': mean_dy,
        'path_points': path_points,
        'vector_data': vector_data,
        'detection_frame': detection_frame,  # V3: Hangi frame'de bulundu
        'pixels_per_meter': PIXELS_PER_METER
    }

    # Sonuclari yazdir
    print(f"\n{'='*50}")
    print("SONUCLAR (V3)")
    print(f"{'='*50}")
    print(f"Tespit Frame: {detection_frame}")
    print(f"Toplam Frame: {frame_count}")
    print(f"Video Suresi: {video_duration:.2f} s")
    print(f"Takip Suresi: {tracking_duration:.2f} s ({tracking_frames} frame)")
    print(f"Dikey Hareket: {vertical_pixels} piksel ({actual_distance_m*100:.2f} cm)")
    print(f"Yatay Sapma: {results['horizontal_drift']} piksel")
    print(f"Ortalama Hareket Yogunlugu: {mean_magnitude:.4f} px/frame")
    print(f"Ortalama dY: {mean_dy:.2f} px/frame")
    print(f"Hiz: {speed_mps:.4f} m/s ({speed_mps*100:.2f} cm/s)")
    print(f"{'='*50}")

    # CSV kaydet
    csv_path = os.path.join(output_dir, 'auto_tracking_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'X', 'Y', 'dX', 'dY', 'Magnitude'])
        for i, (x, y, dx, dy) in enumerate(vector_data):
            mag = np.sqrt(dx**2 + dy**2)
            writer.writerow([i, x, y, dx, dy, mag])
    print(f"Veri kaydedildi: {csv_path}")

    # Ozet CSV kaydet
    summary_path = os.path.join(output_dir, 'summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metrik', 'Deger', 'Birim'])
        writer.writerow(['Video', os.path.basename(video_path), ''])
        writer.writerow(['Tracker Version', 'V3', ''])
        writer.writerow(['Tespit Frame', detection_frame, 'frame'])
        writer.writerow(['Toplam Frame', frame_count, 'frame'])
        writer.writerow(['Takip Frame', tracking_frames, 'frame'])
        writer.writerow(['Video Suresi', f'{video_duration:.2f}', 'saniye'])
        writer.writerow(['Takip Suresi', f'{tracking_duration:.2f}', 'saniye'])
        writer.writerow(['Baslangic X', path_points[0][0], 'piksel'])
        writer.writerow(['Baslangic Y', path_points[0][1], 'piksel'])
        writer.writerow(['Bitis X', path_points[-1][0], 'piksel'])
        writer.writerow(['Bitis Y', path_points[-1][1], 'piksel'])
        writer.writerow(['Dikey Hareket', vertical_pixels, 'piksel'])
        writer.writerow(['Kat Edilen Mesafe', f'{actual_distance_m*100:.2f}', 'cm'])
        writer.writerow(['Yatay Sapma', results['horizontal_drift'], 'piksel'])
        writer.writerow(['Mean Magnitude', f'{mean_magnitude:.4f}', 'px/frame'])
        writer.writerow(['Ortalama dY', f'{mean_dy:.4f}', 'px/frame'])
        writer.writerow(['Hiz', f'{speed_mps*100:.2f}', 'cm/s'])
    print(f"Ozet kaydedildi: {summary_path}")

    # Yol gorseli kaydet
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, final_frame = cap.read()
    cap.release()

    if ret:
        for i in range(1, len(path_points)):
            cv2.line(final_frame, path_points[i-1], path_points[i], (0, 0, 255), 1)
            cv2.circle(final_frame, path_points[i], 2, (0, 0, 255), -1)

        img_path = os.path.join(output_dir, 'auto_tracked_path.jpg')
        cv2.imwrite(img_path, final_frame)
        print(f"Gorsel kaydedildi: {img_path}")

    # Optik akis vektor gorseli
    if vector_data and len(vector_data) > 5:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # 1. Vektor alani
        ax1 = axes[0, 0]
        vector_arr = np.array(vector_data)
        step = max(1, len(vector_arr) // 50)
        ax1.quiver(vector_arr[::step, 0], -vector_arr[::step, 1],
                   vector_arr[::step, 2], -vector_arr[::step, 3],
                   angles='xy', scale_units='xy', scale=0.5, color='blue', alpha=0.7)
        ax1.set_xlabel('X (piksel)')
        ax1.set_ylabel('Y (piksel)')
        ax1.set_title('Optik Akis Vektor Alani')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # 2. dX ve dY zaman serisi
        ax2 = axes[0, 1]
        frames = np.arange(len(vector_arr))
        pixel_to_meter = KNOWN_DISTANCE_M / vertical_pixels if vertical_pixels > 0 else 0.285 / 700
        dx_ms = vector_arr[:, 2] * fps * pixel_to_meter
        dy_ms = vector_arr[:, 3] * fps * pixel_to_meter
        ax2.plot(frames, dx_ms, 'b-', label='dX (yatay)', alpha=0.7)
        ax2.plot(frames, dy_ms, 'r-', label='dY (dikey)', alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Hiz (m/s)')
        ax2.set_title('Yatay ve Dikey Hiz')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Hareket buyuklugu
        ax3 = axes[1, 0]
        magnitudes = np.sqrt(vector_arr[:, 2]**2 + vector_arr[:, 3]**2)
        ax3.plot(frames, magnitudes, 'g-', alpha=0.7)
        ax3.axhline(y=np.mean(magnitudes), color='r', linestyle='--',
                    label=f'Ortalama: {np.mean(magnitudes):.2f}')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Buyukluk (piksel)')
        ax3.set_title('Hareket Buyuklugu (Magnitude)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Yol haritasi
        ax4 = axes[1, 1]
        path_arr = np.array(path_points)
        ax4.plot(path_arr[:, 0], -path_arr[:, 1], 'r-', linewidth=1, alpha=0.7)
        ax4.scatter(path_arr[0, 0], -path_arr[0, 1], c='green', s=100, marker='o', label='Baslangic', zorder=5)
        ax4.scatter(path_arr[-1, 0], -path_arr[-1, 1], c='red', s=100, marker='x', label='Bitis', zorder=5)
        ax4.set_xlabel('X (piksel)')
        ax4.set_ylabel('Y (piksel)')
        ax4.set_title('Parcacik Yolu')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')

        # 5. FFT - dX
        ax5 = axes[0, 2]
        N = len(vector_arr)
        T = 1.0 / fps
        dx_values = vector_arr[:, 2]
        dx_fft = np.fft.fft(dx_values - np.mean(dx_values))
        freqs = np.fft.fftfreq(N, T)[:N//2]
        dx_power = 2.0/N * np.abs(dx_fft[0:N//2])
        ax5.plot(freqs, dx_power, 'b-', alpha=0.7)
        if len(dx_power) > 1:
            dominant_idx = np.argmax(dx_power[1:]) + 1
            dominant_freq_dx = freqs[dominant_idx]
            ax5.axvline(x=dominant_freq_dx, color='r', linestyle='--',
                        label=f'Dominant: {dominant_freq_dx:.2f} Hz')
        ax5.set_xlabel('Frekans (Hz)')
        ax5.set_ylabel('Genlik')
        ax5.set_title('FFT - Yatay Salinium (dX)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, fps/4)

        # 6. FFT - dY
        ax6 = axes[1, 2]
        dy_values = vector_arr[:, 3]
        dy_fft = np.fft.fft(dy_values - np.mean(dy_values))
        dy_power = 2.0/N * np.abs(dy_fft[0:N//2])
        ax6.plot(freqs, dy_power, 'r-', alpha=0.7)
        if len(dy_power) > 1:
            dominant_idx_dy = np.argmax(dy_power[1:]) + 1
            dominant_freq_dy = freqs[dominant_idx_dy]
            ax6.axvline(x=dominant_freq_dy, color='b', linestyle='--',
                        label=f'Dominant: {dominant_freq_dy:.2f} Hz')
        ax6.set_xlabel('Frekans (Hz)')
        ax6.set_ylabel('Genlik')
        ax6.set_title('FFT - Dikey Salinium (dY)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, fps/4)

        plt.tight_layout()
        vector_img_path = os.path.join(output_dir, 'optical_flow_vectors.png')
        plt.savefig(vector_img_path, dpi=150)
        plt.close()
        print(f"Optik akis gorseli kaydedildi: {vector_img_path}")

    return results


# =============================================
# ANA PROGRAM
# =============================================
if __name__ == "__main__":
    video_path = "output.mp4"

    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    results = auto_track_particle(video_path, show_video=True, debug=False)

    if results:
        print("\nBasarili!")
    else:
        print("\nBasarisiz!")
