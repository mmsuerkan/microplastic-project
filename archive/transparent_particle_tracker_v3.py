import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import time
import os

# =============================================
# ŞEFFAF PARÇACIK TAKİP SİSTEMİ v3.0
# Template Matching + Hareket Kısıtlaması
# =============================================

output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)

total_distance_meters = 0.285

cap = cv2.VideoCapture("output.mp4")
ret, frame = cap.read()
if not ret:
    print("Video açılamadı")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 50
print(f"Video FPS: {fps}")

# =============================================
# PARAMETRELER
# =============================================
# Hareket kısıtlamaları (parçacık AŞAĞI düşüyor)
MAX_HORIZONTAL_MOVEMENT = 15   # Yatay hareket limiti (piksel/frame)
MIN_VERTICAL_MOVEMENT = 0      # Minimum dikey hareket
MAX_VERTICAL_MOVEMENT = 30     # Maksimum dikey hareket
EXPECTED_DIRECTION = 1         # 1 = aşağı, -1 = yukarı

# Template matching
TEMPLATE_MATCH_THRESHOLD = 0.5  # Eşleşme eşiği
SEARCH_REGION_PADDING = 50      # Arama bölgesi genişliği

# Takip
NO_MOVEMENT_TIMEOUT = 3
LOST_FRAME_THRESHOLD = 50

# CLAHE
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# =============================================
# ROI SEÇİMİ VE TEMPLATE OLUŞTURMA
# =============================================
cv2.namedWindow("ROI Selection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ROI Selection", 1280, 720)
print("Parçacığı seçin ve ENTER'a basın...")
roi = cv2.selectROI("ROI Selection", frame, False)
cv2.destroyWindow("ROI Selection")

roi_x, roi_y, roi_w, roi_h = [int(v) for v in roi]
print(f"Seçilen ROI: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

# Template oluştur
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_frame = clahe.apply(gray_frame)
template = gray_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
template_h, template_w = template.shape

# Template'i birden fazla scale için hazırla
scales = [0.8, 0.9, 1.0, 1.1, 1.2]

print(f"Template boyutu: {template_w}x{template_h}")

# Başlangıç merkezi
current_center = (roi_x + roi_w // 2, roi_y + roi_h // 2)
prev_center = current_center

# =============================================
# TAKİP DEĞİŞKENLERİ
# =============================================
frame_count = 0
tracking_start_time = None
tracking_end_time = None
no_movement_start_time = None
path_points = [current_center]
last_valid_frame = frame.copy()
lost_frames = 0
consecutive_good_frames = 0

# Hız tahmini için
velocity_history = []
avg_velocity = (0, 5)  # Başlangıç tahmini (aşağı doğru)

# CSV
csv_path = os.path.join(output_dir, 'transparent_tracking_coordinates.csv')
csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'X', 'Y', 'Time', 'Confidence', 'Method'])

vector_data = []

# =============================================
# YARDIMCI FONKSİYONLAR
# =============================================
def is_valid_movement(old_pos, new_pos, avg_vel):
    """Hareketin fiziksel olarak mantıklı olup olmadığını kontrol et"""
    dx = new_pos[0] - old_pos[0]
    dy = new_pos[1] - old_pos[1]

    # Yatay hareket çok fazla mı?
    if abs(dx) > MAX_HORIZONTAL_MOVEMENT:
        return False, "horizontal_too_large"

    # Dikey hareket yönü doğru mu? (aşağı doğru düşmeli)
    if EXPECTED_DIRECTION == 1 and dy < MIN_VERTICAL_MOVEMENT:
        # Yukarı gitmemeli (çok az yukarı tolerans)
        if dy < -5:
            return False, "wrong_direction"

    # Dikey hareket çok fazla mı?
    if abs(dy) > MAX_VERTICAL_MOVEMENT:
        return False, "vertical_too_large"

    return True, "ok"


def find_particle_template_matching(gray, search_center, template, search_padding):
    """Template matching ile parçacık ara"""
    h, w = gray.shape
    th, tw = template.shape

    # Arama bölgesi
    x1 = max(0, search_center[0] - search_padding - tw//2)
    y1 = max(0, search_center[1] - search_padding - th//2)
    x2 = min(w, search_center[0] + search_padding + tw//2)
    y2 = min(h, search_center[1] + search_padding + th//2)

    search_region = gray[y1:y2, x1:x2]

    if search_region.shape[0] < th or search_region.shape[1] < tw:
        return None, 0

    best_match = None
    best_val = -1
    best_scale = 1.0

    # Multi-scale template matching
    for scale in scales:
        scaled_w = int(tw * scale)
        scaled_h = int(th * scale)

        if scaled_w < 5 or scaled_h < 5:
            continue
        if scaled_w > search_region.shape[1] or scaled_h > search_region.shape[0]:
            continue

        scaled_template = cv2.resize(template, (scaled_w, scaled_h))

        result = cv2.matchTemplate(search_region, scaled_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_match = (max_loc[0] + x1 + scaled_w//2, max_loc[1] + y1 + scaled_h//2)
            best_scale = scale

    return best_match, best_val


def predict_next_position(current_pos, velocity):
    """Sonraki konumu tahmin et"""
    return (
        int(current_pos[0] + velocity[0]),
        int(current_pos[1] + velocity[1])
    )


def update_velocity(velocity_history, new_velocity, max_history=10):
    """Ortalama hız güncelle"""
    velocity_history.append(new_velocity)
    if len(velocity_history) > max_history:
        velocity_history.pop(0)

    if len(velocity_history) > 0:
        avg_vx = sum(v[0] for v in velocity_history) / len(velocity_history)
        avg_vy = sum(v[1] for v in velocity_history) / len(velocity_history)
        return (avg_vx, avg_vy)
    return (0, 5)


# =============================================
# ANA DÖNGÜ
# =============================================
print("Takip başlıyor...")
print("Tuşlar: SPACE=duraklat, Q=çık, R=template yenile")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)

    # Sonraki konumu tahmin et
    predicted_pos = predict_next_position(current_center, avg_velocity)

    # Template matching ile ara
    match_pos, confidence = find_particle_template_matching(
        gray, predicted_pos, template, SEARCH_REGION_PADDING
    )

    detection_method = "template_matching"
    center = None

    if match_pos is not None and confidence > TEMPLATE_MATCH_THRESHOLD:
        # Hareket kontrolü
        is_valid, reason = is_valid_movement(current_center, match_pos, avg_velocity)

        if is_valid:
            center = match_pos
            consecutive_good_frames += 1
        else:
            # Geçersiz hareket - tahmin kullan
            detection_method = f"rejected_{reason}"
            lost_frames += 1
    else:
        lost_frames += 1

    # Eğer bulunamadıysa veya reddedildiyse
    if center is None:
        # Tahmin kullan ama sadece makul bir süre
        if lost_frames <= 10:
            center = predicted_pos
            detection_method = "prediction"
        else:
            # Frame differencing dene
            if frame_count > 0:
                old_gray = cv2.cvtColor(last_valid_frame, cv2.COLOR_BGR2GRAY)
                old_gray = clahe.apply(old_gray)

                diff = cv2.absdiff(old_gray, gray)
                _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

                # Sadece beklenen bölgede ara
                h, w = thresh.shape
                mask = np.zeros_like(thresh)
                px, py = predicted_pos
                cv2.circle(mask, (px, py), SEARCH_REGION_PADDING, 255, -1)
                thresh = cv2.bitwise_and(thresh, mask)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # En büyük konturu al
                    largest = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        is_valid, _ = is_valid_movement(current_center, (cx, cy), avg_velocity)
                        if is_valid:
                            center = (cx, cy)
                            detection_method = "frame_diff"
                            lost_frames = max(0, lost_frames - 1)

    # Hala bulunamadıysa
    if center is None:
        center = predicted_pos
        detection_method = "fallback_prediction"
        print(f"Frame {frame_count}: Kayıp ({lost_frames})")

    # Güncelle
    if tracking_start_time is None:
        tracking_start_time = time.time()
    tracking_end_time = time.time()

    # Hız güncelle
    dx = center[0] - current_center[0]
    dy = center[1] - current_center[1]

    if detection_method in ["template_matching", "frame_diff"]:
        avg_velocity = update_velocity(velocity_history, (dx, dy))
        lost_frames = 0

    prev_center = current_center
    current_center = center
    path_points.append(center)

    if detection_method in ["template_matching", "frame_diff"]:
        last_valid_frame = frame.copy()

    elapsed_time = tracking_end_time - tracking_start_time
    csv_writer.writerow([frame_count, center[0], center[1], elapsed_time,
                        confidence if match_pos else 0, detection_method])

    # Hareket vektörü
    if abs(dx) > 0 or abs(dy) > 0:
        vector_data.append([center[0], center[1], dx, dy])

    # Görselleştirme
    # Arama bölgesi
    cv2.rectangle(frame,
                  (predicted_pos[0] - SEARCH_REGION_PADDING, predicted_pos[1] - SEARCH_REGION_PADDING),
                  (predicted_pos[0] + SEARCH_REGION_PADDING, predicted_pos[1] + SEARCH_REGION_PADDING),
                  (100, 100, 100), 1)

    # Merkez noktaları
    color = (0, 255, 0) if detection_method == "template_matching" else (0, 255, 255)
    cv2.circle(frame, center, 5, color, -1)
    cv2.circle(frame, predicted_pos, 5, (255, 0, 255), 2)  # Tahmin (mor)

    # Yolu çiz
    for i in range(1, len(path_points)):
        cv2.line(frame, path_points[i-1], path_points[i], (0, 0, 255), 1)

    # Bilgi
    info = f"F:{frame_count} | {detection_method} | Lost:{lost_frames} | Vel:({avg_velocity[0]:.1f},{avg_velocity[1]:.1f})"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Hareketsizlik kontrolü
    movement = np.sqrt(dx**2 + dy**2)
    if movement < 2 and detection_method == "template_matching":
        if no_movement_start_time is None:
            no_movement_start_time = time.time()
        elif time.time() - no_movement_start_time > NO_MOVEMENT_TIMEOUT:
            print(f"Parçacık hareketsiz. Durduruluyor.")
            break
    else:
        no_movement_start_time = None

    # Çok uzun süre kayıp
    if lost_frames > LOST_FRAME_THRESHOLD:
        print(f"Parçacık {LOST_FRAME_THRESHOLD} frame kayıp. Durduruluyor.")
        break

    cv2.imshow("Tracking", frame)

    frame_count += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)
    elif key == ord('r'):
        # Template yenile
        print("Yeni template için ROI seçin...")
        new_roi = cv2.selectROI("Tracking", frame, False)
        if new_roi[2] > 0 and new_roi[3] > 0:
            rx, ry, rw, rh = [int(v) for v in new_roi]
            template = gray[ry:ry+rh, rx:rx+rw].copy()
            print("Template güncellendi!")

# =============================================
# SONUÇLAR
# =============================================
csv_file.close()
cap.release()
cv2.destroyAllWindows()

# Yol görselleştirme
if path_points:
    # Son frame'i kullan
    cap = cv2.VideoCapture("output.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    ret, final_frame = cap.read()
    cap.release()

    if ret:
        for i, pt in enumerate(path_points):
            cv2.circle(final_frame, pt, 2, (0, 0, 255), -1)
            if i > 0:
                cv2.line(final_frame, path_points[i-1], pt, (0, 0, 255), 1)

        cv2.imwrite(os.path.join(output_dir, "transparent_tracked_path.jpg"), final_frame)
        print(f"Yol kaydedildi: transparent_tracked_path.jpg")

# İstatistikler
if tracking_start_time and tracking_end_time:
    total_time = tracking_end_time - tracking_start_time

    # Sadece dikey mesafe hesapla (Y değişimi)
    if len(path_points) > 1:
        start_y = path_points[0][1]
        end_y = path_points[-1][1]
        vertical_pixels = abs(end_y - start_y)

        print(f"\n=== SONUÇLAR ===")
        print(f"Toplam Frame: {frame_count}")
        print(f"Takip Süresi: {total_time:.2f} s")
        print(f"Dikey Mesafe: {vertical_pixels} piksel")
        print(f"Ortalama Hız: {total_distance_meters/total_time:.4f} m/s")
        print(f"Piksel/Metre: {vertical_pixels/total_distance_meters:.1f} px/m")

# Vektör kaydet
if vector_data:
    with open(os.path.join(output_dir, 'transparent_vector_data.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['X', 'Y', 'dX', 'dY'])
        writer.writerows(vector_data)

print("\nTamamlandı!")
