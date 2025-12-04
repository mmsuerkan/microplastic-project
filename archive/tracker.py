import cv2
import time
import csv

# Toplam mesafeyi 28,5 santimetre olarak kabul ediyoruz
total_distance_meters = 0.285

# Video dosyasını aç
cap = cv2.VideoCapture("output.mp4")

# İlk kareyi oku
ret, frame = cap.read()
if not ret:
    print("Kamera açılamadı veya video dosyası bulunamadı")
    exit()

cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking", 640, 480)
bbox = cv2.selectROI("Tracking", frame, False)

tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

# CSV dosyasını oluştur
with open('tracking_coordinates.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'X', 'Y', 'Time'])
    frame_count = 0

    # Takip süreleri
    tracking_start_time = None
    tracking_end_time = None

    prev_bbox = bbox
    no_movement_start_time = None
    path_points = []
    last_valid_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)

        if success:
            if tracking_start_time is None:
                tracking_start_time = time.time()
            tracking_end_time = time.time()

            last_valid_frame = frame.copy()
            center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
            path_points.append(center)

            current_time = tracking_end_time - tracking_start_time
            writer.writerow([frame_count, center[0], center[1], current_time])

            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            # Hareket kontrolü
            if int(bbox[0]) == int(prev_bbox[0]) and int(bbox[1]) == int(prev_bbox[1]):
                if no_movement_start_time is None:
                    no_movement_start_time = time.time()
                elif time.time() - no_movement_start_time > 2:
                    print("Nesne 2 saniye boyunca aynı konumda kaldı, videoyu bitiriyorum.")
                    break
            else:
                no_movement_start_time = None

            prev_bbox = bbox
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            print("Tracking failure detected, videoyu bitiriyorum.")
            break

        for point in path_points:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

        cv2.imshow('Tracking', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Son geçerli kareyi kaydet
    if last_valid_frame is not None:
        for point in path_points:
            cv2.circle(last_valid_frame, point, 2, (0, 255, 0), -1)
        cv2.imwrite("tracked_path_last_frame.jpg", last_valid_frame)
    else:
        print("Son geçerli kare bulunamadı, resim kaydedilemedi.")

cap.release()
cv2.destroyAllWindows()

# Ortalama hız hesapla
if tracking_start_time and tracking_end_time:
    total_tracking_time = tracking_end_time - tracking_start_time
    average_speed = total_distance_meters / total_tracking_time
    print(f"Toplam Mesafe: {total_distance_meters:.2f} m"
          f"\nToplam Takip Süresi: {total_tracking_time:.2f} s")
    print(f"Ortalama Hız: {average_speed:.2f} m/s")
else:
    print("Takip süresi ölçülemedi, ortalama hız hesaplanamadı.")
