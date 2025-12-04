import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

# Video okuma
cap = cv2.VideoCapture("output.mp4")
ret, old_frame = cap.read()
if not ret:
    print("Video açılamadı")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select ROI", 640, 480)
roi = cv2.selectROI("Select ROI", old_frame, False)
cv2.destroyWindow("Select ROI")

# ROI içinde takip noktaları
roi_x, roi_y, roi_w, roi_h = roi
mask = np.zeros_like(old_gray)
mask[int(roi_y):int(roi_y + roi_h), int(roi_x):int(roi_x + roi_w)] = 255
p0 = cv2.goodFeaturesToTrack(old_gray, 25, 0.3, 7, mask=mask)

# Vektör verilerini saklamak için listeler
vector_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)

    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            vector_data.append([c, d, a - c, b - d])  # [x, y, dx, dy]

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Vektör görselleştirme
plt.figure(figsize=(10, 10))
vector_data = np.array(vector_data)
plt.quiver(vector_data[:, 0], vector_data[:, 1],
           vector_data[:, 2], vector_data[:, 3],
           angles='xy', scale_units='xy', scale=1,
           color='b', width=0.003)

plt.gca().invert_yaxis()  # Y eksenini ters çevir (görüntü koordinat sistemi için)
plt.title('Motion Vectors')
plt.savefig('vector_visualization.png')
plt.close()


# Ortalama hareket büyüklüğünü hesapla
magnitudes = np.sqrt(vector_data[:, 2]**2 + vector_data[:, 3]**2)
mean_magnitude = np.mean(magnitudes)
print(f"Ortalama Hareket Yoğunluğu: {mean_magnitude:.4f}")

# Vektör verilerini CSV'ye kaydet
with open('vector_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Start_X', 'Start_Y', 'Vector_X', 'Vector_Y'])
    writer.writerows(vector_data)