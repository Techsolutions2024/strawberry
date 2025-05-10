import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("best.pt")  # đảm bảo file best.pt nằm cùng thư mục hoặc sửa đường dẫn

# Create output directory
os.makedirs("results/center_points/coords", exist_ok=True)
csv_path = "results/center_points/coords/center_points.csv"

# Tạo file CSV và ghi header nếu chưa tồn tại
if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"])

# Mở webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không mở được webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize về 640x640
    frame = cv2.resize(frame, (640, 640))

    # Dự đoán
    results = model.predict(source=frame, conf=0.25, verbose=False)
    result = results[0]

    if not result.boxes:
        continue  # Không hiển thị nếu không phát hiện gì

    # Vẽ khung hình và ghi tọa độ
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Vẽ dấu chấm tại tọa độ trung tâm
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # Ghi vào CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{class_name}_{timestamp}.jpg"

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([filename, class_name, round(conf, 2), x1, y1, x2, y2, center_x, center_y])

    # Hiển thị frame
    cv2.imshow("Detections", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
