import sys
import os
import csv
import cv2
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout, QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer


class StrawberryDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🍓 Strawberry Ripeness Detector")
        self.model = None
        self.class_names = []
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.video_label = QLabel("🔴 Camera Feed")
        self.video_label.setFixedSize(640, 640)

        self.load_model_btn = QPushButton("📦 Load Model")
        self.load_video_btn = QPushButton("📂 Open Video")
        self.open_cam_btn = QPushButton("🎥 Open Camera")
        self.open_image_btn = QPushButton("🖼️ Open Image")
        self.stop_btn = QPushButton("⛔ Stop")

        layout = QGridLayout()
        layout.addWidget(self.video_label, 0, 0, 1, 2)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.load_model_btn)
        btn_layout.addWidget(self.load_video_btn)
        btn_layout.addWidget(self.open_cam_btn)
        btn_layout.addWidget(self.open_image_btn)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout, 1, 0, 1, 2)

        self.setLayout(layout)

        self.load_model_btn.clicked.connect(self.load_model)
        self.load_video_btn.clicked.connect(self.load_video)
        self.open_cam_btn.clicked.connect(self.open_camera)
        self.open_image_btn.clicked.connect(self.load_image)
        self.stop_btn.clicked.connect(self.stop_video)

        # Create folders if not exist
        os.makedirs("results/center_points/coords", exist_ok=True)

        # CSV path
        self.center_csv_path = "results/center_points/coords/center_points.csv"
        with open(self.center_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "class", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"])

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select YOLOv8 Model (.pt)", "", "Model Files (*.pt)")
        if model_path:
            try:
                self.model = YOLO(model_path)
                self.class_names = self.model.names
            except Exception as e:
                print("Failed to load model:", e)

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.timer.start(30)

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def load_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.png *.jpeg)")
        if image_path and self.model:
            frame = cv2.imread(image_path)
            self.process_frame(frame)

    def stop_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.stop_video()
                return
            self.process_frame(frame)

    def process_frame(self, frame):
        frame = cv2.resize(frame, (640, 640))

        results = self.model.predict(source=frame, conf=0.25, verbose=False)
        result = results[0]

        if not result.boxes:
            return  # Không hiển thị nếu không phát hiện đối tượng

        annotated_frame = frame.copy()

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = self.class_names[cls_id]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Vẽ chấm đỏ tại trung tâm đối tượng
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 255), -1)

            # Ghi vào CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{class_name}_{timestamp}.jpg"

            with open(self.center_csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([filename, class_name, round(conf, 2), x1, y1, x2, y2, center_x, center_y])

        # Hiển thị khung hình lên giao diện
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qimg = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StrawberryDetector()
    window.show()
    sys.exit(app.exec_())
