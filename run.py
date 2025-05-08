import sys
import os
import csv
import cv2
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout, QGridLayout, QScrollArea
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
        self.video_label.setFixedSize(1000, 800)

        self.crop_scroll_area = QScrollArea()
        self.crop_scroll_area.setWidgetResizable(True)
        self.crop_container = QWidget()
        self.crop_grid = QGridLayout()
        self.crop_container.setLayout(self.crop_grid)
        self.crop_scroll_area.setWidget(self.crop_container)

        self.load_model_btn = QPushButton("📦 Load Model")
        self.load_video_btn = QPushButton("📂 Open Video")
        self.open_cam_btn = QPushButton("🎥 Open Camera")
        self.stop_btn = QPushButton("⛔ Stop")

        layout = QGridLayout()
        layout.addWidget(self.video_label, 0, 0)
        layout.addWidget(self.crop_scroll_area, 0, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.load_model_btn)
        btn_layout.addWidget(self.load_video_btn)
        btn_layout.addWidget(self.open_cam_btn)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout, 1, 0, 1, 2)
        layout.addWidget(QLabel("🖼️ Detected Strawberry Crops:"), 2, 1)

        self.setLayout(layout)

        self.load_model_btn.clicked.connect(self.load_model)
        self.load_video_btn.clicked.connect(self.load_video)
        self.open_cam_btn.clicked.connect(self.open_camera)
        self.stop_btn.clicked.connect(self.stop_video)

        os.makedirs("results/images", exist_ok=True)
        os.makedirs("results/coords", exist_ok=True)
        self.csv_path = "results/coords/detections.csv"
        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "class", "confidence", "x1", "y1", "x2", "y2"])

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

            if self.model:
                results = self.model.predict(source=frame, conf=0.25, verbose=False)
                result = results[0]
                annotated_frame = result.plot()

                for i in reversed(range(self.crop_grid.count())):
                    widget_to_remove = self.crop_grid.itemAt(i).widget()
                    if widget_to_remove is not None:
                        widget_to_remove.setParent(None)

                col_count = 2
                row = 0
                col = 0

                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = self.class_names[cls_id]

                    crop_img = frame[y1:y2, x1:x2]
                    crop_img = cv2.resize(crop_img, (160, 120))
                    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{class_name}_{timestamp}.jpg"
                    save_path = os.path.join("results/images", filename)
                    cv2.imwrite(save_path, crop_rgb[..., ::-1])  # convert to BGR for saving

                    with open(self.csv_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([filename, class_name, round(conf, 2), x1, y1, x2, y2])

                    h, w, ch = crop_rgb.shape
                    qimg = QImage(crop_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)

                    label = QLabel()
                    label.setPixmap(pixmap)
                    label_text = QLabel(f"{class_name} ({conf:.2f})\n[{x1}, {y1}, {x2}, {y2}]")
                    vbox = QVBoxLayout()
                    vbox.addWidget(label)
                    vbox.addWidget(label_text)
                    container = QWidget()
                    container.setLayout(vbox)

                    self.crop_grid.addWidget(container, row, col)

                    col += 1
                    if col >= col_count:
                        col = 0
                        row += 1

                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qimg = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qimg))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StrawberryDetector()
    window.show()
    sys.exit(app.exec_())
