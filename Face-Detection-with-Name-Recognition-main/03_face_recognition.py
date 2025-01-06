import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Khởi tạo recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX
names = ['None', 'Dat']  # Tên tương ứng với ID khuôn mặt


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("900x600")
        self.root.config(bg="#2e2e2e")

        # Khung hiển thị ảnh và video
        self.display_frame = tk.Frame(root, bg="#444444", width=640, height=480)
        self.display_frame.grid(row=0, column=0, padx=10, pady=10, rowspan=6)
        self.display_frame.grid_propagate(False)

        self.video_label = tk.Label(self.display_frame, bg="#444444")
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # Các nút chức năng
        self.video_btn = tk.Button(root, text="Start Video", command=self.start_video, bg="#5cb85c", fg="white")
        self.video_btn.grid(row=0, column=1, padx=10, pady=10)

        self.image_btn = tk.Button(root, text="Upload Image", command=self.upload_image, bg="#5bc0de", fg="white")
        self.image_btn.grid(row=1, column=1, padx=10, pady=10)

        self.video_file_btn = tk.Button(root, text="Upload Video", command=self.upload_video, bg="#5bc0de", fg="white")
        self.video_file_btn.grid(row=2, column=1, padx=10, pady=10)

        self.exit_btn = tk.Button(root, text="Exit", command=root.quit, bg="#d9534f", fg="white")
        self.exit_btn.grid(row=3, column=1, padx=10, pady=10)

        self.video_source = None  # Nguồn video
        self.cam = None  # Camera

    def resize_image(self, img, max_size=(640, 480)):

        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return img

    def start_video(self):

        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 640)  # Chiều rộng
        self.cam.set(4, 480)  # Chiều cao
        self.update_video()

    def update_video(self):

        if self.cam:
            ret, img = self.cam.read()
            if ret:
                self.process_frame(img)
            self.video_label.after(10, self.update_video)

    def stop_video(self):

        self.running = False
        if self.cam is not None and self.cam.isOpened():
            self.cam.release()
        if self.video_source is not None and self.video_source.isOpened():
            self.video_source.release()
    def upload_image(self):
        self.stop_video()
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            img = cv2.imread(file_path)
            self.process_frame(img)

    def upload_video(self):
        self.stop_video()
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.video_source = cv2.VideoCapture(file_path)
            self.update_video_file()

    def update_video_file(self):

        if self.video_source:
            ret, img = self.video_source.read()
            if ret:
                self.process_frame(img)
                self.video_label.after(10, self.update_video_file)
            else:
                self.video_source.release()
                self.video_source = None
                messagebox.showinfo("Video", "Video đã phát hết.")

    def process_frame(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        if len(faces) == 0:
            print("Face Detection", "Không tìm thấy khuôn mặt.")
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                if confidence < 100:
                    id = names[id]
                    confidence_text = f"{100 - confidence:.2f}%"
                else:
                    id = "Unknown"
                    confidence_text = f"{100 - confidence:.2f}%"

                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        # Hiển thị hình ảnh
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_resized = self.resize_image(img_pil)  # Điều chỉnh kích thước
        img_tk = ImageTk.PhotoImage(img_resized)
        self.video_label.img_tk = img_tk
        self.video_label.config(image=img_tk)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
