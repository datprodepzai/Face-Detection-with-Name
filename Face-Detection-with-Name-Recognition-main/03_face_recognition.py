import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Khởi tạo recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['None', 'Dat']

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("800x600")
        self.root.config(bg="#2e2e2e")  # Màu nền giao diện

        # Khung cho video và ảnh
        self.display_frame = tk.Frame(root, bg="#444444")
        self.display_frame.grid(row=0, column=0, padx=10, pady=10, rowspan=6)

        self.video_label = tk.Label(self.display_frame, bg="#444444")
        self.video_label.pack()

        self.image_label = tk.Label(self.display_frame, bg="#444444")
        self.image_label.pack()

        # Nút để bắt đầu video từ camera
        self.video_btn = tk.Button(root, text="Start Video", command=self.start_video, bg="#5cb85c", fg="white")
        self.video_btn.grid(row=0, column=1, padx=10, pady=10)

        # Nút để tải ảnh
        self.image_btn = tk.Button(root, text="Upload Image", command=self.upload_image, bg="#5bc0de", fg="white")
        self.image_btn.grid(row=1, column=1, padx=10, pady=10)

        # Nút để tải video
        self.video_file_btn = tk.Button(root, text="Upload Video", command=self.upload_video, bg="#5bc0de", fg="white")
        self.video_file_btn.grid(row=2, column=1, padx=10, pady=10)

        # Nút để thoát
        self.exit_btn = tk.Button(root, text="Exit", command=root.quit, bg="#d9534f", fg="white")
        self.exit_btn.grid(row=3, column=1, padx=10, pady=10)

        self.video_source = None

    def start_video(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 640)
        self.cam.set(4, 480)
        self.update_video()

    def update_video(self):
        if self.video_source is not None:
            ret, img = self.video_source.read()
            if not ret:
                return
        else:
            ret, img = self.cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

        # Hiển thị video
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.img_tk = img_tk
        self.video_label.config(image=img_tk)

        self.video_label.after(10, self.update_video)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                if confidence < 100:
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

            # Hiển thị ảnh
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)
            self.image_label.img_tk = img_tk
            self.image_label.config(image=img_tk)

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.video_source = cv2.VideoCapture(file_path)
            self.update_video()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
