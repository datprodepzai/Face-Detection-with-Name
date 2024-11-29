import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


# Hàm để tải và chuẩn bị hình ảnh cho việc dự đoán
def prepare_image(image_path, target_size=(256, 256)):
    # Đọc hình ảnh
    img = cv2.imread(image_path)

    # Chuyển đổi hình ảnh sang màu RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Thay đổi kích thước hình ảnh
    img_resized = cv2.resize(img_rgb, target_size)

    # Chuẩn hóa giá trị pixel từ 0-255 về 0-1
    img_normalized = img_resized / 255.0

    # Thêm chiều batch (mô hình yêu cầu 4 chiều dữ liệu: batch, height, width, channels)
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch


# Hàm để hiển thị hình ảnh
def display_image(image, title=''):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


# Hàm dự đoán hình ảnh với mô hình
def predict_image(model, image_path):
    # Chuẩn bị hình ảnh đầu vào
    img_batch = prepare_image(image_path)

    # Dự đoán với mô hình
    predicted_img = model.predict(img_batch)[0]

    # Chuyển đổi lại giá trị pixel về phạm vi [0, 255] cho dễ hiển thị
    predicted_img = np.clip(predicted_img * 255.0, 0, 255).astype(np.uint8)

    return predicted_img


# Giao diện Tkinter
class App:
    def __init__(self, root, model):
        self.root = root
        self.root.title("Dự đoán Hình ảnh với Mô hình")
        self.model = model

        # Nút chọn hình ảnh
        self.select_button = tk.Button(root, text="Chọn Hình ảnh", command=self.select_image)
        self.select_button.pack(pady=20)

        # Nhãn hiển thị hình ảnh đầu vào
        self.input_label = tk.Label(root, text="Hình ảnh đầu vào")
        self.input_label.pack(pady=10)

        self.input_canvas = tk.Canvas(root, width=300, height=300)
        self.input_canvas.pack()

        # Nhãn hiển thị hình ảnh dự đoán
        self.output_label = tk.Label(root, text="Hình ảnh Dự đoán")
        self.output_label.pack(pady=10)

        self.output_canvas = tk.Canvas(root, width=300, height=300)
        self.output_canvas.pack()

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.display_input_image(file_path)
            self.make_prediction(file_path)

    def display_input_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)

        # Cập nhật hình ảnh đầu vào lên giao diện
        self.input_canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.input_canvas.img = img_tk  # Giữ tham chiếu đến ảnh để không bị thu hồi

    def make_prediction(self, image_path):
        # Dự đoán hình ảnh
        predicted_img = predict_image(self.model, image_path)

        # Chuyển hình ảnh dự đoán thành đối tượng Tkinter
        img_pil = Image.fromarray(predicted_img)
        img_pil = img_pil.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img_pil)

        # Cập nhật hình ảnh dự đoán lên giao diện
        self.output_canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.output_canvas.img = img_tk  # Giữ tham chiếu đến ảnh để không bị thu hồi


# Khởi tạo mô hình (Giả sử mô hình đã huấn luyện và lưu trữ)
# Thay 'your_model_path' bằng đường dẫn đến mô hình đã lưu của bạn
model = tf.keras.models.load_model('final_model.h5')

# Tạo giao diện Tkinter
root = tk.Tk()
app = App(root, model)

# Chạy ứng dụng
root.mainloop()
