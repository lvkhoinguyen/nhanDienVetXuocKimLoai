import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ====== CẤU HÌNH ======
IMAGE_SIZE = 128
TEST_PATH = "dataset/test/"
MODEL_PATH = "models/"
NUM_SAMPLES = 5  # ảnh hiển thị mỗi lớp

# ====== TẢI ẢNH ======
def load_images(path, size=128, num=5):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith((".jpg", ".png"))][:num]
    imgs = []
    for p in image_paths:
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size, size))
        img = img.astype('float32') / 255.0
        imgs.append(img)
    return np.array(imgs)

# ====== DỰ ĐOÁN VÀ ĐÁNH GIÁ ======
defect_classes = os.listdir(TEST_PATH)

for cls in defect_classes:
    print(f"\n🔎 Đang đánh giá lớp: {cls}")
    model_file = os.path.join(MODEL_PATH, f"{cls}_autoencoder.h5")
    test_folder = os.path.join(TEST_PATH, cls)

    if not os.path.exists(model_file):
        print(f"⚠️ Không tìm thấy mô hình cho lớp {cls}, bỏ qua.")
        continue

    model = load_model(model_file)
    X_test = load_images(test_folder, IMAGE_SIZE, NUM_SAMPLES)
    preds = model.predict(X_test)

    for i in range(len(X_test)):
        original = X_test[i]
        reconstructed = preds[i]
        error = np.mean((original - reconstructed) ** 2)

        print(f"Ảnh {i+1} - MSE: {error:.5f}")

        # Hiển thị
        diff = np.abs(original - reconstructed)

        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(original)
        plt.title("Ảnh gốc")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(reconstructed)
        plt.title("Tái tạo")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(diff)
        plt.title(f"Sai số\nMSE: {error:.5f}")
        plt.axis("off")

        plt.suptitle(f"Lớp lỗi: {cls}")
        plt.tight_layout()
        plt.show()
