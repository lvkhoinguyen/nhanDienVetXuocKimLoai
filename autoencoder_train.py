import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# ========== CẤU HÌNH ==========
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
DATASET_PATH = "dataset/train/"
SAVE_MODEL_PATH = "models"

os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

# ========== HÀM LOAD ẢNH ==========
def load_images(path, size=128):
    image_paths = glob(os.path.join(path, "*.*"))
    data = []
    for img_path in image_paths:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (size, size))
        img = img.astype('float32') / 255.0
        data.append(img)
    return np.array(data)

# ========== HÀM XÂY DỰNG AUTOENCODER ==========
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Bottleneck
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(input_img, decoded)
    model.compile(optimizer=Adam(1e-4), loss='mse')
    return model

# ========== HUẤN LUYỆN CHO TỪNG CLASS ==========
defect_classes = os.listdir(DATASET_PATH)

for defect_class in defect_classes:
    class_path = os.path.join(DATASET_PATH, defect_class)
    print(f"\n📁 Huấn luyện cho lớp lỗi: {defect_class}")

    X = load_images(class_path, IMAGE_SIZE)
    if len(X) == 0:
        print("⚠️ Không tìm thấy ảnh. Bỏ qua.")
        continue

    model = build_autoencoder((IMAGE_SIZE, IMAGE_SIZE, 3))

    checkpoint_path = os.path.join(SAVE_MODEL_PATH, f"{defect_class}_autoencoder.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='loss', mode='min')

    history = model.fit(X, X,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        validation_split=0.1,
                        callbacks=[checkpoint])

    # Vẽ loss
    plt.figure(figsize=(6, 3))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss - {defect_class}')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_MODEL_PATH}/{defect_class}_loss.png")
    plt.close()

print("\n✅ Huấn luyện xong tất cả các lớp lỗi.")
