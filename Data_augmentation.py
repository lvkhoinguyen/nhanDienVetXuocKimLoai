import cv2
import numpy as np
import imgaug.augmenters as iaa
import os
from glob import glob
import matplotlib.pyplot as plt
import random

# Thư mục chứa ảnh đầu vào và ảnh đầu ra
input_folder = ""
output_folder = ""

# Tạo các augmenter cần thiết
augmenters = {
    "rotated": iaa.Affine(rotate=(-30, 30)),
    "bright_contrast": iaa.LinearContrast((0.6, 1.4)),
    "zoomed": iaa.CropAndPad(percent=(-0.2, 0.2)),
    "noisy": iaa.AdditiveGaussianNoise(scale=(10, 30)),
}

# Hàm xử lý ảnh
def process_image(image_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Lỗi đọc ảnh: {image_path}")
        return

    # Tạo thư mục cho từng ảnh
    save_dir = os.path.join(output_folder, image_name)
    os.makedirs(save_dir, exist_ok=True)

    # Lưu ảnh gốc RGB
    cv2.imencode('.jpg', image)[1].tofile(os.path.join(save_dir, "rgb.jpg"))

    # 1. Các augment từ imgaug
    for aug_name, augmenter in augmenters.items():
        aug_image = augmenter(image=image)
        cv2.imwrite(os.path.join(save_dir, f"{aug_name}.jpg"), aug_image)

    # 2. Chuyển sang xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(save_dir, "gray.jpg"), gray)

    # 3. Trích xuất biên
    edges = cv2.Canny(image, 100, 2
                      ;;00)
    cv2.imwrite(os.path.join(save_dir, "edges.jpg"), edges)

    # 4. Tách nền đơn giản với GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, image.shape[1]-20, image.shape[0]-20)
    try:
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        segmented = image * mask2[:, :, np.newaxis]
        cv2.imwrite(os.path.join(save_dir, "segmented.jpg"), segmented)
    except Exception as e:
        print(f"Tách nền lỗi với ảnh: {image_path} ({e})")

def show_sample_result():
    # Lấy danh sách thư mục ảnh đã xử lý
    if not os.path.exists(output_folder):
        print(f"Thư mục {output_folder} không tồn tại!")
        return
    subfolders = [f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, f))]
    if not subfolders:
        print(f"Không tìm thấy ảnh đã xử lý trong {output_folder}")
        return
    # Chọn ngẫu nhiên 1 thư mục ảnh
    sample_folder = random.choice(subfolders)
    sample_path = os.path.join(output_folder, sample_folder)
    # Định nghĩa các file cần hiển thị
    file_list = [
        ("Gốc (RGB)", "rgb.jpg"),
        ("Xoay", "rotated.jpg"),
        ("Tương phản", "bright_contrast.jpg"),
        ("Zoom", "zoomed.jpg"),
        ("Nhiễu", "noisy.jpg"),
        ("Xám", "gray.jpg"),
        ("Biên", "edges.jpg"),
        ("Tách nền", "segmented.jpg"),
    ]
    plt.figure(figsize=(16, 8))
    for idx, (title, fname) in enumerate(file_list):
        fpath = os.path.join(sample_path, fname)
        if not os.path.exists(fpath):
            continue
        img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # Nếu ảnh xám hoặc biên, chuyển sang RGB để hiển thị đúng màu
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 4, idx+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.suptitle(f"Kết quả xử lý cho ảnh: {sample_folder}")
    plt.tight_layout()
    plt.show()

# Quét tất cả ảnh trong thư mục
image_paths = glob(os.path.join(input_folder, "*.*"))
image_paths = [p for p in image_paths if p.lower().endswith((".jpg", ".png", ".jpeg"))]

print(f"🔍 Phát hiện {len(image_paths)} ảnh. Bắt đầu xử lý...")

for path in image_paths:
    process_image(path)

print("✅ Đã hoàn tất tăng cường dữ liệu và lưu vào thư mục 'augmented/'")
show_sample_result()
