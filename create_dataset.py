import os
import shutil
import random
import cv2
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor

# --- PHẦN 1: GỘP DỮ LIỆU VÀ CHIA TRAIN/VAL/TEST ---
def merge_and_split_data(source_root, target_root):
    for split in ['train', 'validation', 'test']:
        for folder in ['image', 'label']:
            os.makedirs(os.path.join(target_root, split, folder), exist_ok=True)

    for tumor_type_folder in os.listdir(source_root):
        tumor_path = os.path.join(source_root, tumor_type_folder)
        if not os.path.isdir(tumor_path):
            continue  

        train_image_path = os.path.join(tumor_path, 'train', 'image')
        train_label_path = os.path.join(tumor_path, 'train', 'label')

        if os.path.exists(train_image_path):
            for filename in os.listdir(train_image_path):
                shutil.copy2(os.path.join(train_image_path, filename), os.path.join(target_root, 'train', 'image', filename))

        if os.path.exists(train_label_path):
            for filename in os.listdir(train_label_path):
                shutil.copy2(os.path.join(train_label_path, filename), os.path.join(target_root, 'train', 'label', filename))

        val_image_path = os.path.join(tumor_path, 'validation', 'image')
        val_label_path = os.path.join(tumor_path, 'validation', 'label')

        if os.path.exists(val_image_path) and os.path.exists(val_label_path):
            image_files = sorted(os.listdir(val_image_path))
            label_files = sorted(os.listdir(val_label_path))

            combined = list(zip(image_files, label_files))
            random.shuffle(combined)
            split_idx = len(combined) // 2
            val_split = combined[:split_idx]
            test_split = combined[split_idx:]

            for img_name, lbl_name in val_split:
                shutil.copy2(os.path.join(val_image_path, img_name), os.path.join(target_root, 'validation', 'image', img_name))
                shutil.copy2(os.path.join(val_label_path, lbl_name), os.path.join(target_root, 'validation', 'label', lbl_name))

            for img_name, lbl_name in test_split:
                shutil.copy2(os.path.join(val_image_path, img_name), os.path.join(target_root, 'test', 'image', img_name))
                shutil.copy2(os.path.join(val_label_path, lbl_name), os.path.join(target_root, 'test', 'label', lbl_name))


# --- PHẦN 2: CHUYỂN MASK MÀU SANG MASK NHỊ PHÂN ---

color_to_class = {
    (64, 0, 0): 0,
    (0, 0, 128): 1,
    (0, 128, 0): 2,
    (64, 64, 64): 3,
    (0, 64, 0): 4,
    (64, 64, 0): 5,
    (0, 0, 64): 6,
    (64, 0, 64): 7,
}

def color_mask_to_binary(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Không đọc được ảnh: {input_path}")
        return
    
    binary_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for color in color_to_class.keys():
        bgr_color = (color[2], color[1], color[0])
        mask = cv2.inRange(img, np.array(bgr_color), np.array(bgr_color))
        binary_mask = cv2.bitwise_or(binary_mask, mask)
    
    cv2.imwrite(output_path, binary_mask)
    print(f"Đã lưu ảnh mask nhị phân: {output_path}")

def process_mask_file(args):
    input_path, output_path = args
    color_mask_to_binary(input_path, output_path)

def process_mask_folder(input_mask_folder, output_mask_folder, max_workers=4):
    os.makedirs(output_mask_folder, exist_ok=True)

    mask_files = [
        (os.path.join(input_mask_folder, f), os.path.join(output_mask_folder, f))
        for f in os.listdir(input_mask_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_mask_file, mask_files)


def main():
    parser = argparse.ArgumentParser(description="Merge and process tumor mask dataset.")
    parser.add_argument("--source_root", type=str, default="OTU2D_8_layers_splitted", help="Đường dẫn thư mục dữ liệu nguồn (chứa các folder tumor).")
    parser.add_argument("--target_root", type=str, default="OTU2D_8_layers_merged", help="Đường dẫn thư mục đích để lưu dữ liệu gộp.")

    args = parser.parse_args()

    print(f"Bắt đầu gộp và xử lý từ {args.source_root} -> {args.target_root}")
    merge_and_split_data(args.source_root, args.target_root)

    splits = ['train', 'validation', 'test']
    for split in splits:
        mask_folder = os.path.join(args.target_root, split, 'label')
        process_mask_folder(mask_folder, mask_folder, max_workers=4)

    print("Hoàn tất: Đã gộp và xử lý nhị phân các mask.")

if __name__ == "__main__":
    main()
