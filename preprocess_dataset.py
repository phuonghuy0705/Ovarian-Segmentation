import cv2
import numpy as np
import os
import concurrent.futures
import argparse

def preprocess_image_and_mask(image_file, image_folder, label_folder, output_image_folder, output_label_folder, target_size=(640, 640)):
    try:
        image_path = os.path.join(image_folder, image_file)
        mask_file = os.path.splitext(image_file)[0] + '.png'  
        mask_path = os.path.join(label_folder, mask_file)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Không đọc được ảnh: {image_path}")
            return
        if mask is None:
            print(f"Không đọc được mask: {mask_path}")
            return

        # Denoising 
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # CLAHE 
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        def resize_and_pad(img, target_size):
            old_size = img.shape[:2]
            ratio = float(target_size[0]) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            img_resized = cv2.resize(img, (new_size[1], new_size[0]))
            delta_w = target_size[1] - new_size[1]
            delta_h = target_size[0] - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            if len(img.shape) == 3:
                color = [0, 0, 0]
            else:
                color = 0
            img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            return img_padded

        image_padded = resize_and_pad(image, target_size)
        mask_padded = resize_and_pad(mask, target_size)

        os.makedirs(output_image_folder, exist_ok=True)
        os.makedirs(output_label_folder, exist_ok=True)
        cv2.imwrite(os.path.join(output_image_folder, image_file), image_padded)
        cv2.imwrite(os.path.join(output_label_folder, mask_file), mask_padded)

    except Exception as e:
        print(f"Lỗi xử lý {image_file}: {e}")

def process_split(task, target_root):
    image_folder = os.path.join(target_root, task, 'image')
    label_folder = os.path.join(target_root, task, 'label')
    output_image_folder = os.path.join(target_root + '_preprocessed', task, 'images')
    output_label_folder = os.path.join(target_root + '_preprocessed', task, 'labels')

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg"))]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for image_file in image_files:
            futures.append(executor.submit(
                preprocess_image_and_mask,
                image_file, image_folder, label_folder,
                output_image_folder, output_label_folder
            ))
        concurrent.futures.wait(futures)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tiền xử lý ảnh và mask.")
    parser.add_argument("source_root", nargs='?', default="OTU2D_8_layers_merged", 
                        help="Đường dẫn thư mục dữ liệu nguồn (mặc định: OTU2D_8_layers_merged)")
    parser.add_argument("target_root", nargs='?', default="OTU2D_8_layers_merged_preprocessed", 
                        help="Đường dẫn thư mục lưu kết quả (mặc định: OTU2D_8_layers_merged_preprocessed)")
    args = parser.parse_args()

    for task in ['train', 'validation', 'test']:
        process_split(task, args.source_root)

    print("Hoàn tất tiền xử lý ảnh và mask màu.")
