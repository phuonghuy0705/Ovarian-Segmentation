import cv2
import os
import numpy as np
import argparse

def visualize_and_save(image_path, label_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không đọc được ảnh: {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if not os.path.exists(label_path):
        print(f"Không tìm thấy label: {label_path}")
        return

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue  

        class_id = int(parts[0])
        polygon_coords = list(map(float, parts[5:]))

        polygon_points = []
        for i in range(0, len(polygon_coords), 2):
            px = int(polygon_coords[i] * w)
            py = int(polygon_coords[i + 1] * h)
            polygon_points.append((px, py))

        polygon_points = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=2)

        if len(polygon_points) > 0:
            x_text, y_text = polygon_points[0][0][0], polygon_points[0][0][1]
            cv2.putText(img, str(class_id), (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Lưu ảnh đã vẽ
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_bgr)
    print(f"Đã lưu: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLOv8 segmentation annotations and save them.")
    parser.add_argument('--input_dir', type=str, default="OTU2D_8_layers_merged_preprocessed_augmented_yolo_new",
                        help='Path to input directory containing train/validation/test folders with images and labels')
    parser.add_argument('--output_dir', type=str, default="visual_OTU2D_8_layers_merged_preprocessed_augmented_yolo_new",
                        help='Path to output directory for saving visualized images')

    args = parser.parse_args()

    for split in ["train", "validation", "test"]:
        img_dir = os.path.join(args.input_dir, split, "image")
        label_dir = os.path.join(args.input_dir, split, "label_txt")
        save_dir = os.path.join(args.output_dir, split)

        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            print(f"Bỏ qua '{split}' vì thiếu thư mục ảnh hoặc nhãn.")
            continue

        for file_name in os.listdir(img_dir):
            if file_name.endswith((".jpg", ".png", ".jpeg", ".JPG", ".PNG")):
                image_path = os.path.join(img_dir, file_name)
                base_name = os.path.splitext(file_name)[0]
                label_path = os.path.join(label_dir, f"{base_name}.txt")
                save_path = os.path.join(save_dir, file_name)
                visualize_and_save(image_path, label_path, save_path)

if __name__ == "__main__":
    main()
