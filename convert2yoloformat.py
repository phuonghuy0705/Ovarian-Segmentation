import os
import cv2
import numpy as np
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor

def process_mask_to_yolov8_segmentation(mask_path, color_to_class):
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"Failed to load image: {mask_path}")
        return []

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_height, mask_width = mask.shape[:2]
    polygons_info = []

    for color, class_id in color_to_class.items():
        color_mask = cv2.inRange(mask_rgb, np.array(color), np.array(color))
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            contour = contour.squeeze(1)
            polygon = [class_id]
            for (px, py) in contour:
                polygon.append(px / mask_width)
                polygon.append(py / mask_height)
            polygons_info.append(polygon)

    return polygons_info

def write_yolov8_segmentation(output_path, image_name, polygons_info):
    annotation_file_path = os.path.join(output_path, image_name)
    with open(annotation_file_path, "w") as f:
        for polygon in polygons_info:
            line = " ".join(map(str, polygon)) + "\n"
            f.write(line)

def process_one_file(file_name, input_mask_path, input_image_path, output_label_path, output_image_path, color_to_class):
    print(f"Processing mask: {file_name}")
    mask_path = os.path.join(input_mask_path, file_name)
    polygons_info = process_mask_to_yolov8_segmentation(mask_path, color_to_class)

    txt_name = file_name.rsplit('.', 1)[0] + ".txt"
    write_yolov8_segmentation(output_label_path, txt_name, polygons_info)

    base_name = file_name.rsplit('.', 1)[0]
    found_image = None
    for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
        candidate = base_name + ext
        candidate_path = os.path.join(input_image_path, candidate)
        if os.path.exists(candidate_path):
            found_image = candidate_path
            break

    if found_image:
        dst_image_path = os.path.join(output_image_path, os.path.basename(found_image))
        shutil.copy2(found_image, dst_image_path)
    else:
        print(f"Không tìm thấy file ảnh tương ứng cho mask {file_name}")

def main():
    parser = argparse.ArgumentParser(description="Convert masks to YOLOv8 segmentation format.")
    parser.add_argument('--input_dir', type=str, default='OTU2D_8_layers_merged_preprocessed_augmented',
                        help='Directory containing input masks and images (default: %(default)s)')
    parser.add_argument('--output_dir', type=str, default='OTU2D_8_layers_merged_preprocessed_augmented_yolo',
                        help='Directory to save YOLOv8 formatted labels and images (default: %(default)s)')
    args = parser.parse_args()

    color_to_class = {
        (255, 255, 255): 0,
    }

    for task in ['train', 'validation', 'test']:
        input_mask_path = os.path.join(args.input_dir, task, 'labels')
        input_image_path = os.path.join(args.input_dir, task, 'images')
        output_label_path = os.path.join(args.output_dir, task, 'labels')
        output_image_path = os.path.join(args.output_dir, task, 'images')

        os.makedirs(output_label_path, exist_ok=True)
        os.makedirs(output_image_path, exist_ok=True)

        if not os.path.exists(input_mask_path):
            print(f"Directory {input_mask_path} does not exist.")
            continue
        if not os.path.exists(input_image_path):
            print(f"Directory {input_image_path} does not exist.")
            continue

        image_files = [f for f in os.listdir(input_mask_path) if f.endswith(('.png', '.PNG'))]

        with ThreadPoolExecutor(max_workers=8) as executor:
            for file_name in image_files:
                executor.submit(process_one_file, file_name, input_mask_path, input_image_path,
                                output_label_path, output_image_path, color_to_class)

        print(f"Processing completed for {task}.")

if __name__ == "__main__":
    main()
