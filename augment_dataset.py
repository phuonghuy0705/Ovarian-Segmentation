import os
import numpy as np
import cv2
import random
import albumentations as A
from usaugment.albumentations import DepthAttenuation, GaussianShadow, HazeArtifact, SpeckleReduction
import shutil
from concurrent.futures import ThreadPoolExecutor
import argparse

# Geometry-based transforms
geometry_augs = {
    "crop": A.RandomCrop(height=480, width=480, p=1.0),
    "hflip": A.HorizontalFlip(p=1.0),
    "vflip": A.VerticalFlip(p=1.0),
    "rotate": A.Rotate(limit=30, p=1.0),
    "translate_x": A.Affine(translate_percent={"x": 0.2, "y": 0}, p=1.0),
    "translate_y": A.Affine(translate_percent={"x": 0, "y": 0.2}, p=1.0),
    "shear_x": A.Affine(shear={"x": (-20, 20), "y": (0, 0)}, p=1.0),
    "shear_y": A.Affine(shear={"x": (0, 0), "y": (-20, 20)}, p=1.0),
    "elastic": A.ElasticTransform(p=1.0),
    "grid_dist": A.GridDistortion(p=1.0),
    "scale": A.Affine(translate_percent={"x": 0.1, "y": 0.1}, scale=(0.9, 1.1), rotate=(-15, 15), p=1.0),
}

# Pixel-based transforms
pixel_augs = {
    "brightness": A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0, p=1.0),
    "contrast": A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.3, p=1.0),
    "saturation": A.HueSaturationValue(sat_shift_limit=20, p=1.0),
    "gauss_blur": A.GaussianBlur(p=1.0),
    "equalize": A.Equalize(p=1.0),
    "median_blur": A.MedianBlur(blur_limit=3, p=1.0),
    "gauss_noise": A.GaussNoise(p=1.0),
    "depth_attenuation":  DepthAttenuation(p=1.0, attenuation_rate=1.0),
    "gaussian_shadow": GaussianShadow(p=1.0, strength=0.5, sigma_x=0.2, sigma_y=0.1),
    "haze_artifact": HazeArtifact(p=1.0, radius=0.5, sigma=0.05),
    "speckle_reduction": SpeckleReduction(p=1.0),
}

def save_aug(image, mask, base_name, aug_name, output_img_dir, output_mask_dir):
    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    mask = np.nan_to_num(mask, nan=0.0, posinf=255.0, neginf=0.0)

    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    if mask.dtype != np.uint8:
        mask = (mask * 255).clip(0, 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)

    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_{aug_name}.png"), image)
    cv2.imwrite(os.path.join(output_mask_dir, f"{base_name}_{aug_name}.png"), mask)

def apply_single_augment(image, mask, aug_name, transform, is_geom):
    if is_geom:
        transformed = A.Compose([transform], additional_targets={"mask": "mask"})(image=image, mask=mask)
        return transformed["image"], transformed["mask"]
    else:
        if aug_name in ["depth_attenuation", "gaussian_shadow", "haze_artifact", "speckle_reduction"]:
            transformed = A.Compose([transform])(image=image, scan_mask=mask)
        else:
            transformed = A.Compose([transform])(image=image)
        return transformed["image"], mask

def augment_image(image_path, mask_path, output_img_dir, output_mask_dir, num_geometry, num_pixel):
    image = cv2.imread(image_path).astype(np.float32) / 255.0
    mask_color = cv2.imread(mask_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

    mask_gray = cv2.cvtColor(mask_color, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask_gray, 0.1, 1.0, cv2.THRESH_BINARY)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_aug(image, mask_bin, base_name, "orig", output_img_dir, output_mask_dir)

    selected_geom = random.sample(list(geometry_augs.items()), num_geometry)
    selected_pixel = random.sample(list(pixel_augs.items()), num_pixel)
    selected_augs = selected_geom + selected_pixel

    for aug_name, transform in selected_augs:
        is_geom = aug_name in geometry_augs
        img_aug, mask_aug = apply_single_augment(image, mask_bin, aug_name, transform, is_geom)
        save_aug(img_aug, mask_aug, base_name, aug_name, output_img_dir, output_mask_dir)

def process_dataset_split(split, input_dir, output_dir, num_geometry, num_pixel):
    input_img_dir = os.path.join(input_dir, split, "images")
    input_mask_dir = os.path.join(input_dir, split, "labels")
    output_img_dir = os.path.join(output_dir, split, "images")
    output_mask_dir = os.path.join(output_dir, split, "labels")

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    file_list = os.listdir(input_img_dir)

    if split == "train":
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for filename in file_list:
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                image_path = os.path.join(input_img_dir, filename)
                mask_path = os.path.join(input_mask_dir, os.path.splitext(filename)[0] + ".PNG")
                if os.path.exists(mask_path):
                    futures.append(executor.submit(
                        augment_image, image_path, mask_path, output_img_dir, output_mask_dir, num_geometry, num_pixel
                    ))
            for f in futures:
                f.result()
    else:
        for filename in file_list:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            image_path = os.path.join(input_img_dir, filename)
            mask_path = os.path.join(input_mask_dir, os.path.splitext(filename)[0] + ".PNG")
            if os.path.exists(mask_path):
                shutil.copy2(image_path, os.path.join(output_img_dir, filename))
                shutil.copy2(mask_path, os.path.join(output_mask_dir, os.path.splitext(filename)[0] + ".png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply geometry and pixel augmentation to a dataset.")
    parser.add_argument("--input_dir", type=str, default="OTU2D_8_layers_merged_preprocessed", help="Input dataset directory")
    parser.add_argument("--output_dir", type=str, default="OTU2D_8_layers_merged_preprocessed_augmented", help="Output dataset directory")
    parser.add_argument("--num_geometry", type=int, default=3, help="Number of geometry-based augmentations")
    parser.add_argument("--num_pixel", type=int, default=3, help="Number of pixel-based augmentations")
    args = parser.parse_args()

    for split in ['train', 'validation', 'test']:
        process_dataset_split(split, args.input_dir, args.output_dir, args.num_geometry, args.num_pixel)
