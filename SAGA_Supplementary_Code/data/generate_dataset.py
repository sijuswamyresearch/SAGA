"""
SAGA Medical Image Degradation Pipeline
---------------------------------------
This unified script generates the synthetic dataset pairs (sharp and degraded) 
for both the Chest CT and DXA Osteoporosis datasets. 

Usage:
    python generate_dataset.py --dataset CT
    python generate_dataset.py --dataset Osteoporosis
"""

import os
import io
import math
import random
import zipfile
import warnings
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Global Configuration & Hyperparameters ---
TASKS = {"data_dblur": "deblur", "data_deblock": "deblock", "data_sr": "sr"}
TARGET_SPLIT_PAIRS = {'train': 4000, 'val': 500, 'test': 500}
USE_PATCHES = True
PATCH_SIZE = (256, 256)

# Degradation Parameters
BLUR_TYPES = ['gaussian', 'motion', 'defocus']
GAUSSIAN_SIGMA_RANGE = (1.0, 5.0)
GAUSSIAN_KERNEL_SIZES = [5, 7, 9, 11]
MOTION_KERNEL_SIZES = [9, 15, 21]
DEFOCUS_RADIUS_RANGE = (2, 5)
JPEG_QUALITY_RANGE = (10, 40)
SR_DOWN_FACTORS = [2, 3, 4]
SR_PRE_BLUR_SIGMA = 1.0


# --- Degradation Functions ---

def apply_gaussian_blur(img_tensor, kernel_size=5, sigma=3.0):
    kernel_size = int(kernel_size) // 2 * 2 + 1
    sigma = max(0.1, float(sigma))
    try:
        blur_transform = transforms.GaussianBlur(kernel_size, sigma=sigma)
    except TypeError:
        blur_transform = transforms.GaussianBlur(kernel_size, sigma=(sigma, sigma))
    return blur_transform(img_tensor.float())

def apply_motion_blur(img_tensor, kernel_size=15):
    img_tensor = img_tensor.float()
    if img_tensor.dim() == 3: img_tensor = img_tensor.unsqueeze(0)
    channels = img_tensor.shape[1]
    kernel_size = int(kernel_size) // 2 * 2 + 1
    
    kernel = torch.zeros((kernel_size, kernel_size), device=img_tensor.device, dtype=torch.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    
    padding = kernel_size // 2
    blurred = F.conv2d(img_tensor, kernel, padding=padding, groups=channels)
    if blurred.shape[0] == 1: blurred = blurred.squeeze(0)
    return blurred

def apply_defocus_blur(img_tensor, radius=3):
    img_tensor = img_tensor.float()
    if img_tensor.dim() == 3: img_tensor = img_tensor.unsqueeze(0)
    channels = img_tensor.shape[1]
    kernel_size = int(radius) * 2 + 1
    
    center = radius
    Y, X = np.ogrid[:kernel_size, :kernel_size]
    dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
    kernel_np = (dist_from_center <= radius).astype(np.float32)
    kernel_np /= kernel_np.sum()
    
    kernel = torch.from_numpy(kernel_np).to(device=img_tensor.device, dtype=torch.float32)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    
    padding = kernel_size // 2
    blurred = F.conv2d(img_tensor, kernel, padding=padding, groups=channels)
    if blurred.shape[0] == 1: blurred = blurred.squeeze(0)
    return blurred

def apply_jpeg_blocking(sharp_pil_image_bw, quality_range=(10, 40)):
    quality = random.randint(quality_range[0], quality_range[1])
    buffer = io.BytesIO()
    sharp_pil_image_bw.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('L')

def apply_sr_degradation(sharp_tensor_01, down_factor, pre_blur_sigma=1.0, target_size=None):
    if pre_blur_sigma > 0:
        kernel_size = int(math.ceil(pre_blur_sigma * 3)) * 2 + 1
        sharp_tensor_01 = apply_gaussian_blur(sharp_tensor_01, kernel_size=kernel_size, sigma=pre_blur_sigma)
    h, w = sharp_tensor_01.shape[-2:]
    lr_h, lr_w = max(1, h // down_factor), max(1, w // down_factor)
    
    try:
        lr_tensor = transforms.functional.resize(sharp_tensor_01, (lr_h, lr_w), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    except TypeError:
        lr_tensor = transforms.functional.resize(sharp_tensor_01, (lr_h, lr_w), interpolation=transforms.InterpolationMode.BICUBIC)
        
    if target_size:
        try:
            return transforms.functional.resize(lr_tensor, target_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        except TypeError:
            return transforms.functional.resize(lr_tensor, target_size, interpolation=transforms.InterpolationMode.BICUBIC)
    return lr_tensor


# --- Main Generation Pipeline ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # 1. Argument Parsing inside the execution block
    parser = argparse.ArgumentParser(description="Generate Degraded Medical Image Datasets.")
    parser.add_argument('--dataset', type=str, required=True, choices=['CT', 'Osteoporosis'], 
                        help="Specify which dataset to generate: 'CT' or 'Osteoporosis'")
    args = parser.parse_args()

    # 2. Dynamic Configuration
    if args.dataset == 'CT':
        SOURCE_ZIP_FILE_PATH = "CT_final.zip"
        NUM_SOURCE_IMAGES_TO_LOAD = 1000
        OUTPUT_ROOT_DIR = "CT_dataset"
    else:
        SOURCE_ZIP_FILE_PATH = "Osteoporosis_final.zip"
        NUM_SOURCE_IMAGES_TO_LOAD = 212
        OUTPUT_ROOT_DIR = "Osteoporosis_dataset"

    print(f"[INFO] Initializing dataset pipeline for: {args.dataset}")
    print(f"[INFO] Loading images from: {SOURCE_ZIP_FILE_PATH}")
    
    # 3. Directory Creation
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    for task_folder in TASKS:
        task_path = os.path.join(OUTPUT_ROOT_DIR, task_folder)
        for split in TARGET_SPLIT_PAIRS.keys():
            os.makedirs(os.path.join(task_path, split, 'sharp'), exist_ok=True)
            pair_type = 'blur' if TASKS[task_folder] == 'deblur' else ('blocked' if TASKS[task_folder] == 'deblock' else 'lr')
            os.makedirs(os.path.join(task_path, split, pair_type), exist_ok=True)

    # 4. Zip Extraction
    source_images_pil, source_filenames = [], []
    processed_files = set()

    try:
        with zipfile.ZipFile(SOURCE_ZIP_FILE_PATH, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

            for filename in tqdm(file_list, desc="Extracting Scans"):
                if not filename.startswith('__MACOSX') and filename.lower().endswith(image_extensions) and not filename.endswith('/'):
                    base_filename = os.path.basename(filename)
                    if base_filename in processed_files: continue
                    try:
                        image_data = zip_ref.read(filename)
                        img_bw = Image.open(io.BytesIO(image_data)).convert('L')
                        source_images_pil.append(img_bw)
                        source_filenames.append(base_filename)
                        processed_files.add(base_filename)
                    except Exception:
                        continue
    except Exception as e:
        print(f"[ERROR] Failed to load zip file. Ensure {SOURCE_ZIP_FILE_PATH} is in the same directory. Error: {e}")
        exit(1)

    num_total_source = len(source_images_pil)
    print(f"[INFO] Loaded {num_total_source} valid clinical scans.")

    # 5. Stratified Splitting
    selected_indices = list(range(min(NUM_SOURCE_IMAGES_TO_LOAD, num_total_source)))
    random.seed(42)
    random.shuffle(selected_indices)

    test_split_size = max(1, int(0.20 * len(selected_indices)))
    val_split_size = max(1, int(0.20 * len(selected_indices)))
    
    train_val_indices, test_indices = train_test_split(selected_indices, test_size=test_split_size, random_state=42)
    actual_val_split_size = min(val_split_size, len(train_val_indices))
    train_indices, val_indices = train_test_split(train_val_indices, test_size=actual_val_split_size, random_state=43) if train_val_indices else ([], [])

    split_indices = {'train': list(train_indices), 'val': list(val_indices), 'test': list(test_indices)}
    
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    to_tensor_transform = transforms.ToTensor()

    global_pair_counter = 0

    # 6. Degradation Loop
    for task_folder, task_name in TASKS.items():
        print(f"\n[INFO] Processing Task: {task_name.upper()}")
        task_path = os.path.join(OUTPUT_ROOT_DIR, task_folder)
        degraded_folder = 'blur' if task_name == 'deblur' else ('blocked' if task_name == 'deblock' else 'lr')

        for split, indices in split_indices.items():
            if not indices: continue
            
            sharp_dir = os.path.join(task_path, split, 'sharp')
            degraded_dir = os.path.join(task_path, split, degraded_folder)
            target_split_pairs = TARGET_SPLIT_PAIRS[split]
            split_pair_count = 0
            source_img_idx_cycle = 0 

            print(f"  -> Generating {split.upper()} split ({target_split_pairs} pairs)...")

            while split_pair_count < target_split_pairs:
                source_idx = indices[source_img_idx_cycle % len(indices)]
                source_img_idx_cycle += 1

                try:
                    source_img_pil = source_images_pil[source_idx]
                    original_filename_base = os.path.splitext(source_filenames[source_idx])[0]
                    
                    img_w, img_h = source_img_pil.size
                    target_h, target_w = PATCH_SIZE

                    if USE_PATCHES and img_h >= target_h and img_w >= target_w:
                        i, j, h, w = transforms.RandomCrop.get_params(source_img_pil, output_size=PATCH_SIZE)
                        sharp_pil_processed = transforms.functional.crop(source_img_pil, i, j, h, w)
                    else:
                        sharp_pil_processed = transforms.Resize(PATCH_SIZE, interpolation=transforms.InterpolationMode.BICUBIC)(source_img_pil)

                    sharp_pil_aug = augment_transform(sharp_pil_processed)
                    sharp_tensor_01 = to_tensor_transform(sharp_pil_aug)

                    # Apply Task-Specific Degradation
                    if task_name == 'deblur':
                        blur_type = random.choice(BLUR_TYPES)
                        if blur_type == 'gaussian':
                            degraded_tensor_01 = apply_gaussian_blur(sharp_tensor_01, kernel_size=random.choice(GAUSSIAN_KERNEL_SIZES), sigma=random.uniform(*GAUSSIAN_SIGMA_RANGE))
                        elif blur_type == 'motion':
                            degraded_tensor_01 = apply_motion_blur(sharp_tensor_01, kernel_size=random.choice(MOTION_KERNEL_SIZES))
                        else:
                            degraded_tensor_01 = apply_defocus_blur(sharp_tensor_01, radius=random.randint(*DEFOCUS_RADIUS_RANGE))
                    elif task_name == 'deblock':
                        blocked_pil = apply_jpeg_blocking(transforms.ToPILImage()(sharp_tensor_01).convert('L'), JPEG_QUALITY_RANGE)
                        degraded_tensor_01 = to_tensor_transform(blocked_pil)
                    elif task_name == 'sr':
                        degraded_tensor_01 = apply_sr_degradation(sharp_tensor_01, random.choice(SR_DOWN_FACTORS), SR_PRE_BLUR_SIGMA, target_size=PATCH_SIZE)

                    if degraded_tensor_01 is None or torch.isnan(degraded_tensor_01).any(): continue

                    # Save tensors (Unique numbering via split_pair_count ensures no overwrites)
                    pair_filename = f"{original_filename_base}_{split_pair_count:04d}.pt"
                    torch.save(sharp_tensor_01, os.path.join(sharp_dir, pair_filename))
                    torch.save(degraded_tensor_01, os.path.join(degraded_dir, pair_filename))
                    
                    split_pair_count += 1
                    global_pair_counter += 1

                except Exception:
                    continue

    print(f"\n[SUCCESS] {args.dataset} Pipeline complete. Total generated pairs: {global_pair_counter}")