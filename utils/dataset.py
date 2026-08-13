"""
dataset.py
----------
This file handles loading chest X-ray images from disk and preparing them
for the neural network.

WHY THIS FILE EXISTS:
Neural networks don't understand "image files". We need to:
  1. Read image files from folders
  2. Resize them all to the same size (networks need fixed input size)
  3. Convert pixel values into a normalized numeric format
  4. Optionally apply "augmentation" (small random changes like flips/rotations)
     so the model sees more variety and doesn't just memorize the training set

We use PyTorch's `ImageFolder` dataset format, which expects your data to be
organized like this:

    dataset/
        train/
            Normal/
                img1.png
                img2.png
            COVID/
                img1.png
            Lung_Opacity/
                ...
            Viral_Pneumonia/
                ...
        val/
            Normal/
            COVID/
            ...
        test/
            Normal/
            COVID/
            ...

This is exactly how the COVID-19 Radiography Database is structured (after
you split it into train/val/test, which we do below).
"""

import os
import shutil
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------------------------------------------
# STEP 1: Define image transformations
# -----------------------------------------------------------------
# Pretrained models (like ResNet) expect:
#   - 224x224 pixel images
#   - 3 color channels (RGB) even though X-rays are grayscale
#   - Pixel values normalized using ImageNet's mean/std
#     (because our model was originally trained on ImageNet, and expects
#      input data distributed the same way)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),   # small chance to flip left-right
    transforms.RandomRotation(10),             # small random rotation (+-10 degrees)
    transforms.ToTensor(),                     # convert image to a PyTorch tensor
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# For validation/test, we do NOT augment — we want consistent, real evaluation
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# -----------------------------------------------------------------
# STEP 2: Split raw dataset into train/val/test folders
# -----------------------------------------------------------------
def split_dataset(raw_dir, output_dir, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    The COVID-19 Radiography Database (as downloaded from Kaggle) comes as
    one folder per class, with all images together (no train/val/test split).

    This function randomly splits each class into train/val/test folders,
    keeping the same class balance in each split.

    raw_dir: path to the original downloaded dataset
              e.g. raw_dir/COVID/*.png, raw_dir/Normal/*.png, etc.
    output_dir: where to create the new train/val/test structure
    """
    random.seed(seed)
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    print(f"Found classes: {classes}")

    for cls in classes:
        cls_path = os.path.join(raw_dir, cls)
        images = os.listdir(cls_path)
        random.shuffle(images)

        n_total = len(images)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_val - n_test

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split_name, split_images in splits.items():
            split_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img_name in split_images:
                src = os.path.join(cls_path, img_name)
                dst = os.path.join(split_dir, img_name)
                if not os.path.exists(dst):
                    shutil.copyfile(src, dst)

        print(f"{cls}: {n_train} train / {n_val} val / {n_test} test")

    print("Dataset split complete.")


# -----------------------------------------------------------------
# STEP 3: Create PyTorch DataLoaders
# -----------------------------------------------------------------
def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    """
    Returns train_loader, val_loader, test_loader, and class_names.

    A DataLoader automatically:
      - grabs batches of images (e.g. 32 at a time)
      - shuffles training data each epoch
      - applies the transforms defined above
      - loads data in parallel using multiple workers (faster)
    """
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes  # e.g. ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']
    print(f"Classes (in index order): {class_names}")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names
