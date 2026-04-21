"""
split_data.py
Splits a flat image+label directory into train / val / test subsets
in the folder structure expected by YOLOv8.

Output structure:
    dataset/
        train/images/  train/labels/
        val/images/    val/labels/
        test/images/   test/labels/

Usage:
    python src/split_data.py
"""

import os
import shutil
import random


def split_data(
    image_dir: str,
    label_dir: str,
    output_base: str,
    split_ratios: tuple = (0.7, 0.2, 0.1),
    seed: int = 42,
) -> None:
    """
    Split images and matching labels into train/val/test folders.

    Args:
        image_dir:    Directory containing source images (.jpg/.jpeg/.png)
        label_dir:    Directory containing matching .txt label files
        output_base:  Root output directory (will be created if absent)
        split_ratios: (train, val, test) fractions — must sum to 1.0
        seed:         Random seed for reproducibility
    """
    # Validate source directories
    for path, name in [(image_dir, "image_dir"), (label_dir, "label_dir")]:
        if not os.path.exists(path):
            print(f"Error: {name} does not exist: {path}")
            return

    random.seed(seed)

    # Collect all image files
    exts = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]
    random.shuffle(files)

    n = len(files)
    train_end = int(n * split_ratios[0])
    val_end = train_end + int(n * split_ratios[1])

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:],
    }

    print(f"Total images: {n}")
    for name, subset in splits.items():
        print(f"  {name}: {len(subset)}")

    missing_labels = []

    for split_name, file_list in splits.items():
        img_out = os.path.join(output_base, split_name, "images")
        lbl_out = os.path.join(output_base, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for filename in file_list:
            # Copy image
            shutil.copy2(os.path.join(image_dir, filename), os.path.join(img_out, filename))

            # Copy matching label
            base = os.path.splitext(filename)[0]
            src_label = os.path.join(label_dir, base + ".txt")
            if os.path.exists(src_label):
                shutil.copy2(src_label, os.path.join(lbl_out, base + ".txt"))
            else:
                missing_labels.append(filename)
                print(f"Warning: label missing for {filename}")

    if missing_labels:
        print(f"\n{len(missing_labels)} images had no matching label file.")
    else:
        print("\nDataset split complete — no missing labels.")


if __name__ == "__main__":
    split_data(
        image_dir="data/train/images",
        label_dir="data/labels",
        output_base="dataset",
    )