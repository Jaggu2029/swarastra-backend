"""
Renames every image inside each subfolder of `dataset/` so its filename
matches the folder name, e.g.:

    dataset/ma1/ba_000.jpg  ->  dataset/ma1/ma1_000.jpg
    dataset/ma1/ba_001.jpg  ->  dataset/ma1/ma1_001.jpg

Run this AFTER you've finished renaming the folders themselves to the
correct labels.

Usage:
    python rename_to_folder.py --dataset dataset
"""

import argparse
import os


def rename_folder_images(folder_path, label):
    files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    # Two-pass rename avoids collisions when old/new names could overlap
    temp_names = []
    for i, fname in enumerate(files):
        ext = os.path.splitext(fname)[1]
        old_path = os.path.join(folder_path, fname)
        temp_path = os.path.join(folder_path, f"__tmp_{i:03d}{ext}")
        os.rename(old_path, temp_path)
        temp_names.append(temp_path)

    for i, temp_path in enumerate(temp_names):
        ext = os.path.splitext(temp_path)[1]
        new_name = f"{label}_{i:03d}{ext}"
        new_path = os.path.join(folder_path, new_name)
        os.rename(temp_path, new_path)

    print(f"{label:>10}: renamed {len(files)} files")
    return len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset", help="Path to the dataset root folder")
    args = ap.parse_args()

    if not os.path.isdir(args.dataset):
        raise FileNotFoundError(f"Dataset folder not found: {args.dataset}")

    subfolders = sorted(
        d for d in os.listdir(args.dataset)
        if os.path.isdir(os.path.join(args.dataset, d))
    )

    total = 0
    for label in subfolders:
        folder_path = os.path.join(args.dataset, label)
        total += rename_folder_images(folder_path, label)

    print(f"\nDone. Renamed {total} files across {len(subfolders)} folders.")


if __name__ == "__main__":
    main()
