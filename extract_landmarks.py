"""
Extracts MediaPipe hand landmarks from every image in dataset/<label>/*.jpg
and saves them to a single CSV for training. Frames where no hand is
detected are logged and skipped (this doubles as the "bad frame" cleanup
step of your EDA).

Uses the modern MediaPipe Tasks API (HandLandmarker), since the legacy
`mp.solutions.hands` API has been deprecated and removed in current
MediaPipe releases. This requires a one-time model download:

    python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task', 'hand_landmarker.task')"

Landmarks are normalized so the model learns hand SHAPE, not position or
size in the frame:
    1. Translate so the wrist (landmark 0) is the origin
    2. Scale so the average landmark distance from the wrist is 1.0

Usage:
    python extract_landmarks.py --dataset dataset --out landmarks.csv --model hand_landmarker.task
"""

import argparse
import csv
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


def normalize_landmarks(landmarks):
    """
    landmarks: list of 21 (x, y, z) tuples from MediaPipe (already normalized
    0-1 by image size). Made robust to camera differences via three steps:
      1. Translate so the wrist (landmark 0) is the origin
      2. Rotate so the wrist->middle-finger-MCP direction always points
         "up" (removes sensitivity to hand tilt / camera angle)
      3. Scale so the average landmark distance from the wrist is 1.0
         (removes sensitivity to hand size / distance from camera)
    """
    pts = np.array(landmarks, dtype=np.float32)  # shape (21, 3)
    wrist = pts[0].copy()
    pts -= wrist  # translate

    # rotate in the image plane so wrist->middle_mcp points straight up
    ref = pts[9][:2]  # middle finger MCP, relative to wrist
    theta = np.arctan2(ref[1], ref[0])
    delta = -np.pi / 2 - theta
    cos_d, sin_d = np.cos(delta), np.sin(delta)
    rot = np.array([[cos_d, -sin_d], [sin_d, cos_d]], dtype=np.float32)
    pts[:, :2] = (rot @ pts[:, :2].T).T

    scale = np.mean(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale  # scale-invariant

    return pts.flatten()  # 63 values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset", help="Path to dataset root (folders per label)")
    ap.add_argument("--out", default="landmarks.csv", help="Output CSV path")
    ap.add_argument("--model", default="hand_landmarker.task",
                     help="Path to the downloaded hand_landmarker.task model file")
    ap.add_argument("--min_detection_confidence", type=float, default=0.5)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model file not found: {args.model}\n"
            "Download it first with:\n"
            "  python -c \"import urllib.request; urllib.request.urlretrieve("
            "'https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task', "
            "'hand_landmarker.task')\""
        )

    base_options = mp_tasks.BaseOptions(model_asset_path=args.model)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=args.min_detection_confidence,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    labels = sorted(
        d for d in os.listdir(args.dataset)
        if os.path.isdir(os.path.join(args.dataset, d))
    )

    rows = []
    dropped = []
    per_label_kept = {}

    for label in labels:
        folder = os.path.join(args.dataset, label)
        files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        kept = 0
        for fname in files:
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                dropped.append((label, fname, "could not read image"))
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = detector.detect(mp_image)

            if not result.hand_landmarks:
                dropped.append((label, fname, "no hand detected"))
                continue

            hand = result.hand_landmarks[0]
            coords = [(lm.x, lm.y, lm.z) for lm in hand]
            feats = normalize_landmarks(coords)

            row = {"label": label, "file": fname}
            for i, v in enumerate(feats):
                row[f"f{i}"] = v
            rows.append(row)
            kept += 1

        per_label_kept[label] = kept
        print(f"{label:>10}: kept {kept}/{len(files)} frames")

    detector.close()

    if not rows:
        print("\nNo landmarks extracted at all — check your dataset path / images.")
        return

    fieldnames = ["label", "file"] + [f"f{i}" for i in range(63)]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} landmark rows to {args.out}")

    if dropped:
        print(f"\nDropped {len(dropped)} frames (no hand detected / unreadable):")
        for label, fname, reason in dropped[:30]:
            print(f"  {label}/{fname}: {reason}")
        if len(dropped) > 30:
            print(f"  ... and {len(dropped) - 30} more")

    low = {k: v for k, v in per_label_kept.items() if v < 10}
    if low:
        print(f"\n[!] Warning: these labels have fewer than 10 usable frames — "
              f"consider re-extracting more frames for them: {low}")


if __name__ == "__main__":
    main()
