"""
Recognize a Gujarati Sign Language letter using the trained model.

Lets the user choose at runtime:
    1) Live webcam
    2) An image file from disk

Uses the exact same MediaPipe hand-landmark + normalization pipeline as
training, so predictions line up with what the model actually learned.

Usage:
    python inference.py
    (or non-interactively:)
    python inference.py --mode webcam
    python inference.py --mode image --image path/to/photo.jpg
"""

import argparse
import os

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL_TASK_PATH = "hand_landmarker.task"
CLASSIFIER_PATH = "gsl_classifier.joblib"


def normalize_landmarks(landmarks):
    """Same normalization used during training: wrist-centered, rotation-
    corrected, and scale-invariant."""
    pts = np.array(landmarks, dtype=np.float32)
    wrist = pts[0].copy()
    pts -= wrist

    ref = pts[9][:2]
    theta = np.arctan2(ref[1], ref[0])
    delta = -np.pi / 2 - theta
    cos_d, sin_d = np.cos(delta), np.sin(delta)
    rot = np.array([[cos_d, -sin_d], [sin_d, cos_d]], dtype=np.float32)
    pts[:, :2] = (rot @ pts[:, :2].T).T

    scale = np.mean(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def load_detector():
    if not os.path.exists(MODEL_TASK_PATH):
        raise FileNotFoundError(
            f"{MODEL_TASK_PATH} not found. Download it first with:\n"
            "  python -c \"import urllib.request; urllib.request.urlretrieve("
            "'https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task', "
            "'hand_landmarker.task')\""
        )
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def load_classifier():
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"{CLASSIFIER_PATH} not found — train the model first.")
    bundle = joblib.load(CLASSIFIER_PATH)
    return bundle["model"], bundle["label_encoder"]


def predict_frame(detector, model, label_encoder, frame_bgr):
    """
    Tries both orientations (original and mirrored) automatically and uses
    whichever the model is more confident about. Returns
    (label, confidence, hand_landmarks, used_mirror) — used_mirror tells the
    caller whether to flip the display frame so the landmark overlay lines up.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    candidates = []
    for mirror in (False, True):
        img = np.ascontiguousarray(frame_rgb[:, ::-1, :]) if mirror else frame_rgb
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        result = detector.detect(mp_image)
        if not result.hand_landmarks:
            continue

        hand = result.hand_landmarks[0]
        coords = [(lm.x, lm.y, lm.z) for lm in hand]
        feats = normalize_landmarks(coords).reshape(1, -1)

        pred_idx = model.predict(feats)[0]
        label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(feats)[0][pred_idx]

        candidates.append((confidence if confidence is not None else 0.0, label, confidence, hand, mirror))

    if not candidates:
        return None, None, None, False

    candidates.sort(key=lambda c: c[0], reverse=True)
    _, label, confidence, hand, used_mirror = candidates[0]
    return label, confidence, hand, used_mirror


def draw_landmarks(frame_bgr, hand_landmarks):
    h, w = frame_bgr.shape[:2]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),          # index
        (5, 9), (9, 10), (10, 11), (11, 12),     # middle
        (9, 13), (13, 14), (14, 15), (15, 16),   # ring
        (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (0, 17),
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in connections:
        cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame_bgr, (x, y), 4, (0, 0, 255), -1)


def run_webcam(detector, model, label_encoder):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Is it connected / not in use by another app?")
        return

    print("Webcam started. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        label, confidence, hand, used_mirror = predict_frame(detector, model, label_encoder, frame)
        display_frame = cv2.flip(frame, 1) if used_mirror else frame

        if hand is not None:
            draw_landmarks(display_frame, hand)
            text = f"{label} ({confidence:.0%})" if confidence is not None else label
            cv2.putText(display_frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)
        else:
            cv2.putText(display_frame, "No hand detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2)

        cv2.imshow("GSL Recognition — press 'q' to quit", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(detector, model, label_encoder, image_path=None):
    if image_path is None:
        image_path = input("Enter the path to your image file: ").strip().strip('"')

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image (unsupported format?): {image_path}")
        return

    label, confidence, hand, used_mirror = predict_frame(detector, model, label_encoder, frame)
    if used_mirror:
        frame = cv2.flip(frame, 1)

    if hand is None:
        print("No hand detected in this image.")
        return

    if confidence is not None:
        print(f"\nPredicted sign: {label}  (confidence: {confidence:.1%})")
    else:
        print(f"\nPredicted sign: {label}")

    draw_landmarks(frame, hand)
    text = f"{label} ({confidence:.0%})" if confidence is not None else label
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Prediction — press any key to close", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["webcam", "image"], default=None,
                     help="Skip the interactive menu and go straight to this mode")
    ap.add_argument("--image", default=None, help="Image path (only used with --mode image)")
    args = ap.parse_args()

    print("Loading hand detector and classifier...")
    detector = load_detector()
    model, label_encoder = load_classifier()
    print(f"Ready. Classes: {list(label_encoder.classes_)}\n")

    mode = args.mode
    if mode is None:
        print("How would you like to provide input?")
        print("  1) Live webcam")
        print("  2) Image file from device")
        choice = input("Enter 1 or 2: ").strip()
        mode = "webcam" if choice == "1" else "image"

    if mode == "webcam":
        run_webcam(detector, model, label_encoder)
    else:
        run_image(detector, model, label_encoder, args.image)

    detector.close()


if __name__ == "__main__":
    main()
