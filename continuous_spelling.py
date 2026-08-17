"""
Continuous sign-to-text "spelling" mode.

Watches the live webcam feed continuously (not one snapshot at a time) and
builds up a spelled word as you hold each sign. Entirely offline - uses only
your own trained model, no internet or API key needed.

How it decides a letter is "confirmed":
    - Looks at the last ~15 frames of predictions
    - If the same sign appears in most of them with decent confidence, it
      commits that letter to the spelled word
    - Requires you to change your hand shape (or briefly remove your hand)
      before it will accept the SAME letter again - this stops "aaaaaa"
      from spamming while you hold one sign

Controls:
    q = quit
    r = reset / clear the spelled word
    b = backspace (remove last letter)
    space (while no hand is shown) = add a space between words automatically

Usage:
    python continuous_spelling.py
"""

import os
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL_TASK_PATH = "hand_landmarker.task"
CLASSIFIER_PATH = "gsl_classifier.joblib"

HISTORY_LEN = 15          # how many recent frames to look at
STABILITY_THRESHOLD = 0.7  # fraction of history that must agree
MIN_CONFIDENCE = 0.5       # per-frame confidence needed to count
NO_HAND_FRAMES_FOR_SPACE = 20  # ~ frames of no hand before auto-inserting a space


def normalize_landmarks(landmarks):
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
        raise FileNotFoundError(f"{MODEL_TASK_PATH} not found in this folder.")
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
        raise FileNotFoundError(f"{CLASSIFIER_PATH} not found in this folder.")
    bundle = joblib.load(CLASSIFIER_PATH)
    return bundle["model"], bundle["label_encoder"]


def predict_one_frame(detector, model, label_encoder, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None, None, None

    hand = result.hand_landmarks[0]
    coords = [(lm.x, lm.y, lm.z) for lm in hand]
    feats = normalize_landmarks(coords).reshape(1, -1)

    pred_idx = model.predict(feats)[0]
    label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(feats)[0][pred_idx]
    return label, confidence, hand


def draw_landmarks(frame_bgr, hand_landmarks):
    h, w = frame_bgr.shape[:2]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in connections:
        cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame_bgr, (x, y), 4, (0, 0, 255), -1)


def main():
    print("Loading models...")
    detector = load_detector()
    model, label_encoder = load_classifier()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Ready. Controls: q=quit  r=reset  b=backspace\n")

    history = deque(maxlen=HISTORY_LEN)
    spelled_word = ""
    last_committed = None
    no_hand_streak = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        label, confidence, hand = predict_one_frame(detector, model, label_encoder, frame)

        if label is not None and (confidence is None or confidence >= MIN_CONFIDENCE):
            history.append(label)
            no_hand_streak = 0
        else:
            history.append(None)
            no_hand_streak += 1

        # check for a stable, confirmed letter
        if len(history) == HISTORY_LEN:
            non_none = [h for h in history if h is not None]
            if non_none:
                most_common = max(set(non_none), key=non_none.count)
                agreement = non_none.count(most_common) / HISTORY_LEN
                if agreement >= STABILITY_THRESHOLD and most_common != last_committed:
                    spelled_word += most_common
                    last_committed = most_common
                    history.clear()  # require fresh stability before next letter

        # allow the same letter again once the hand has been away for a bit
        if no_hand_streak > 5:
            last_committed = None

        # auto space after a longer pause with no hand
        if no_hand_streak == NO_HAND_FRAMES_FOR_SPACE and spelled_word and not spelled_word.endswith(" "):
            spelled_word += " "

        # ---- draw UI ----
        if hand is not None:
            draw_landmarks(frame, hand)
        text = f"{label} ({confidence:.0%})" if label and confidence is not None else "No hand"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.rectangle(frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(frame, f"Spelled: {spelled_word}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("Continuous Sign-to-Text (offline)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            spelled_word = ""
            last_committed = None
            history.clear()
        elif key == ord("b"):
            spelled_word = spelled_word[:-1]

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"\nFinal spelled text: {spelled_word}")


if __name__ == "__main__":
    main()
