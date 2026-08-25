"""
Exposes the trained sign-language model as a simple REST API so your
friends' frontend/backend (any language/framework) can call it over HTTP
without needing to know anything about Python, MediaPipe, or the model.

Run with:
    python model_api.py

Then it's live at:  http://localhost:5000

Endpoints:
    GET  /health          -> {"status": "ok", "classes": [...]}
    POST /predict          -> send an image file, get back the predicted sign

Example request (curl):
    curl -X POST -F "image=@test.jpg" http://localhost:5000/predict

Example response:
    {"label": "ka", "confidence": 0.97, "hand_detected": true}
    or, if no hand found:
    {"label": null, "confidence": null, "hand_detected": false}
"""

import io

import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
import joblib

MODEL_TASK_PATH = "hand_landmarker.task"
CLASSIFIER_PATH = "gsl_classifier.joblib"

app = Flask(__name__)
CORS(app)  # allows browsers on other origins (e.g. your friend's React dev server) to call this API

print("Loading models...")
_base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_TASK_PATH)
_options = mp_vision.HandLandmarkerOptions(
    base_options=_base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    running_mode=mp_vision.RunningMode.IMAGE,
)
detector = mp_vision.HandLandmarker.create_from_options(_options)

_bundle = joblib.load(CLASSIFIER_PATH)
model = _bundle["model"]
label_encoder = _bundle["label_encoder"]
print(f"Ready. Classes: {list(label_encoder.classes_)}")


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


def predict_sign(pil_image):
    """
    Tries both orientations (original and mirrored) automatically and uses
    whichever the model is more confident about.
    """
    image_rgb = np.array(pil_image.convert("RGB"))

    candidates = []
    for mirror in (False, True):
        img = np.ascontiguousarray(image_rgb[:, ::-1, :]) if mirror else image_rgb
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
            confidence = float(model.predict_proba(feats)[0][pred_idx])

        candidates.append((confidence if confidence is not None else 0.0, label, confidence))

    if not candidates:
        return None, None

    candidates.sort(key=lambda c: c[0], reverse=True)
    _, label, confidence = candidates[0]
    return label, confidence


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": list(label_encoder.classes_)})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file field in request"}), 400

    file = request.files["image"]
    try:
        pil_image = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    label, confidence = predict_sign(pil_image)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "hand_detected": label is not None,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
