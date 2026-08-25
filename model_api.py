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
    """Tries natural orientation first, falling back to mirror if needed."""
    image_rgb = np.array(pil_image.convert("RGB"))

    # 1. Natural (Unmirrored) Orientation
    mp_image_orig = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result_orig = detector.detect(mp_image_orig)
    
    orig_label, orig_conf = None, 0.0
    if result_orig.hand_landmarks:
        coords = [(lm.x, lm.y, lm.z) for lm in result_orig.hand_landmarks[0]]
        feats = normalize_landmarks(coords).reshape(1, -1)
        pred_idx = model.predict(feats)[0]
        orig_label = label_encoder.inverse_transform([pred_idx])[0]
        if hasattr(model, "predict_proba"):
            orig_conf = float(model.predict_proba(feats)[0][pred_idx])

    # 2. Mirrored Orientation
    mirrored_rgb = np.ascontiguousarray(image_rgb[:, ::-1, :])
    mp_image_mirr = mp.Image(image_format=mp.ImageFormat.SRGB, data=mirrored_rgb)
    result_mirr = detector.detect(mp_image_mirr)
    
    mirr_label, mirr_conf = None, 0.0
    if result_mirr.hand_landmarks:
        coords = [(lm.x, lm.y, lm.z) for lm in result_mirr.hand_landmarks[0]]
        feats = normalize_landmarks(coords).reshape(1, -1)
        pred_idx = model.predict(feats)[0]
        mirr_label = label_encoder.inverse_transform([pred_idx])[0]
        if hasattr(model, "predict_proba"):
            mirr_conf = float(model.predict_proba(feats)[0][pred_idx])

    # Prefer natural orientation if hand is detected with good confidence
    if orig_label and orig_conf >= 0.35:
        return orig_label, orig_conf
    if mirr_label and mirr_conf >= 0.35:
        return mirr_label, mirr_conf
    if orig_label or mirr_label:
        return (orig_label, orig_conf) if orig_conf >= mirr_conf else (mirr_label, mirr_conf)

    return None, None


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
