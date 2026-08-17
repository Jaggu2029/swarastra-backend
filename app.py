"""
SwarAstra - Gujarati Sign Language Learning App
Single self-contained Streamlit app. No separate API server needed -
the model runs directly in this process.

Run with:
    streamlit run app.py

Requires (same folder):
    gsl_classifier.joblib
    hand_landmarker.task
    dataset/               (folders of images per sign, used for the
                             Learning slider reference photos)
    questions_math.json
    questions_science.json
"""

import json
import os
import random
import smtplib
from email.mime.text import MIMEText

import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

import gemini_helper

MODEL_TASK_PATH = "hand_landmarker.task"
CLASSIFIER_PATH = "gsl_classifier.joblib"
DATASET_DIR = "dataset"
PROGRESS_FILE = "progress_data.json"

# Gujarati script for known alphabet labels - fallback to just showing the
# English name for any label not in this map (e.g. newer additions).
GUJARATI_SCRIPT = {
    "ka": "ક", "kha": "ખ", "ga": "ગ", "gha": "ઘ", "nga": "ઙ",
    "cha": "ચ", "chha": "છ", "ja": "જ", "jha": "ઝ", "nya": "ઞ",
    "ta": "ત", "tha": "થ", "da": "દ", "dha": "ધ", "na": "ન",
    "pa": "પ", "pha": "ફ", "ba": "બ", "bha": "ભ", "ma": "મ",
    "ya": "ય", "ra": "ર", "la": "લ", "va": "વ", "sha": "શ",
    "sa": "સ", "ha": "હ",
    "Ta": "ટ", "Tha": "ઠ", "Da": "ડ", "Dha": "ઢ", "Na": "ણ",
    "Sha": "ષ", "La": "ળ", "ksha": "ક્ષ", "gyna": "જ્ઞ",
}

EMAIL_ENABLED = False
SENDER_EMAIL = "your_app_email@gmail.com"
SENDER_APP_PASSWORD = "your_16_char_app_password"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

MATH_LEVELS = [
    {"id": 1, "name": "Level 1 (Addition)"},
    {"id": 2, "name": "Level 2 (Large Addition)"},
    {"id": 3, "name": "Level 3 (Subtraction)"},
    {"id": 4, "name": "Level 4 (Multiplication)"},
]
SCIENCE_LEVELS = [
    {"id": 1, "name": "Level 1 (Living & Non-Living / Our Body)"},
    {"id": 2, "name": "Level 2 (Plants & Animals)"},
    {"id": 3, "name": "Level 3 (Matter & Simple Machines)"},
    {"id": 4, "name": "Level 4 (Forces, Energy & Environment)"},
]

# ============================== Styling ==============================

def inject_css():
    st.markdown("""
    <style>
    .stApp { background-color: #090909; }
    section[data-testid="stSidebar"] { display: none; }
    .card {
        border-radius: 20px; padding: 28px 24px; cursor: pointer;
        color: white; margin-bottom: 16px; transition: transform 0.15s;
    }
    .card:hover { transform: translateY(-2px); }
    .card-title { font-size: 20px; font-weight: 800; margin-bottom: 4px; }
    .card-sub { font-size: 13px; opacity: 0.85; }
    .swar-title {
        font-size: 42px; font-weight: 900; text-align: center;
        background: linear-gradient(90deg, #6a4cf5, #d44df0, #ff7a3d);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .swar-sub { text-align: center; color: #999; margin-bottom: 24px; }
    div.stButton > button {
        border-radius: 12px; font-weight: 700; padding: 10px 18px;
    }
    </style>
    """, unsafe_allow_html=True)


def card(title, subtitle, gradient, key):
    st.markdown(f"""
    <div class="card" style="background: {gradient};">
        <div class="card-title">{title}</div>
        <div class="card-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)
    return st.button("Open", key=key, use_container_width=True)


# ============================== Model loading ==============================

@st.cache_resource
def load_sign_detector():
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=1,
        min_hand_detection_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


@st.cache_resource
def load_sign_classifier():
    bundle = joblib.load(CLASSIFIER_PATH)
    return bundle["model"], bundle["label_encoder"]


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


def predict_sign(detector, model, label_encoder, pil_image):
    """Tries both orientations automatically, picks the more confident one."""
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
            confidence = model.predict_proba(feats)[0][pred_idx]
        candidates.append((confidence if confidence is not None else 0.0, label, confidence))
    if not candidates:
        return None, None
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, label, confidence = candidates[0]
    return label, confidence


# ============================== Data helpers ==============================

def load_questions(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sign_labels():
    if not os.path.isdir(DATASET_DIR):
        return []
    return sorted(d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)))


def get_reference_image(label):
    folder = os.path.join(DATASET_DIR, label)
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return os.path.join(folder, files[0]) if files else None


def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_progress_entry(module, level, accuracy):
    data = load_progress()
    data.setdefault(module, {})
    key = str(level)
    prev = data[module].get(key, 0)
    data[module][key] = max(prev, accuracy)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f)


def get_best_accuracy(module, level):
    data = load_progress()
    return data.get(module, {}).get(str(level), 0)


def send_report_email(parent_email, student_name, results):
    if not EMAIL_ENABLED:
        return False, "Email sending is not configured (EMAIL_ENABLED = False in app.py)."
    body = f"Learning report for {student_name}\n\n" + "\n".join(
        f"{s}: {sc}/{t}" for s, sc, t in results
    )
    msg = MIMEText(body)
    msg["Subject"] = f"Learning Report - {student_name}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = parent_email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, [parent_email], msg.as_string())
        return True, "Email sent successfully."
    except Exception as e:
        return False, f"Failed to send email: {e}"


# ============================== Session state ==============================

def init_state():
    defaults = {
        "page": "age_gate",
        "is_minor": None,
        "student_name": "", "student_email": "", "parent_email": "", "contact_num": "",
        "role": None,
        "results": [],
        "sign_slider_idx": 0,
        "sign_targets": [], "sign_index": 0, "sign_score": 0,
        "quiz_module": None, "quiz_level": None,
        "quiz_index": 0, "quiz_score": 0, "quiz_questions": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def goto(page):
    st.session_state.page = page
    st.rerun()


# ============================== Onboarding pages ==============================

def page_age_gate():
    st.markdown('<div class="swar-title">SwarAstra</div>', unsafe_allow_html=True)
    st.markdown('<div class="swar-sub">Are you 14 years old or older?</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    if col1.button("I am 14 or older", use_container_width=True):
        st.session_state.is_minor = False
        goto("role_select")
    if col2.button("I am under 14", use_container_width=True):
        st.session_state.is_minor = True
        goto("minor_form")


def page_minor_form():
    st.markdown('<div class="swar-title">Tell us about yourself</div>', unsafe_allow_html=True)
    with st.form("minor_form"):
        name = st.text_input("Your name")
        email = st.text_input("Your email (optional)")
        parent_email = st.text_input("Parent's email (for progress reports)")
        contact = st.text_input("Contact number")
        submitted = st.form_submit_button("Continue")
    if submitted:
        if not name or not parent_email:
            st.error("Please fill in at least your name and parent's email.")
        else:
            st.session_state.student_name = name
            st.session_state.student_email = email
            st.session_state.parent_email = parent_email
            st.session_state.contact_num = contact
            goto("home")
    if st.button("Back"):
        goto("age_gate")


def page_role_select():
    st.markdown('<div class="swar-title">Are you a...</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    if col1.button("Teacher", use_container_width=True):
        st.session_state.role = "teacher"
        goto("teacher_home")
    if col2.button("Student", use_container_width=True):
        st.session_state.role = "student"
        goto("minor_form")
    if st.button("Back"):
        goto("age_gate")


# ============================== Home ==============================

def page_home():
    st.markdown('<div class="swar-title">Learn. Practice. Grow.</div>', unsafe_allow_html=True)
    name = st.session_state.student_name or "Friend"
    st.markdown(f'<div class="swar-sub">Welcome, {name} 👋 &nbsp; Master Gujarati sign language, maths and science — track your progress in one place.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if card("Sign Language", "Tap to explore →", "linear-gradient(135deg,#6a4cf5,#a63cf0)", "card_sign"):
            goto("sign_menu")
    with c2:
        if card("Maths", "Tap to explore →", "linear-gradient(135deg,#ff7a3d,#ff5470)", "card_maths"):
            goto("maths_menu")
    with c3:
        if card("Science", "Tap to explore →", "linear-gradient(135deg,#d44df0,#8a4cf5)", "card_science"):
            goto("science_menu")

    if card("Progress Report", "Tap to explore →", "linear-gradient(135deg,#2ecc71,#27ae60)", "card_progress"):
        goto("progress_report")

    st.divider()
    st.markdown("#### 🤖 Ask AI Tutor")
    question = st.text_input("What is the sign for 'Hello'?", key="ai_question", label_visibility="collapsed")
    if st.button("Ask"):
        if question.strip():
            with st.spinner("Thinking..."):
                answer = gemini_helper.ask_gemini(
                    f"You are a friendly tutor for a Gujarati sign language learning app. Explain simply and briefly: {question}"
                )
            st.info(answer)


# ============================== Sign Language ==============================

def page_sign_menu():
    st.markdown('<div class="swar-title">Sign Language</div>', unsafe_allow_html=True)
    if st.button("← Back to Home"):
        goto("home")
    col1, col2 = st.columns(2)
    with col1:
        if card("Learning", "Tap to explore →", "linear-gradient(135deg,#6a4cf5,#a63cf0)", "sign_learn_card"):
            st.session_state.sign_slider_idx = 0
            goto("sign_learn")
    with col2:
        if card("Practice", "Tap to explore →", "linear-gradient(135deg,#a63cf0,#d44df0)", "sign_practice_card"):
            goto("sign_practice")


def page_sign_learn():
    labels = get_sign_labels()
    if not labels:
        st.error("No sign folders found in dataset/. Make sure app.py is in the same folder as dataset/.")
        if st.button("← Back"):
            goto("sign_menu")
        return

    idx = st.session_state.sign_slider_idx % len(labels)
    label = labels[idx]
    gujarati = GUJARATI_SCRIPT.get(label, "")

    st.markdown('<div class="swar-title">Sign Language</div>', unsafe_allow_html=True)
    if st.button("← Back"):
        goto("sign_menu")

    st.markdown(f'<p style="text-align:center;color:#999;">{idx + 1} / {len(labels)}</p>', unsafe_allow_html=True)

    img_path = get_reference_image(label)
    col_l, col_mid, col_r = st.columns([1, 4, 1])
    with col_l:
        if st.button("←", key="slider_prev", use_container_width=True):
            st.session_state.sign_slider_idx = (idx - 1) % len(labels)
            st.rerun()
    with col_mid:
        if img_path:
            st.image(img_path, use_container_width=True)
        st.markdown(f'<div style="text-align:center;"><span style="font-size:40px;font-weight:900;color:#6a4cf5;">{label}</span> &nbsp; <span style="font-size:32px;">{gujarati}</span></div>', unsafe_allow_html=True)
    with col_r:
        if st.button("→", key="slider_next", use_container_width=True):
            st.session_state.sign_slider_idx = (idx + 1) % len(labels)
            st.rerun()


SIGN_QUESTIONS_PER_PRACTICE = 10

def page_sign_practice():
    st.markdown('<div class="swar-title">Sign Practice</div>', unsafe_allow_html=True)
    if st.button("← Back to Sign Language"):
        st.session_state.sign_targets = []
        goto("sign_menu")

    detector = load_sign_detector()
    model, label_encoder = load_sign_classifier()

    if not st.session_state.sign_targets:
        classes = list(label_encoder.classes_)
        st.session_state.sign_targets = random.sample(classes, k=min(SIGN_QUESTIONS_PER_PRACTICE, len(classes)))
        st.session_state.sign_index = 0
        st.session_state.sign_score = 0

    idx = st.session_state.sign_index
    targets = st.session_state.sign_targets

    if idx >= len(targets):
        accuracy = round((st.session_state.sign_score / len(targets)) * 100)
        save_progress_entry("sign", 1, accuracy)
        st.success(f"Test complete! Score: {st.session_state.sign_score}/{len(targets)} ({accuracy}%)")
        st.session_state.results.append(("Sign Language", st.session_state.sign_score, len(targets)))
        if st.button("Practice Again"):
            st.session_state.sign_targets = []
            st.rerun()
        return

    target = targets[idx]
    st.markdown(f'<p style="text-align:center;color:#999;">Sign {idx + 1} / {len(targets)}</p>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;font-size:36px;font-weight:800;color:#6a4cf5;">Show: {target}</div>', unsafe_allow_html=True)

    img = st.camera_input("Show your hand sign", key=f"cam_{idx}", label_visibility="collapsed")

    if img is not None:
        pil_img = Image.open(img)
        label, confidence = predict_sign(detector, model, label_encoder, pil_img)
        if label is None:
            st.warning("No hand detected — try again with your hand clearly in frame.")
        else:
            conf_text = f" ({confidence:.0%} confident)" if confidence is not None else ""
            if label == target:
                st.success(f"Correct! You signed '{label}'{conf_text}")
            else:
                st.error(f"Not quite — you signed '{label}'{conf_text}, target was '{target}'")
            if st.button("Next sign"):
                if label == target:
                    st.session_state.sign_score += 1
                st.session_state.sign_index += 1
                st.rerun()


# ============================== Maths / Science (leveled) ==============================

def page_level_menu(subject, levels, path):
    st.markdown(f'<div class="swar-title">{subject.capitalize()}</div>', unsafe_allow_html=True)
    if st.button("← Back to Home"):
        goto("home")
    st.markdown('<div class="swar-sub">Select a Level</div>', unsafe_allow_html=True)

    all_questions = load_questions(path)
    cols = st.columns(2)
    for i, lvl in enumerate(levels):
        best = get_best_accuracy(subject, lvl["id"])
        with cols[i % 2]:
            st.markdown(f"""
            <div class="card" style="background:#1a1a1a;border:1px solid #333;">
                <div class="card-title">{lvl['name']}</div>
                <div class="card-sub">Best Accuracy: {best}%</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Start", key=f"{subject}_lvl_{lvl['id']}", use_container_width=True):
                n = max(1, len(all_questions) // len(levels))
                start = (lvl["id"] - 1) * n
                q_slice = all_questions[start:start + n] or all_questions[:5]
                st.session_state.quiz_module = subject
                st.session_state.quiz_level = lvl["id"]
                st.session_state.quiz_questions = q_slice
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                goto("quiz")


def page_quiz():
    module = st.session_state.quiz_module
    questions = st.session_state.quiz_questions
    st.markdown(f'<div class="swar-title">{module.capitalize()} Quiz</div>', unsafe_allow_html=True)

    idx = st.session_state.quiz_index
    if idx >= len(questions):
        accuracy = round((st.session_state.quiz_score / len(questions)) * 100)
        save_progress_entry(module, st.session_state.quiz_level, accuracy)
        st.success(f"Quiz complete! Score: {st.session_state.quiz_score}/{len(questions)} ({accuracy}%)")
        st.session_state.results.append((module.capitalize(), st.session_state.quiz_score, len(questions)))
        if st.button("Back to Levels"):
            goto(f"{module}_menu")
        return

    q = questions[idx]
    st.markdown(f'<p style="color:#999;">Question {idx + 1} / {len(questions)}</p>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:28px;font-weight:700;text-align:center;color:#ff7a3d;">{q["question"]}</div>', unsafe_allow_html=True)
    choice = st.radio("Choose an answer:", q["options"], key=f"q_{idx}", label_visibility="collapsed")

    if st.button("Submit answer"):
        if choice == q["answer"]:
            st.session_state.quiz_score += 1
            st.success("Correct!")
        else:
            st.error(f"Incorrect. The answer was: {q['answer']}")
        st.session_state.quiz_index += 1
        st.rerun()

    if st.button("Quit quiz"):
        goto(f"{module}_menu")


# ============================== Progress Report ==============================

def page_progress_report():
    st.markdown('<div class="swar-title">Progress Report</div>', unsafe_allow_html=True)
    if st.button("← Back to Home"):
        goto("home")

    data = load_progress()
    if not data:
        st.info("No practice attempts yet — go try a module!")
        return

    for module, levels in data.items():
        st.markdown(f"#### {module.capitalize()}")
        for level, acc in levels.items():
            st.progress(acc / 100, text=f"Level {level}: {acc}%")

    if st.session_state.parent_email:
        st.divider()
        if st.button("Send report to parent"):
            ok, msg = send_report_email(
                st.session_state.parent_email, st.session_state.student_name, st.session_state.results
            )
            (st.success if ok else st.warning)(msg)


def page_teacher_home():
    st.markdown('<div class="swar-title">Teacher Dashboard</div>', unsafe_allow_html=True)
    if st.button("← Back"):
        goto("age_gate")

    subject = st.selectbox("Subject", ["math", "science"])
    path = f"questions_{subject}.json"
    questions = load_questions(path)

    with st.form("add_question"):
        q_text = st.text_input("Question")
        opt1 = st.text_input("Option 1")
        opt2 = st.text_input("Option 2")
        opt3 = st.text_input("Option 3")
        opt4 = st.text_input("Option 4")
        answer = st.text_input("Correct answer (must match one option exactly)")
        submitted = st.form_submit_button("Add question")

    if submitted:
        options = [opt1, opt2, opt3, opt4]
        if not q_text or not all(options) or answer not in options:
            st.error("Fill in all fields, and make sure the correct answer matches one of the options.")
        else:
            questions.append({"question": q_text, "options": options, "answer": answer})
            with open(path, "w", encoding="utf-8") as f:
                json.dump(questions, f, indent=2, ensure_ascii=False)
            st.success("Question added!")

    st.divider()
    st.markdown(f"#### Current {subject} questions ({len(questions)})")
    for i, q in enumerate(questions, 1):
        st.write(f"{i}. {q['question']} (Answer: {q['answer']})")


# ============================== Main ==============================

def main():
    st.set_page_config(page_title="SwarAstra", layout="centered")
    inject_css()
    init_state()

    page = st.session_state.page
    if page == "age_gate":
        page_age_gate()
    elif page == "minor_form":
        page_minor_form()
    elif page == "role_select":
        page_role_select()
    elif page == "home":
        page_home()
    elif page == "sign_menu":
        page_sign_menu()
    elif page == "sign_learn":
        page_sign_learn()
    elif page == "sign_practice":
        page_sign_practice()
    elif page == "maths_menu":
        page_level_menu("maths", MATH_LEVELS, "questions_math.json")
    elif page == "science_menu":
        page_level_menu("science", SCIENCE_LEVELS, "questions_science.json")
    elif page == "quiz":
        page_quiz()
    elif page == "progress_report":
        page_progress_report()
    elif page == "teacher_home":
        page_teacher_home()


if __name__ == "__main__":
    main()
