"""
Small wrapper around the Gemini API for the app's AI features:
    - Personalized feedback after a test (sign language / math / science)
    - A simple "Ask AI Tutor" doubt-solving box

Setup (one-time):
    python -m pip install google-genai

Set your API key as an environment variable BEFORE running streamlit,
so it never ends up hardcoded in a file you might share/commit:

    Windows (Anaconda Prompt), same session you run streamlit from:
        set GEMINI_API_KEY=your_actual_key_here
        streamlit run app.py

If you restart the terminal you'll need to set it again in that session.
"""

import os
from google import genai

MODEL_NAME = "gemini-3.5-flash"

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Run: set GEMINI_API_KEY=your_key_here  (before starting streamlit)"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def ask_gemini(prompt: str) -> str:
    """Sends a prompt to Gemini and returns the plain-text response.
    Returns a friendly error string instead of raising, so the UI never crashes."""
    try:
        client = get_client()
        interaction = client.interactions.create(model=MODEL_NAME, input=prompt)
        return interaction.output_text
    except Exception as e:
        return f"(AI tutor unavailable right now: {e})"


def generate_feedback(student_name: str, results: list) -> str:
    """results: list of (subject, score, total) tuples."""
    lines = [f"{subject}: {score}/{total}" for subject, score, total in results]
    results_text = "\n".join(lines)
    prompt = (
        f"You are a warm, encouraging tutor for a deaf/mute student named {student_name} "
        f"learning Gujarati Sign Language, Math, and Science.\n"
        f"Here are their test results:\n{results_text}\n\n"
        f"Write a short (3-4 sentences), encouraging, simple-language summary of how they did, "
        f"praise their strengths, and give one gentle tip for what to practice next. "
        f"Keep it warm and age-appropriate."
    )
    return ask_gemini(prompt)
