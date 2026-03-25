"""
config.py — Configuration for the AI English Education Platform.

Loads API keys from environment variables, defines model routing,
PDF paths, TTS settings, RAG parameters, and app settings.

Key-loading priority (most secure first):
  1. Environment variables (export GEMINI_API_KEY=... in your shell or .env file)
  2. config_secrets.json in the project root (beginner-friendly fallback — git-ignored)

Never hardcode real API keys in this file or commit them to version control.
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present (standard python-dotenv approach)
load_dotenv()

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Secrets-file fallback
# ─────────────────────────────────────────────

def _load_secrets_file() -> dict:
    """
    Load API keys from config_secrets.json in the project root.

    This file is listed in .gitignore and must never be committed.
    It provides a beginner-friendly alternative to shell environment variables.
    Returns an empty dict if the file is absent or cannot be parsed.
    """
    secrets_path = Path(__file__).parent / "config_secrets.json"
    if not secrets_path.exists():
        return {}
    try:
        with secrets_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            _logger.warning("config_secrets.json must contain a JSON object — ignoring.")
            return {}
        return data
    except (json.JSONDecodeError, OSError) as exc:
        _logger.warning("Could not read config_secrets.json: %s", exc)
        return {}


def _get_key(name: str, secrets: dict, default: str = "") -> str:
    """
    Return the value for *name*, checking environment variables first,
    then the secrets dict (loaded from config_secrets.json), then *default*.

    Environment variables take priority even when their value is an empty string,
    so an explicit `export FOO=` in the shell is respected and the secrets file
    is not consulted.
    """
    if name in os.environ:
        return os.environ[name]
    return secrets.get(name, default)


_SECRETS: dict = _load_secrets_file()

# ─────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────

# Common misspellings of GEMINI_API_KEY that users accidentally set.
_GEMINI_KEY_TYPOS: tuple = (
    "GEMINIAI_API_KEY",   # extra "AI" — most common user mistake
    "GEMINI_APIKEY",
    "GEMINI_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GEMINI_API_KEY",
)

# Placeholder strings that must never be treated as real keys.
# Keep these in sync with config_secrets.json.example and .env.example.
_KEY_PLACEHOLDERS: frozenset = frozenset({
    "your_gemini_api_key_here",
    "AIza...",
    "placeholder",
    "your_key_here",
    "",
})


def _warn_gemini_key_typos() -> None:
    """
    Emit a clear warning if a common GEMINI_API_KEY misspelling is set in
    the environment but the canonical GEMINI_API_KEY is missing.

    This helps users who copied code like ``os.environ["GEMINIAI_API_KEY"] = …``
    without realising the correct name is ``GEMINI_API_KEY``.
    """
    if os.environ.get("GEMINI_API_KEY"):
        return  # Correct key present — nothing to warn about.
    for typo in _GEMINI_KEY_TYPOS:
        if os.environ.get(typo):
            _logger.warning(
                "Found environment variable '%s' but the app requires 'GEMINI_API_KEY'. "
                "Please rename it: export GEMINI_API_KEY=<your_key>",
                typo,
            )
            return


def validate_gemini_key(key: str) -> tuple[bool, str]:
    """
    Perform basic sanity checks on a Gemini API key string.

    Returns:
        (is_valid, message) — ``is_valid`` is False when a known problem is
        detected; ``message`` explains what to check.
    """
    if not key:
        return False, (
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or config_secrets.json. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
    stripped = key.strip()
    if stripped != key:
        return False, (
            "GEMINI_API_KEY contains leading or trailing whitespace. "
            "Make sure you copy only the key characters with no spaces."
        )
    if key in _KEY_PLACEHOLDERS:
        return False, (
            "GEMINI_API_KEY looks like a placeholder, not a real key. "
            "Replace it with your actual key from https://aistudio.google.com/app/apikey"
        )
    if not key.startswith("AIza"):
        return False, (
            "GEMINI_API_KEY does not look like a valid Google API key "
            "(expected to start with 'AIza'). "
            "Double-check that you copied the key correctly."
        )
    return True, "OK"


_warn_gemini_key_typos()

GEMINI_API_KEY: str = _get_key("GEMINI_API_KEY", _SECRETS)
AZURE_SPEECH_KEY: str = _get_key("AZURE_SPEECH_KEY", _SECRETS)
AZURE_SPEECH_REGION: str = _get_key("AZURE_SPEECH_REGION", _SECRETS)
TTS_PROVIDER: str = _get_key("TTS_PROVIDER", _SECRETS, "edge")  # "edge" (default, free) or "azure"

# ─────────────────────────────────────────────
# Model Routing Config
# Structured for easy expansion to OpenAI / Claude later.
# Each entry has: model_id, temperature defaults, max_tokens.
# ─────────────────────────────────────────────
MODELS: dict = {
    "gemini": {
        "provider": "google",
        "model_id": "gemini-1.5-flash",
        "temperature_explain": 0.7,    # Conversational explanations
        "temperature_quiz": 0.3,       # Deterministic quiz generation
        "max_tokens": 2048,
        "api_key_env": "GEMINI_API_KEY",
    },
    # Placeholder entries for future expansion:
    "openai": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "temperature_explain": 0.7,
        "temperature_quiz": 0.3,
        "max_tokens": 2048,
        "api_key_env": "OPENAI_API_KEY",
    },
    "claude": {
        "provider": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",
        "temperature_explain": 0.7,
        "temperature_quiz": 0.3,
        "max_tokens": 2048,
        "api_key_env": "ANTHROPIC_API_KEY",
    },
}

# Which model to use for all tasks right now
ACTIVE_MODEL: str = "gemini"

# ─────────────────────────────────────────────
# Curriculum PDF Paths
# ─────────────────────────────────────────────
CURRICULUM_DIR: Path = Path(__file__).parent / "curriculum"
PDF_PATHS: dict = {
    "phonics": CURRICULUM_DIR / "phonics.pdf",
    "reading": CURRICULUM_DIR / "reading.pdf",
    "spelling": CURRICULUM_DIR / "Spelling.pdf",
}

# ─────────────────────────────────────────────
# RAG (Retrieval-Augmented Generation) Config
# ─────────────────────────────────────────────
RAG_CONFIG: dict = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "all-MiniLM-L6-v2",  # Free, local HuggingFace model
    "faiss_index_path": str(Path(__file__).parent / "data" / "faiss_index"),
    "top_k": 4,  # Number of relevant chunks to retrieve
}

# ─────────────────────────────────────────────
# TTS (Text-to-Speech) Config
# ─────────────────────────────────────────────
TTS_CONFIG: dict = {
    "provider": TTS_PROVIDER,              # "edge" or "azure"
    "edge_voice": "en-US-AriaNeural",      # High-quality free Microsoft voice
    "azure_voice": "en-US-JennyNeural",    # Azure premium voice
    "azure_key": AZURE_SPEECH_KEY,
    "azure_region": AZURE_SPEECH_REGION,
    "audio_temp_path": "/tmp/tts_audio.mp3",
}

# ─────────────────────────────────────────────
# App Settings
# ─────────────────────────────────────────────
APP_CONFIG: dict = {
    "title": "Taunggyi English Tutor 🎓",
    "icon": "📚",
    "layout": "wide",
    "theme_primary": "#4A90D9",
    "theme_secondary": "#F5A623",
    "theme_success": "#7ED321",
    "theme_error": "#D0021B",
    "grade_levels": [1, 2, 3, 4, 5],
    "subjects": ["Phonics", "Reading", "Spelling"],
    "db_path": str(Path(__file__).parent / "data" / "students.db"),
}

# ─────────────────────────────────────────────
# Personalization Settings
# ─────────────────────────────────────────────
CONFUSION_THRESHOLD: int = 2          # wrong answers in a row to trigger confusion detection
MASTERY_ADVANCE_THRESHOLD: float = 0.90  # 90% mastery to suggest grade advancement
MASTERY_SKIP_THRESHOLD: float = 0.95     # 95% to skip/not repeat content
HINT_LEVELS: int = 4                  # number of progressive hints before revealing answer
PACE_FAST_THRESHOLD: float = 0.80     # accuracy above this = speed up
PACE_SLOW_THRESHOLD: float = 0.50     # accuracy below this = slow down

# Spaced repetition for vocabulary
VOCAB_REVIEW_INTERVALS: list = [1, 3, 7, 14, 30]  # days between reviews

# Session planning
DEFAULT_SESSION_DURATION: int = 30    # minutes
SESSION_WARMUP_RATIO: float = 0.07    # 7% of session
SESSION_NEW_CONTENT_RATIO: float = 0.50  # 50% of session
SESSION_PRACTICE_RATIO: float = 0.27  # 27% of session
SESSION_COOLDOWN_RATIO: float = 0.16  # 16% of session

# Grade levels available
GRADE_LEVELS: list = [1, 2, 3, 4, 5]  # expandable
SUBJECTS: list = ["Phonics", "Reading", "Spelling"]  # expandable

# Voice/STT settings
AZURE_STT_LANGUAGE: str = "en-US"
PRONUNCIATION_EXCELLENT_THRESHOLD: float = 0.90
PRONUNCIATION_GOOD_THRESHOLD: float = 0.70

# Language support
SUPPORTED_LANGUAGES: list = ["en"]
DEFAULT_LANGUAGE: str = "en"

# ─────────────────────────────────────────────
# AI Teacher Identity
# Change this one value to update the teacher's name everywhere.
# ─────────────────────────────────────────────
AI_TEACHER_NAME: str = "San"

# ─────────────────────────────────────────────
# Safety Settings for Gemini (education-safe)
# ─────────────────────────────────────────────
GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
