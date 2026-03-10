"""
config.py — Configuration for the AI English Education Platform.

Loads API keys from environment variables, defines model routing,
PDF paths, TTS settings, RAG parameters, and app settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ─────────────────────────────────────────────
# API Keys (loaded from environment variables)
# ─────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
AZURE_SPEECH_KEY: str = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION: str = os.getenv("AZURE_SPEECH_REGION", "")
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "edge")  # "edge" (default, free) or "azure"

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
    "title": "English Tutor 🎓",
    "icon": "📚",
    "layout": "wide",
    "theme_primary": "#4A90D9",
    "theme_secondary": "#F5A623",
    "theme_success": "#7ED321",
    "theme_error": "#D0021B",
    "grade_levels": [1, 2, 3],
    "subjects": ["Phonics", "Reading", "Spelling"],
    "db_path": str(Path(__file__).parent / "data" / "students.db"),
}

# ─────────────────────────────────────────────
# Safety Settings for Gemini (education-safe)
# ─────────────────────────────────────────────
GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
