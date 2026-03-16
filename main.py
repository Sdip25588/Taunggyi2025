"""
main.py — Streamlit application entry point.

Run with:
    streamlit run main.py

Handles:
- Page configuration
- Username-based login
- RAG pipeline initialisation
- Session state management
- Routing to the main GUI
"""

import logging
from pathlib import Path

import streamlit as st

from config import APP_CONFIG, GEMINI_API_KEY
import ai_teacher
import learning_orchestrator
import student as student_db
import gui_engine

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Page Config (must be FIRST Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon=APP_CONFIG["icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Taunggyi English Tutor** — AI-powered English education platform "
            "for Grades 1-3. Powered by Google Gemini 1.5 Flash + RAG."
        ),
    },
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #FAFAFA; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: bold; }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74, 144, 217, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────

def _init_session_state() -> None:
    """Initialise all session state keys with safe defaults."""
    defaults = {
        "student_profile": None,
        "chat_history": [],
        "current_subject": "Phonics",
        "pending_quiz": None,
        "pending_quiz_list": None,
        "quiz_index": 0,
        "quiz_score": 0,
        "rag_initialised": False,
        "show_visual": False,
        "suggested_visual": None,
        # Professor / conversation mode state
        "conv_state": learning_orchestrator.CONV_GREETING,
        "greeting_done": False,
        "todays_focus": None,
        # Extended conversation mode state (PR #4 additions)
        "conversation_mode": True,
        "conversation_state": "GREETING",
        "pending_tutor_prompt": "",
        "awaiting_student_reply": False,
        "conversation_greeted": False,
        # Onboarding state
        "onboard_visual": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────
# RAG Initialisation
# ─────────────────────────────────────────────

def _init_rag() -> None:
    """
    Initialise the RAG pipeline once per session.

    Tries to load an existing FAISS index; if none exists, tries to build
    one from available PDFs.  Shows a graceful warning if PDFs are missing.
    """
    if st.session_state.rag_initialised:
        return

    rag_status = ai_teacher.get_rag_status()

    if rag_status["index_exists"]:
        # Load existing index silently
        ai_teacher.load_faiss_index()
        st.session_state.rag_initialised = True
        logger.info("FAISS index loaded from disk.")
        return

    # Check if any PDFs exist
    pdfs = rag_status["pdfs"]
    any_pdf = any(pdfs.values())

    if any_pdf:
        with st.spinner("📚 Loading curriculum PDFs and building index... (first run only)"):
            success = ai_teacher.build_faiss_index()
        if success:
            st.toast("✅ Curriculum loaded successfully!", icon="📚")
        else:
            st.warning(
                "⚠️ Could not build curriculum index. "
                "Check that PDF dependencies are installed."
            )
    else:
        # No PDFs available — show setup instructions
        logger.info("No curriculum PDFs found. RAG will be disabled.")

    st.session_state.rag_initialised = True


# ─────────────────────────────────────────────
# API Key Check
# ─────────────────────────────────────────────

def _check_api_keys() -> None:
    """Display a warning banner if the Gemini API key is missing."""
    if not GEMINI_API_KEY:
        st.warning(
            "⚠️ **Gemini API key not configured.** "
            "AI features will show error messages until you add your key.\n\n"
            "**Setup:** Copy `.env.example` to `.env` and add your `GEMINI_API_KEY`.\n\n"
            "Get a free key at: https://aistudio.google.com/app/apikey",
            icon="🔑",
        )


# ─────────────────────────────────────────────
# Login Screen
# ─────────────────────────────────────────────

def _render_login() -> None:
    """Render the simple username-based login screen."""
    st.title(f"{APP_CONFIG['icon']} {APP_CONFIG['title']}")
    st.markdown(
        "### Welcome! Please enter your name to start learning. 👋\n\n"
        "No password needed — just your name!"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "📝 Your Name",
                placeholder="e.g., Aung Myat",
                max_chars=50,
                help="Enter your name to load your learning profile.",
            )
            grade = st.selectbox(
                "🎓 Your Grade",
                APP_CONFIG["grade_levels"],
                format_func=lambda g: f"Grade {g}",
            )
            submitted = st.form_submit_button(
                "🚀 Start Learning!", use_container_width=True
            )

        if submitted:
            if not username.strip():
                st.error("Please enter your name to continue.")
            else:
                with st.spinner("Loading your profile..."):
                    profile = student_db.get_or_create_student(username.strip())
                    # Update grade if changed
                    if profile.get("grade_level") != grade:
                        student_db.update_student_field(
                            username.strip(), "grade_level", grade
                        )
                        profile["grade_level"] = grade

                st.session_state.student_profile = profile
                st.session_state.current_subject = profile.get(
                    "current_subject", "Phonics"
                )
                # Reset professor/conversation state for the new session
                st.session_state.conv_state = learning_orchestrator.CONV_GREETING
                st.session_state.greeting_done = False
                st.session_state.todays_focus = None
                st.session_state.chat_history = []
                st.rerun()

    # Feature highlights
    st.divider()
    st.markdown("### ✨ What You Can Learn")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🔤 **Phonics**\nSounds, letters, blends, digraphs")
    with col2:
        st.success("📖 **Reading**\nMcGuffey's First Reader — step by step")
    with col3:
        st.warning("📝 **Spelling**\nWord families, patterns, and practice")


# ─────────────────────────────────────────────
# Main Application Flow
# ─────────────────────────────────────────────

def main() -> None:
    """Application entry point."""
    _init_session_state()

    if st.session_state.student_profile is None:
        # Show login screen
        _check_api_keys()
        _render_login()
    else:
        # Initialise RAG pipeline (once per session)
        _init_rag()
        _check_api_keys()
        # Render the main application
        gui_engine.render_app()


if __name__ == "__main__":
    main()
