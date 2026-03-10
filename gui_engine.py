"""
gui_engine.py — Streamlit web UI for the AI English Education Platform.

Renders the full Streamlit interface: sidebar, chat, quizzes, progress
dashboard, visual aids, and TTS controls.
"""

import asyncio
import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

from config import APP_CONFIG, TTS_CONFIG
import learning_orchestrator
import visual_teacher
import adaptive_path
import student as student_db
import ai_teacher

logger = logging.getLogger(__name__)

AVATAR_TEACHER = "🧑‍🏫"
AVATAR_STUDENT = "🧒"


# ─────────────────────────────────────────────
# Main entry point called from main.py
# ─────────────────────────────────────────────

def render_app() -> None:
    """Render the full application UI after login."""
    _render_sidebar()
    _render_main_area()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def _render_sidebar() -> None:
    """Render the left sidebar with student info and controls."""
    with st.sidebar:
        st.title("📚 My Learning Dashboard")
        st.divider()

        # Student info
        username = st.session_state.student_profile.get("username", "Student")
        grade = st.session_state.student_profile.get("grade_level", 1)
        st.markdown(f"👤 **{username}**")
        st.markdown(f"🏫 Grade **{grade}**")
        st.divider()

        # Grade selector
        new_grade = st.selectbox(
            "🎓 Select Grade",
            APP_CONFIG["grade_levels"],
            index=grade - 1,
            key="sidebar_grade",
        )
        if new_grade != grade:
            student_db.update_student_field(username, "grade_level", new_grade)
            st.session_state.student_profile["grade_level"] = new_grade
            st.rerun()

        # Subject selector
        current_subject = st.session_state.get("current_subject", "Phonics")
        new_subject = st.selectbox(
            "📖 Select Subject",
            APP_CONFIG["subjects"],
            index=APP_CONFIG["subjects"].index(current_subject),
            key="sidebar_subject",
        )
        if new_subject != current_subject:
            st.session_state.current_subject = new_subject
            student_db.update_student_field(username, "current_subject", new_subject)
            st.rerun()

        st.divider()

        # Progress overview
        stats = student_db.get_stats(username)
        st.subheader("📊 Progress")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Lessons", stats.get("total_lessons", 0))
            st.metric("Streak 🔥", f"{stats.get('streak_days', 0)} days")
        with col2:
            st.metric("Accuracy", f"{stats.get('accuracy_pct', 0):.0f}%")
            st.metric("Quizzes", stats.get("total_quizzes", 0))

        # Accuracy progress bar
        accuracy = stats.get("accuracy_pct", 0)
        st.progress(min(accuracy / 100, 1.0), text=f"Accuracy: {accuracy:.0f}%")

        # Badges
        badges = stats.get("badges", [])
        if badges:
            st.divider()
            st.subheader("🏅 Badges")
            for badge in badges:
                st.markdown(f"• {badge}")

        st.divider()

        # RAG / setup status
        rag_status = ai_teacher.get_rag_status()
        if not rag_status["available"]:
            st.warning("⚠️ Curriculum PDFs not loaded.\nAdd PDFs to `curriculum/` and restart.")
        else:
            st.success("✅ Curriculum loaded")

        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ─────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────

def _render_main_area() -> None:
    """Render the main chat and learning area."""
    subject = st.session_state.get("current_subject", "Phonics")
    username = st.session_state.student_profile.get("username", "Student")
    grade = st.session_state.student_profile.get("grade_level", 1)

    # Current topic
    stats = student_db.get_stats(username)
    current_topic = adaptive_path.get_current_topic(
        subject, stats.get("current_lesson_index", 0)
    )

    # Header
    st.title(f"{APP_CONFIG['icon']} {APP_CONFIG['title']}")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption(f"📖 Subject: **{subject}** | 🎯 Topic: **{current_topic}**")
    with col2:
        if st.button("📊 Show Visual"):
            st.session_state.show_visual = True
    with col3:
        if st.button("🔄 New Topic", help="Move to the next topic"):
            _advance_topic(username, subject, stats)

    st.divider()

    # Tabs
    tab_learn, tab_quiz, tab_progress = st.tabs(["💬 Learn", "📝 Quiz", "📈 Progress"])

    with tab_learn:
        _render_chat_tab(username, grade, subject, current_topic, stats)

    with tab_quiz:
        _render_quiz_tab(username, grade, subject, current_topic, stats)

    with tab_progress:
        _render_progress_tab(username, subject, stats)


def _render_chat_tab(
    username: str,
    grade: int,
    subject: str,
    current_topic: str,
    stats: dict,
) -> None:
    """Render the main chat / explanation interface."""
    # Visual aid panel (collapsible)
    if st.session_state.get("show_visual"):
        with st.expander("🎨 Visual Aid", expanded=True):
            _render_visual(subject, current_topic)
        if st.button("✕ Hide Visual"):
            st.session_state.show_visual = False
            st.rerun()

    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            avatar = AVATAR_TEACHER if role == "assistant" else AVATAR_STUDENT
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])
                if role == "assistant":
                    _render_tts_button(msg["content"], key=f"tts_{msg.get('id', id(msg))}")

    # Initial greeting if no history
    if not st.session_state.chat_history:
        greeting = (
            f"👋 Hello, **{username}**! I'm your English Tutor.\n\n"
            f"Today we're working on **{subject} — {current_topic}** for Grade {grade}.\n\n"
            f"You can:\n"
            f"- 📖 Ask me to **explain** anything (e.g., 'What is a consonant blend?')\n"
            f"- 📝 Say **'quiz me'** to test your knowledge\n"
            f"- 🔍 Say **'review'** to go over what you've learned\n"
            f"- 📊 Say **'show alphabet chart'** for visual help\n\n"
            f"What would you like to learn today? 😊"
        )
        with st.chat_message("assistant", avatar=AVATAR_TEACHER):
            st.markdown(greeting)
            _render_tts_button(greeting, key="tts_greeting")

    # Chat input
    user_input = st.chat_input(
        f"Ask about {subject}...",
        key="chat_input",
    )

    if user_input:
        # Display student message
        with st.chat_message("user", avatar=AVATAR_STUDENT):
            st.markdown(user_input)

        # Add to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "id": len(st.session_state.chat_history),
        })

        # Get AI response
        with st.chat_message("assistant", avatar=AVATAR_TEACHER):
            with st.spinner("Thinking... 🤔"):
                response = learning_orchestrator.process_student_input(
                    student_input=user_input,
                    username=username,
                    subject=subject,
                    grade=grade,
                    current_topic=current_topic,
                    pending_quiz=st.session_state.get("pending_quiz"),
                    session_history=st.session_state.chat_history,
                )

            msg_text = response["message"]
            st.markdown(msg_text)
            _render_tts_button(msg_text, key=f"tts_{len(st.session_state.chat_history)}")

            # Show performance tip if available
            perf = response.get("performance", {})
            if perf.get("recommendation") and perf.get("rolling_accuracy") is not None:
                st.info(perf["recommendation"])

            # Show visual if suggested
            if response.get("visual_type"):
                st.session_state.suggested_visual = response["visual_type"]

        # Add AI response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": msg_text,
            "id": len(st.session_state.chat_history),
        })

        # Handle quiz questions
        if response.get("quiz_questions"):
            st.session_state.pending_quiz_list = response["quiz_questions"]
            st.session_state.quiz_index = 0
            st.rerun()

        st.rerun()


def _render_quiz_tab(
    username: str,
    grade: int,
    subject: str,
    current_topic: str,
    stats: dict,
) -> None:
    """Render the interactive quiz interface."""
    st.subheader(f"📝 Quiz: {subject} — {current_topic}")

    # Generate quiz button
    if not st.session_state.get("pending_quiz_list"):
        st.markdown(
            f"Test your knowledge of **{current_topic}**! "
            f"I'll ask you {3} questions."
        )
        if st.button("🚀 Start Quiz!", use_container_width=True):
            with st.spinner("Generating quiz questions... 📝"):
                from config import APP_CONFIG
                import human_engine, ai_teacher
                rag_ctx = ai_teacher.retrieve_context(
                    f"{subject} {current_topic} quiz", subject=subject
                )
                import mistake_analyzer as ma
                history = student_db.get_mistake_history(username)
                weak = ma.get_repeated_weak_areas(history)
                prompt = human_engine.build_quiz_prompt(
                    subject=subject, grade=grade,
                    topic=current_topic,
                    difficulty=stats.get("difficulty_level", "Beginner"),
                    rag_context=rag_ctx, weak_areas=weak, num_questions=3,
                )
                raw = ai_teacher.call_llm(prompt, mode="quiz")
                questions = learning_orchestrator.parse_quiz_json(raw)

            if questions:
                st.session_state.pending_quiz_list = questions
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.rerun()
            else:
                st.error("Could not generate quiz. Please check your API key.")
        return

    # Show current question
    questions = st.session_state.pending_quiz_list
    idx = st.session_state.get("quiz_index", 0)

    if idx >= len(questions):
        # Quiz complete
        score = st.session_state.get("quiz_score", 0)
        total = len(questions)
        pct = int(score / total * 100)

        if pct >= 80:
            st.balloons()
            st.success(f"🎉 Excellent! You scored **{score}/{total}** ({pct}%)!")
        elif pct >= 50:
            st.info(f"👍 Good effort! You scored **{score}/{total}** ({pct}%). Keep practising!")
        else:
            st.warning(f"💪 You scored **{score}/{total}** ({pct}%). Let's review and try again!")

        if st.button("🔄 Try Another Quiz", use_container_width=True):
            del st.session_state["pending_quiz_list"]
            del st.session_state["quiz_index"]
            del st.session_state["quiz_score"]
            st.rerun()
        return

    q = questions[idx]
    st.markdown(f"**Question {idx + 1} of {len(questions)}**")
    st.progress((idx + 1) / len(questions))
    st.markdown(f"### {q.get('question', '')}")

    # Show hint if available
    if q.get("hint"):
        with st.expander("💡 Need a hint?"):
            st.info(q["hint"])

    q_type = q.get("type", "multiple_choice")
    answer_submitted = False
    student_answer = ""

    if q_type == "multiple_choice":
        options = q.get("options", [])
        if options:
            choice = st.radio("Choose your answer:", options, key=f"q_{idx}", index=None)
            if st.button("✅ Submit Answer", key=f"submit_{idx}", disabled=choice is None):
                student_answer = choice or ""
                answer_submitted = True

    else:  # fill_blank
        typed = st.text_input("Type your answer:", key=f"q_input_{idx}")
        if st.button("✅ Submit Answer", key=f"submit_{idx}", disabled=not typed.strip()):
            student_answer = typed.strip()
            answer_submitted = True

    if answer_submitted and student_answer:
        correct = q.get("correct_answer", "")
        import mistake_analyzer as ma
        analysis = ma.analyze_answer(student_answer, correct, subject)

        student_db.update_progress(username, {"correct": analysis["is_correct"]})

        if analysis["is_correct"]:
            st.success(f"✅ Correct! {q.get('explanation', '')}")
            st.session_state.quiz_score = st.session_state.get("quiz_score", 0) + 1
        else:
            st.error(f"❌ {analysis['explanation']}")
            student_db.add_mistake(username, {
                "word": correct,
                "correct_answer": correct,
                "student_answer": student_answer,
                "type": analysis["type"],
            })

        if st.button("➡️ Next Question", key=f"next_{idx}"):
            st.session_state.quiz_index = idx + 1
            st.rerun()


def _render_progress_tab(username: str, subject: str, stats: dict) -> None:
    """Render the progress dashboard."""
    st.subheader("📈 Your Progress")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏆 Accuracy", f"{stats.get('accuracy_pct', 0):.0f}%")
    with col2:
        st.metric("📚 Lessons", stats.get("total_lessons", 0))
    with col3:
        st.metric("✅ Correct", stats.get("correct_answers", 0))
    with col4:
        st.metric("🔥 Streak", f"{stats.get('streak_days', 0)} days")

    st.divider()

    # Performance evaluation
    perf = adaptive_path.evaluate_performance(stats)
    if perf.get("recommendation"):
        st.info(perf["recommendation"])

    # Topic mastery chart
    st.subheader(f"🗺️ {subject} Topic Mastery")
    mistake_history = student_db.get_mistake_history(username)
    mastery = adaptive_path.calculate_topic_mastery(mistake_history, subject)
    fig = visual_teacher.create_mastery_bar_chart(mastery, subject)
    st.pyplot(fig)

    # Badges
    badges = stats.get("badges", [])
    if badges:
        st.divider()
        st.subheader("🏅 Your Badges")
        cols = st.columns(min(len(badges), 4))
        for i, badge in enumerate(badges):
            with cols[i % 4]:
                st.markdown(
                    f"<div style='background:#4A90D9;padding:10px;border-radius:8px;"
                    f"text-align:center;color:white;margin:4px'>{badge}</div>",
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────
# Visual Aid Panel
# ─────────────────────────────────────────────

def _render_visual(subject: str, topic: str) -> None:
    """Render the appropriate visual aid."""
    visual_type = st.session_state.get("suggested_visual", "")
    topic_lower = topic.lower()

    options = {
        "Alphabet Chart": "alphabet",
        "Short & Long Vowels": "vowels",
        "Consonant Blends": "blends",
        "Digraphs": "digraphs",
        "Word Family (-at)": "word_family_at",
        "Word Family (-ig)": "word_family_ig",
        "Word Family (-ot)": "word_family_ot",
    }

    selected = st.selectbox("Choose a visual:", list(options.keys()), key="visual_select")
    vtype = options.get(selected, "alphabet")

    with st.spinner("Generating visual... 🎨"):
        if vtype == "alphabet":
            fig = visual_teacher.create_alphabet_chart()
        elif vtype == "vowels":
            fig = visual_teacher.create_phonics_sound_chart("vowels")
        elif vtype == "blends":
            fig = visual_teacher.create_phonics_sound_chart("consonant_blends")
        elif vtype == "digraphs":
            fig = visual_teacher.create_phonics_sound_chart("digraphs")
        elif vtype == "word_family_at":
            fig = visual_teacher.create_word_family_diagram("-at")
        elif vtype == "word_family_ig":
            fig = visual_teacher.create_word_family_diagram("-ig")
        elif vtype == "word_family_ot":
            fig = visual_teacher.create_word_family_diagram("-ot")
        else:
            fig = visual_teacher.create_alphabet_chart()

    st.pyplot(fig)


# ─────────────────────────────────────────────
# TTS (Text-to-Speech)
# ─────────────────────────────────────────────

def _render_tts_button(text: str, key: str) -> None:
    """Render a 🔊 Read Aloud button that plays the given text."""
    if st.button("🔊 Read Aloud", key=f"tts_btn_{key}", help="Listen to this response"):
        with st.spinner("Generating audio... 🔊"):
            audio_path = _generate_tts(text)
        if audio_path:
            with open(audio_path, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
        else:
            st.warning("TTS not available. Check your configuration.")


def _generate_tts(text: str) -> Optional[str]:
    """
    Generate TTS audio and return path to the audio file.

    Tries the configured provider (edge-tts or azure), returns None on failure.
    """
    provider = TTS_CONFIG.get("provider", "edge")
    clean_text = _strip_markdown(text)[:800]  # Limit length for TTS

    if provider == "azure" and TTS_CONFIG.get("azure_key"):
        return _azure_tts(clean_text)
    else:
        return _edge_tts(clean_text)


def _edge_tts(text: str) -> Optional[str]:
    """Generate TTS using edge-tts (free, no API key needed)."""
    try:
        import edge_tts

        voice = TTS_CONFIG.get("edge_voice", "en-US-AriaNeural")
        tmp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        async def _run():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(tmp_path)

        asyncio.run(_run())
        return tmp_path
    except Exception as exc:
        logger.warning("edge-tts error: %s", exc)
        return None


def _azure_tts(text: str) -> Optional[str]:
    """Generate TTS using Azure Cognitive Services Speech SDK."""
    try:
        import azure.cognitiveservices.speech as speechsdk

        key = TTS_CONFIG["azure_key"]
        region = TTS_CONFIG["azure_region"]
        voice = TTS_CONFIG.get("azure_voice", "en-US-JennyNeural")

        config = speechsdk.SpeechConfig(subscription=key, region=region)
        config.speech_synthesis_voice_name = voice

        tmp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        audio_config = speechsdk.audio.AudioOutputConfig(filename=tmp_path)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=config, audio_config=audio_config
        )
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return tmp_path
        else:
            logger.warning("Azure TTS failed: %s", result.reason)
            return None
    except Exception as exc:
        logger.warning("Azure TTS error: %s", exc)
        return None


def _strip_markdown(text: str) -> str:
    """Remove common Markdown formatting for clean TTS output."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    return text.strip()


# ─────────────────────────────────────────────
# Topic navigation
# ─────────────────────────────────────────────

def _advance_topic(username: str, subject: str, stats: dict) -> None:
    """Move the student forward to the next topic."""
    current_idx = stats.get("current_lesson_index", 0)
    next_idx = current_idx + 1
    student_db.update_student_field(username, "current_lesson_index", next_idx)
    st.session_state.student_profile = student_db.get_or_create_student(username)
    st.success("✅ Moving to next topic!")
    st.rerun()
