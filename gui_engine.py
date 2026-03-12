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
from learning_orchestrator import ConversationState, handle_student_utterance
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

        st.divider()

        # Conversation mode toggle
        st.subheader("🗣️ Conversation Mode")
        conv_mode = st.session_state.get("conversation_mode", True)
        new_conv_mode = st.toggle(
            "Voice Conversation",
            value=conv_mode,
            key="conv_mode_toggle",
            help="Turn on for a natural voice-first tutoring experience",
        )
        if new_conv_mode != conv_mode:
            st.session_state.conversation_mode = new_conv_mode
            st.rerun()

        if st.session_state.get("conversation_mode"):
            conv_state = st.session_state.get("conversation_state", ConversationState.GREETING)
            st.caption(f"💬 Stage: **{conv_state}**")
            if st.button("🔁 Restart Conversation", use_container_width=True, key="restart_conv"):
                from learning_orchestrator import get_initial_greeting
                username_r = st.session_state.student_profile.get("username", "Student")
                greeting = get_initial_greeting(username_r)
                st.session_state.chat_history = [{
                    "role": "assistant",
                    "content": greeting,
                    "id": 0,
                }]
                st.session_state.conversation_state = ConversationState.GREETING
                st.session_state.awaiting_student_reply = True
                st.session_state.pending_tutor_prompt = greeting
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
    """Render the main chat / explanation interface with voice input support."""
    # Visual aid panel (collapsible)
    if st.session_state.get("show_visual"):
        with st.expander("🎨 Visual Aid", expanded=True):
            _render_visual(subject, current_topic)
        if st.button("✕ Hide Visual"):
            st.session_state.show_visual = False
            st.rerun()

    # Conversation mode: show prominent tutor prompt card when not in LESSON stage
    conv_mode = st.session_state.get("conversation_mode", True)
    conv_state = st.session_state.get("conversation_state", ConversationState.LESSON)

    if conv_mode and conv_state != ConversationState.LESSON:
        _render_conversation_prompt_card(username, grade, subject, current_topic, stats)
        return  # Conversation card handles input; skip the regular chat below

    # Voice input panel (collapsible) — always available as a fallback
    with st.expander("🎤 Voice Input (speak your answer)", expanded=False):
        _render_voice_input_panel(username, grade, subject, current_topic, stats)

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

    # Initial greeting if no history (non-conversation mode fallback)
    if not st.session_state.chat_history:
        greeting = (
            f"👋 Hello, **{username}**! I'm your English Tutor.\n\n"
            f"Today we're working on **{subject} — {current_topic}** for Grade {grade}.\n\n"
            f"You can:\n"
            f"- 📖 Ask me to **explain** anything (e.g., 'What is a consonant blend?')\n"
            f"- 📝 Say **'quiz me'** to test your knowledge\n"
            f"- 🔍 Say **'review'** to go over what you've learned\n"
            f"- 📊 Say **'show alphabet chart'** for visual help\n"
            f"- 🗣️ Say **'read aloud'** to practice pronunciation\n"
            f"- 📖 Say **'vocabulary'** to learn new words\n"
            f"- ✍️ Say **'write'** to practice writing\n\n"
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

            # Show confusion strategy suggestion
            personalization = response.get("personalization", {})
            confusion = personalization.get("confusion", {})
            if confusion.get("is_confused") and confusion.get("suggestion"):
                st.warning(f"💡 {confusion['suggestion']}")

            # Show grade advancement notification
            grade_adv = response.get("grade_advancement")
            if grade_adv and grade_adv.get("should_advance"):
                st.success(f"🎉 {grade_adv['message']}")

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


def _render_conversation_prompt_card(
    username: str,
    grade: int,
    subject: str,
    current_topic: str,
    stats: dict,
) -> None:
    """
    Render the conversation mode card: shows the latest tutor prompt prominently,
    a voice recorder, and a text fallback. Used when conversation_state != LESSON.
    """
    # ── Tutor prompt banner ──
    pending_prompt = st.session_state.get("pending_tutor_prompt", "")
    if pending_prompt:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#4A90D9,#6AAFE6);
                        padding:20px 24px;border-radius:14px;
                        color:white;font-size:1.15rem;margin-bottom:16px;">
              🧑‍🏫 <strong>Tutor says:</strong><br><br>
              {pending_prompt}
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Auto-play TTS for the tutor prompt
        _render_tts_button(pending_prompt, key="tts_conv_prompt")

    st.markdown("---")
    st.markdown("#### 🎤 Answer by speaking (or typing below):")

    # ── Voice recorder ──
    voice_text = ""
    try:
        from audio_recorder_streamlit import audio_recorder
        audio_bytes = audio_recorder(
            text="Click to speak",
            recording_color="#4A90D9",
            neutral_color="#6B6B6B",
            icon_name="microphone",
            icon_size="2x",
            key="conv_voice_recorder",
        )
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            if st.button("📨 Use Voice Answer", key="conv_submit_voice"):
                with st.spinner("Converting speech to text... 🎤"):
                    from voice_engine import listen_to_student
                    stt = listen_to_student(audio_bytes)
                if stt["success"]:
                    voice_text = stt["text"]
                    st.success(f"✅ I heard: **\"{voice_text}\"**")
                else:
                    err = stt.get("error", "")
                    if "not configured" in err.lower() or "not installed" in err.lower():
                        st.info("🔤 Azure STT not configured — please type your answer below.")
                    else:
                        st.warning(f"Could not understand audio: {err}. Please type below.")
    except ImportError:
        st.info("🎤 Audio recorder not installed. Please type your answer below.")

    # ── Text fallback always available ──
    col_input, col_send = st.columns([4, 1])
    with col_input:
        typed_input = st.text_input(
            "Type your answer here:",
            key="conv_text_input",
            label_visibility="collapsed",
            placeholder="Type your reply...",
        )
    with col_send:
        send_clicked = st.button("Send ➤", key="conv_send_btn", use_container_width=True)

    # ── Previous conversation history ──
    if st.session_state.chat_history:
        with st.expander("📜 Conversation so far", expanded=False):
            for msg in st.session_state.chat_history:
                role = msg["role"]
                avatar = AVATAR_TEACHER if role == "assistant" else AVATAR_STUDENT
                with st.chat_message(role, avatar=avatar):
                    st.markdown(msg["content"])

    # ── Process student response ──
    student_reply = voice_text or (typed_input.strip() if send_clicked and typed_input.strip() else "")
    if not student_reply:
        return

    # Add student message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": student_reply,
        "id": len(st.session_state.chat_history),
    })

    current_state = st.session_state.get("conversation_state", ConversationState.GREETING)

    with st.spinner("Thinking... 🤔"):
        conv_result = handle_student_utterance(
            state=current_state,
            utterance=student_reply,
            username=username,
            subject=subject,
            grade=grade,
            current_topic=current_topic,
            session_history=st.session_state.chat_history,
        )

    new_state = conv_result["new_state"]
    tutor_text = conv_result["tutor_text"]
    subject_switch = conv_result.get("subject_switch")

    # Apply subject switch if needed
    if subject_switch and subject_switch != subject:
        st.session_state.current_subject = subject_switch
        student_db.update_student_field(username, "current_subject", subject_switch)
        subject = subject_switch

    # If transitioning to LESSON, use process_student_input for the opening
    if new_state == ConversationState.LESSON and tutor_text:
        # Add the transition message to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": tutor_text,
            "id": len(st.session_state.chat_history),
        })
        st.session_state.conversation_state = ConversationState.LESSON
        st.session_state.pending_tutor_prompt = ""
        st.session_state.awaiting_student_reply = False
        st.rerun()
        return

    if tutor_text:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": tutor_text,
            "id": len(st.session_state.chat_history),
        })
        st.session_state.pending_tutor_prompt = tutor_text

    st.session_state.conversation_state = new_state
    st.session_state.awaiting_student_reply = True
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
    """Render the enhanced progress dashboard with mastery, weekly progress, and grade readiness."""
    st.subheader("📈 Your Learning Progress")

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

    # Performance recommendation
    perf = adaptive_path.evaluate_performance(stats)
    if perf.get("recommendation"):
        st.info(perf["recommendation"])

    # Grade readiness gauge
    grade_readiness = student_db.check_grade_readiness(username)
    mastery_pct = grade_readiness.get("mastery_pct", 0.0)
    current_grade = grade_readiness.get("current_grade", stats.get("grade_level", 1))

    st.subheader("🎓 Grade Readiness")
    fig_gauge = visual_teacher.generate_grade_readiness_gauge(mastery_pct, current_grade)
    st.pyplot(fig_gauge)
    st.caption(grade_readiness.get("message", ""))

    if grade_readiness.get("ready_to_advance"):
        next_grade = grade_readiness.get("next_grade", current_grade + 1)
        if st.button(f"🚀 Advance to Grade {next_grade}!", use_container_width=True):
            student_db.update_student_field(username, "grade_level", next_grade)
            st.session_state.student_profile["grade_level"] = next_grade
            st.success(f"🎉 Congratulations! You've moved to Grade {next_grade}!")
            st.rerun()

    st.divider()

    # Topic mastery chart
    st.subheader(f"🗺️ {subject} Topic Mastery")
    topic_mastery_db = student_db.get_topic_mastery(username)
    if topic_mastery_db:
        fig_mastery = visual_teacher.generate_topic_mastery_chart(topic_mastery_db, subject)
    else:
        mistake_history = student_db.get_mistake_history(username)
        mastery_raw = adaptive_path.calculate_topic_mastery(mistake_history, subject)
        # Convert to 0-1 scale for consistency
        mastery_normalized = {k: v / 100 for k, v in mastery_raw.items()}
        fig_mastery = visual_teacher.generate_topic_mastery_chart(mastery_normalized, subject)
    st.pyplot(fig_mastery)

    # Weak area recommendations
    topic_mastery_data = student_db.get_topic_mastery(username) or {}
    weak_topics = sorted(
        [(t, v) for t, v in topic_mastery_data.items() if v < 0.60],
        key=lambda x: x[1],
    )[:3]

    if weak_topics:
        st.divider()
        st.subheader("🎯 Recommended for You")
        for topic, mastery_val in weak_topics:
            pct = round(mastery_val * 100)
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**{topic}** — {pct}% mastery")
                st.progress(mastery_val, text=f"{pct}%")
            with col_b:
                if st.button(f"Practice", key=f"practice_{topic}"):
                    st.session_state.practice_topic = topic
                    st.info(f"Starting practice session for **{topic}**! Go to the Learn tab. 💪")

    st.divider()

    # Weekly progress chart
    st.subheader("📊 Weekly Progress")
    weekly_data = student_db.get_weekly_progress(username)
    if weekly_data:
        fig_weekly = visual_teacher.generate_weekly_progress_chart(
            weekly_data,
            student_name=username,
        )
        st.pyplot(fig_weekly)
    else:
        st.info("📅 Complete a few sessions to see your weekly progress chart!")
        # Show a sample with existing data
        if stats.get("total_quizzes", 0) > 0:
            sample_data = [{"date": "This session", "accuracy": stats.get("accuracy_pct", 0)}]
            fig_weekly = visual_teacher.generate_weekly_progress_chart(sample_data, username)
            st.pyplot(fig_weekly)

    # Vocabulary bank
    vocab_bank = student_db.get_vocabulary_bank(username)
    if vocab_bank:
        st.divider()
        st.subheader("📖 Your Vocabulary Bank")
        st.caption(f"You've learned {len(vocab_bank)} words!")
        fig_vocab = visual_teacher.generate_vocabulary_word_cloud(vocab_bank)
        st.pyplot(fig_vocab)

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
# Voice Input Panel
# ─────────────────────────────────────────────

def _render_voice_input_panel(
    username: str,
    grade: int,
    subject: str,
    current_topic: str,
    stats: dict,
) -> None:
    """
    Render the voice input UI for speech-to-text.

    Tries audio_recorder_streamlit first; falls back to file upload.
    When Azure STT is not configured, gracefully shows text input only.
    """
    try:
        from audio_recorder_streamlit import audio_recorder

        st.markdown("🎤 **Record your voice** and I'll convert it to text:")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#4A90D9",
            neutral_color="#6B6B6B",
            icon_name="microphone",
            icon_size="2x",
            key="voice_recorder",
        )

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            if st.button("📨 Submit Voice Answer", key="submit_voice"):
                with st.spinner("Converting speech to text... 🎤"):
                    from voice_engine import listen_to_student
                    stt_result = listen_to_student(audio_bytes)

                if stt_result["success"]:
                    transcribed = stt_result["text"]
                    st.success(f"✅ I heard: **\"{transcribed}\"**")
                    # Inject into chat as if typed
                    st.session_state.voice_input_text = transcribed
                    st.info("Your voice has been converted! It will appear as your next message.")
                else:
                    err = stt_result.get("error", "Unknown error")
                    if "not configured" in err.lower() or "not installed" in err.lower():
                        st.info(
                            "🔤 **Voice-to-text not available** — "
                            "Azure Speech key not configured or SDK not installed. "
                            "Please type your answer below instead."
                        )
                    else:
                        st.warning(f"Could not understand audio: {err}. Please try again or type your answer.")

    except ImportError:
        # audio_recorder_streamlit not installed — offer file upload
        st.markdown("🎙️ **Upload an audio file** (WAV or MP3):")
        uploaded = st.file_uploader(
            "Upload your recorded audio",
            type=["wav", "mp3"],
            key="voice_upload",
        )
        if uploaded:
            audio_bytes = uploaded.read()
            # Determine format safely
            mime_type = uploaded.type or "audio/wav"
            audio_format = mime_type.split("/")[1] if "/" in mime_type else "wav"
            st.audio(audio_bytes, format=f"audio/{audio_format}")
            if st.button("📨 Submit Audio", key="submit_audio"):
                with st.spinner("Converting speech to text... 🎤"):
                    from voice_engine import listen_to_student
                    stt_result = listen_to_student(audio_bytes)
                if stt_result["success"]:
                    st.success(f"✅ I heard: **\"{stt_result['text']}\"**")
                    st.session_state.voice_input_text = stt_result["text"]
                else:
                    st.info(
                        "🔤 Voice recognition unavailable. "
                        "Please type your answer in the chat below."
                    )


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

    Uses voice_engine.speak_response which handles emoji stripping, SSML voice
    styles, and graceful fallback (edge-tts if Azure not configured).
    """
    try:
        from voice_engine import speak_response
        return speak_response(text)
    except Exception as exc:
        logger.warning("voice_engine.speak_response failed, falling back: %s", exc)

    # Fallback: direct TTS without voice_engine
    provider = TTS_CONFIG.get("provider", "edge")
    clean_text = _strip_markdown(text)[:800]

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
    """Remove common Markdown formatting and emojis for clean TTS output."""
    import re
    import unicodedata
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    # Strip emojis using unicodedata to avoid overly-broad regex ranges
    result = []
    for char in text:
        cp = ord(char)
        cat = unicodedata.category(char)
        if cp > 0x1F000:
            continue  # Skip SMP emoji characters
        if cat in ("So", "Sk") and cp > 0x2000:
            continue  # Skip other symbol emoji-like characters
        result.append(char)
    text = "".join(result)
    # Remove variation selectors
    text = re.sub(r"[\uFE00-\uFE0F]", "", text)
    return re.sub(r"\s+", " ", text).strip()


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
