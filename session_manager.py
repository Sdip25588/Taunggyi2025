"""
session_manager.py — Conversation state machine for voice-first sessions.

Manages the multi-turn greeting → check-in → lesson → wrap-up flow and
exports the helpers used by gui_engine.py.

Public API
----------
  ConversationState         — state-name constants
  get_initial_greeting()    — opening message for a new session
  get_wrapup_message()      — closing message that promotes reflection
  handle_student_utterance() — advance the state machine for one utterance

All imports used at module level are declared here (no hidden function-level
imports), making dependencies visible and easier to mock in tests.
"""

import logging
from typing import Optional

import ai_teacher
import student as student_db
from intent_classifier import determine_intent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Conversation State Machine
# ─────────────────────────────────────────────

class ConversationState:
    """Enumeration of dialog stages for the voice-first conversational loop."""
    GREETING = "GREETING"
    CHECKIN  = "CHECKIN"
    # NOTE: INTENT state was previously defined but is never used.
    # Subject/intent selection is handled inline in the CHECKIN → LESSON
    # transition, so no separate INTENT state is needed.
    LESSON   = "LESSON"
    WRAPUP   = "WRAPUP"


# ─────────────────────────────────────────────
# Mood & subject detection helpers
# ─────────────────────────────────────────────

_NEGATIVE_MOOD_WORDS: frozenset = frozenset({
    "sad", "tired", "bad", "not good", "awful", "terrible", "sick", "upset",
    "bored", "confused", "frustrated", "stressed", "unhappy", "worried", "scared",
    "not great", "not well", "okay i guess", "so so",
})

_POSITIVE_MOOD_WORDS: frozenset = frozenset({
    "good", "great", "fine", "okay", "well", "happy", "wonderful", "amazing",
    "fantastic", "excellent", "super", "brilliant", "ready", "excited",
})

_PHONICS_WORDS: frozenset = frozenset({"phonics", "sounds", "letters", "alphabet"})
_READING_WORDS: frozenset = frozenset({"reading", "read", "story", "text", "passage"})
_SPELLING_WORDS: frozenset = frozenset({"spelling", "spell", "words", "word"})
_CONTINUE_WORDS: frozenset = frozenset({
    "continue", "same", "last time", "where we left", "again", "that",
})


def _detect_mood(text: str) -> str:
    """Detect student mood from utterance.

    Returns:
        "positive", "negative", or "neutral".
    """
    lower = text.lower()
    if any(word in lower for word in _NEGATIVE_MOOD_WORDS):
        return "negative"
    if any(word in lower for word in _POSITIVE_MOOD_WORDS):
        return "positive"
    return "neutral"


def _detect_subject_intent(text: str, current_subject: str) -> Optional[str]:
    """
    Detect if the student is selecting a subject or asking to continue.

    Returns:
        "Phonics", "Reading", "Spelling", "continue", or ``None``.
    """
    lower = text.lower()
    if any(w in lower for w in _CONTINUE_WORDS):
        return "continue"
    if any(w in lower for w in _PHONICS_WORDS):
        return "Phonics"
    if any(w in lower for w in _READING_WORDS):
        return "Reading"
    if any(w in lower for w in _SPELLING_WORDS):
        return "Spelling"
    return None


# ─────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────

def get_initial_greeting(name: str) -> str:
    """Return the opening greeting message for a new session."""
    return f"Hello {name}! How are you today? 😊"


def get_wrapup_message(username: str) -> str:
    """Return a warm wrap-up message that promotes reflection and independence."""
    return (
        f"You did an amazing job today, {username}! 🌟🎉 "
        f"I'm really proud of your effort and independent thinking. "
        f"Before we finish — **what did you figure out by yourself today?** "
        f"(Take a moment to think! 🤔) "
        f"Would you like to try one more challenge, or shall we finish here?"
    )


def _get_state_resume_prompt(
    state: str,
    username: str,
    subject: str,
    current_topic: str,
    stats: dict,
) -> str:
    """Return the re-engagement prompt to resume after answering an inline question."""
    if state == ConversationState.CHECKIN:
        return (
            f"Now, what would you like to practise today — "
            f"**Phonics**, **Reading**, or **Spelling**? "
            f"Or shall we continue with **{current_topic}**?"
        )
    return f"Shall we get back to our lesson on **{current_topic}**? 😊"


def _quick_answer(
    utterance: str,
    subject: str,
    grade: int,
    current_topic: str,
) -> str:
    """
    Handle a quick inline question using Socratic guidance.

    Guides the student to think for themselves before offering a brief hint,
    rather than giving a direct answer.
    """
    prompt = (
        f"A Grade {grade} student asked a question during a conversation: '{utterance}'. "
        f"IMPORTANT: Do NOT give the full answer immediately. "
        f"First ask them one Socratic question to guide their thinking "
        f"(e.g., 'What do you already know about this?', "
        f"'Can you think of a similar word?'). "
        f"Then, if they still need help, offer a brief 1-2 sentence child-friendly hint. "
        f"Do not use bullet points. Keep it warm and conversational."
    )
    try:
        return ai_teacher.call_llm(prompt, mode="explain", subject=subject)
    except Exception:
        return (
            "That's a great question! What do you already know about it? "
            "Let's think together! 🤔"
        )


# ─────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────

def handle_student_utterance(
    state: str,
    utterance: str,
    username: str,
    subject: str,
    grade: int,
    current_topic: str,
    session_history: Optional[list] = None,
) -> dict:
    """
    Process a student utterance within the conversation state machine.

    Handles direct questions at any state (answers them, then resumes state).
    If a mood of confusion/negativity is detected during CHECKIN, responds
    supportively before moving on.

    When in LESSON state this function returns ``tutor_text=""`` intentionally.
    The caller (gui_engine) must then route the utterance through
    ``learning_orchestrator.process_student_input()`` to generate the full
    lesson response.  This split is documented here so it is not accidentally
    bypassed.

    Args:
        state:           Current ConversationState value.
        utterance:       What the student said or typed.
        username:        Student's name.
        subject:         Current subject.
        grade:           Grade level (1-5).
        current_topic:   Current lesson topic.
        session_history: Previous messages for context.

    Returns:
        Dict with keys:
          - "new_state":     str — next ConversationState
          - "tutor_text":    str — what the tutor says next (shown in chat + TTS);
                             empty string when new_state is LESSON (see note above)
          - "lesson_intent": str or None — if LESSON started, the intent to pass on
          - "subject_switch": str or None — if subject changed
    """
    profile = student_db.get_or_create_student(username)
    stats   = student_db.get_stats(username)

    detected = determine_intent(utterance)

    # ── Inline direct-question answering (CHECKIN only) ─────────────────────
    # During GREETING we want the mood/check-in flow uninterrupted.
    # During LESSON, process_student_input handles all intents directly.
    direct_question_intents = {"lesson", "pronunciation", "vocabulary", "hint"}
    if state == ConversationState.CHECKIN and detected in direct_question_intents:
        answer = _quick_answer(utterance, subject, grade, current_topic)
        resume_prompt = _get_state_resume_prompt(
            state, username, subject, current_topic, stats
        )
        return {
            "new_state":     state,
            "tutor_text":    f"{answer}\n\n{resume_prompt}",
            "lesson_intent": None,
            "subject_switch": None,
        }

    # ── GREETING ─────────────────────────────────────────────────────────────
    if state == ConversationState.GREETING:
        mood = _detect_mood(utterance)
        if mood == "negative":
            tutor_text = (
                "I'm sorry to hear that! It's okay — learning can actually cheer us up. 💙 "
                "I'm here with you. What would you like to practise today? "
                "We have Phonics, Reading, or Spelling!"
            )
        elif mood == "positive":
            tutor_text = (
                f"That's wonderful to hear! 🌟 "
                f"I'm glad you're feeling good today! "
                f"What would you like to practise? We can do Phonics, Reading, or Spelling. "
                f"Or shall we continue with **{current_topic}**?"
            )
        else:
            tutor_text = (
                f"Thank you for sharing! 😊 "
                f"What would you like to practise today? "
                f"We have **Phonics**, **Reading**, or **Spelling**. "
                f"Or shall we pick up where we left off with **{current_topic}**?"
            )
        return {
            "new_state":     ConversationState.CHECKIN,
            "tutor_text":    tutor_text,
            "lesson_intent": None,
            "subject_switch": None,
        }

    # ── CHECKIN ───────────────────────────────────────────────────────────────
    if state == ConversationState.CHECKIN:
        chosen = _detect_subject_intent(utterance, subject)
        if chosen == "continue" or chosen is None:
            next_subject = subject
            tutor_text = (
                f"Great! Let's continue with **{subject} — {current_topic}**. 📚\n\n"
                f"I'll start the lesson now. Feel free to ask questions at any time!"
            )
        else:
            next_subject = chosen
            tutor_text = (
                f"Excellent choice! Let's work on **{next_subject}** today. 🎯\n\n"
                f"I'll start the lesson now. Feel free to ask questions anytime!"
            )
        return {
            "new_state":     ConversationState.LESSON,
            "tutor_text":    tutor_text,
            "lesson_intent": "lesson",
            "subject_switch": next_subject if next_subject != subject else None,
        }

    # ── LESSON ────────────────────────────────────────────────────────────────
    # Return empty tutor_text — the GUI routes this through process_student_input.
    if state == ConversationState.LESSON:
        return {
            "new_state":     ConversationState.LESSON,
            "tutor_text":    "",
            "lesson_intent": detected,
            "subject_switch": None,
        }

    # ── WRAPUP ────────────────────────────────────────────────────────────────
    if state == ConversationState.WRAPUP:
        lower = utterance.lower()
        if any(w in lower for w in {"yes", "sure", "okay", "ok", "more", "continue", "another"}):
            return {
                "new_state":     ConversationState.LESSON,
                "tutor_text":    (
                    "Wonderful! Let's keep going! 🚀 "
                    "What would you like to do — quiz, lesson, or something else?"
                ),
                "lesson_intent": "lesson",
                "subject_switch": None,
            }
        return {
            "new_state":     ConversationState.WRAPUP,
            "tutor_text":    (
                f"Fantastic work today, {username}! 🌟 "
                f"Come back whenever you're ready to learn more. See you next time! 👋"
            ),
            "lesson_intent": None,
            "subject_switch": None,
        }

    # ── Default fallback — treat as LESSON ───────────────────────────────────
    return {
        "new_state":     ConversationState.LESSON,
        "tutor_text":    "",
        "lesson_intent": detected,
        "subject_switch": None,
    }
