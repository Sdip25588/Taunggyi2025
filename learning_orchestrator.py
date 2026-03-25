"""
learning_orchestrator.py — Central coordinator for learning sessions.

Receives student input, determines intent, orchestrates RAG retrieval,
LLM calls, mistake analysis, student profile updates, and visual generation.

Also manages the voice-first Conversation Mode state machine:
  GREETING → CHECKIN → LESSON → WRAPUP

  Note: The INTENT stage is folded into the CHECKIN → LESSON transition.
  When the student selects a subject/topic in CHECKIN, the state moves directly
  to LESSON (no separate INTENT state needed).
"""

import json
import logging
import re
from typing import Optional

from config import APP_CONFIG
import ai_teacher
import human_engine
import student as student_db
import mistake_analyzer
import adaptive_path
import visual_teacher
import personalization_engine

logger = logging.getLogger(__name__)

# Default score assigned to a writing sample for tracking purposes
_DEFAULT_WRITING_SCORE = 0.7

# ─────────────────────────────────────────────
# Conversation State Machine
# ─────────────────────────────────────────────

class ConversationState:
    """Enumeration of dialog stages for the voice-first conversational loop."""
    GREETING = "GREETING"
    CHECKIN = "CHECKIN"
    INTENT = "INTENT"
    LESSON = "LESSON"
    WRAPUP = "WRAPUP"


# Mood keywords for supportive detection during check-in
_NEGATIVE_MOOD_WORDS = {
    "sad", "tired", "bad", "not good", "awful", "terrible", "sick", "upset",
    "bored", "confused", "frustrated", "stressed", "unhappy", "worried", "scared",
    "not great", "not well", "okay i guess", "so so",
}
_POSITIVE_MOOD_WORDS = {
    "good", "great", "fine", "okay", "well", "happy", "wonderful", "amazing",
    "fantastic", "excellent", "super", "brilliant", "ready", "excited",
}

# Subject/intent selection keywords for the CHECKIN → INTENT transition
_PHONICS_WORDS = {"phonics", "sounds", "letters", "alphabet"}
_READING_WORDS = {"reading", "read", "story", "text", "passage"}
_SPELLING_WORDS = {"spelling", "spell", "words", "word"}
_CONTINUE_WORDS = {"continue", "same", "last time", "where we left", "again", "that"}


def get_initial_greeting(name: str) -> str:
    """Return the opening greeting message for a new session."""
    return f"Hello {name}! How are you today? 😊"


def _detect_mood(text: str) -> str:
    """Detect student mood from utterance. Returns 'positive', 'negative', or 'neutral'."""
    lower = text.lower()
    if any(word in lower for word in _NEGATIVE_MOOD_WORDS):
        return "negative"
    if any(word in lower for word in _POSITIVE_MOOD_WORDS):
        return "positive"
    return "neutral"


def _detect_subject_intent(text: str, current_subject: str) -> Optional[str]:
    """
    Detect if the student is selecting a subject or asking to continue.

    Returns subject name ("Phonics", "Reading", "Spelling"), "continue", or None.
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

    Args:
        state: Current ConversationState value.
        utterance: What the student said or typed.
        username: Student's name.
        subject: Current subject.
        grade: Grade level (1-5).
        current_topic: Current lesson topic.
        session_history: Previous messages for context.

    Returns:
        Dict with keys:
          - "new_state": str — next ConversationState
          - "tutor_text": str — what the tutor says next (shown in chat + TTS)
          - "lesson_intent": str or None — if LESSON started, the intent to pass on
          - "subject_switch": str or None — if subject changed
    """
    profile = student_db.get_or_create_student(username)
    stats = student_db.get_stats(username)

    detected = determine_intent(utterance)

    # ── Inline direct-question answering: only during CHECKIN (not GREETING or LESSON) ──
    # During GREETING we want the mood/check-in flow uninterrupted.
    # During LESSON process_student_input handles all intents directly.
    direct_question_intents = {"lesson", "pronunciation", "vocabulary", "hint"}
    if state == ConversationState.CHECKIN and detected in direct_question_intents:
        answer = _quick_answer(utterance, subject, grade, current_topic)
        resume_prompt = _get_state_resume_prompt(state, username, subject, current_topic, stats)
        return {
            "new_state": state,
            "tutor_text": f"{answer}\n\n{resume_prompt}",
            "lesson_intent": None,
            "subject_switch": None,
        }

    # ── State-specific handling ──
    if state == ConversationState.GREETING:
        mood = _detect_mood(utterance)
        if mood == "negative":
            tutor_text = (
                f"I'm sorry to hear that! It's okay — learning can actually cheer us up. 💙 "
                f"I'm here with you. What would you like to practice today? "
                f"We have Phonics, Reading, or Spelling!"
            )
        elif mood == "positive":
            tutor_text = (
                f"That's wonderful to hear! 🌟 "
                f"I'm glad you're feeling good today! "
                f"What would you like to practice? We can do Phonics, Reading, or Spelling. "
                f"Or shall we continue with **{current_topic}**?"
            )
        else:
            tutor_text = (
                f"Thank you for sharing! 😊 "
                f"What would you like to practice today? "
                f"We have **Phonics**, **Reading**, or **Spelling**. "
                f"Or shall we pick up where we left off with **{current_topic}**?"
            )
        return {
            "new_state": ConversationState.CHECKIN,
            "tutor_text": tutor_text,
            "lesson_intent": None,
            "subject_switch": None,
        }

    if state == ConversationState.CHECKIN:
        chosen = _detect_subject_intent(utterance, subject)
        if chosen == "continue" or chosen is None:
            # Continue with current subject
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
            "new_state": ConversationState.LESSON,
            "tutor_text": tutor_text,
            "lesson_intent": "lesson",
            "subject_switch": next_subject if next_subject != subject else None,
        }

    if state == ConversationState.LESSON:
        # In LESSON state, fall through to normal process_student_input
        return {
            "new_state": ConversationState.LESSON,
            "tutor_text": "",   # Will be filled by process_student_input
            "lesson_intent": detected,
            "subject_switch": None,
        }

    if state == ConversationState.WRAPUP:
        # Student responded to wrap-up — either continue or end
        lower = utterance.lower()
        if any(w in lower for w in {"yes", "sure", "okay", "ok", "more", "continue", "another"}):
            return {
                "new_state": ConversationState.LESSON,
                "tutor_text": f"Wonderful! Let's keep going! 🚀 What would you like to do — quiz, lesson, or something else?",
                "lesson_intent": "lesson",
                "subject_switch": None,
            }
        else:
            return {
                "new_state": ConversationState.WRAPUP,
                "tutor_text": (
                    f"Fantastic work today, {username}! 🌟 "
                    f"Come back whenever you're ready to learn more. See you next time! 👋"
                ),
                "lesson_intent": None,
                "subject_switch": None,
            }

    # Default fallback — treat as LESSON
    return {
        "new_state": ConversationState.LESSON,
        "tutor_text": "",
        "lesson_intent": detected,
        "subject_switch": None,
    }


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
    state: str, username: str, subject: str, current_topic: str, stats: dict
) -> str:
    """Return the re-engagement prompt to resume after answering an inline question."""
    if state == ConversationState.CHECKIN:
        return (
            f"Now, what would you like to practice today — "
            f"**Phonics**, **Reading**, or **Spelling**? "
            f"Or shall we continue with **{current_topic}**?"
        )
    return f"Shall we get back to our lesson on **{current_topic}**? 😊"


def _quick_answer(utterance: str, subject: str, grade: int, current_topic: str) -> str:
    """
    Handle a quick inline question using Socratic guidance.

    Instead of giving the full answer, guides the student to think for themselves
    before offering a brief clarification.
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
        return "That's a great question! What do you already know about it? Let's think together! 🤔"

# ─────────────────────────────────────────────
# Intent detection keywords
# ─────────────────────────────────────────────
QUIZ_KEYWORDS = {"quiz", "test", "question", "practise", "practice", "try", "challenge"}
REVIEW_KEYWORDS = {"review", "revise", "revision", "remind", "recap", "again", "redo"}
LESSON_KEYWORDS = {"teach", "learn", "explain", "what is", "how do", "show me", "tell me"}
VISUAL_KEYWORDS = {"chart", "diagram", "picture", "show", "alphabet", "phonics chart"}
READ_ALOUD_KEYWORDS = {"read aloud", "read out", "reading practice", "read this",
                       "i'll read", "let me read", "practice reading"}
VOCABULARY_KEYWORDS = {"vocabulary", "vocab", "new words", "word list", "define", "meaning",
                       "what does", "word of the day"}
WRITE_KEYWORDS = {"write", "writing", "grammar", "sentence", "compose", "let me write",
                  "writing practice", "write a sentence"}
PRONUNCIATION_KEYWORDS = {"pronunciation", "how to say", "how do you say", "say this",
                          "how is it said", "sound out", "pronounce"}
ADVANCE_GRADE_KEYWORDS = {"harder", "next grade", "grade 2", "grade 3", "grade 4", "grade 5",
                          "advance", "move up", "level up", "i'm ready for", "too easy"}
HINT_KEYWORDS = {"hint", "clue", "help me", "i need a hint", "give me a hint"}
GREETING_KEYWORDS = {"hey", "hi", "hello", "good morning", "good afternoon", "good evening"}


def _has_keyword_match(text: str, keyword: str) -> bool:
    """
    Return True when a keyword appears naturally in text.

    Single-word keywords are matched on word boundaries to avoid partial
    substring false positives (e.g., "hey" won't match "they"). Multi-word
    phrases are matched as substrings so greetings like "good morning class"
    are still detected.
    """
    if " " in keyword:
        return keyword in text
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def determine_intent(student_input: str) -> str:
    """
    Classify the student's message intent.

    Returns:
        One of: "quiz", "review", "lesson", "visual", "answer", "greeting",
                "read_aloud", "vocabulary", "write", "pronunciation",
                "advance_grade", "hint".
    """
    text = student_input.lower()

    # New academic intents (check before generic ones)
    if any(kw in text for kw in READ_ALOUD_KEYWORDS):
        return "read_aloud"

    if any(kw in text for kw in VOCABULARY_KEYWORDS):
        return "vocabulary"

    if any(kw in text for kw in WRITE_KEYWORDS):
        return "write"

    if any(kw in text for kw in PRONUNCIATION_KEYWORDS):
        return "pronunciation"

    if any(kw in text for kw in ADVANCE_GRADE_KEYWORDS):
        return "advance_grade"

    if any(kw in text for kw in HINT_KEYWORDS):
        return "hint"

    if any(_has_keyword_match(text, kw) for kw in GREETING_KEYWORDS):
        return "greeting"

    # Existing intents
    if any(kw in text for kw in QUIZ_KEYWORDS):
        return "quiz"

    if any(kw in text for kw in REVIEW_KEYWORDS):
        return "review"

    if any(kw in text for kw in VISUAL_KEYWORDS):
        return "visual"

    if any(kw in text for kw in LESSON_KEYWORDS):
        return "lesson"

    # If it's short (1-4 words), might be answering a quiz
    words = student_input.strip().split()
    if len(words) <= 4:
        return "answer"

    return "lesson"


def process_student_input(
    student_input: str,
    username: str,
    subject: str,
    grade: int,
    current_topic: str,
    pending_quiz: Optional[dict] = None,
    session_history: Optional[list] = None,
) -> dict:
    """
    Main orchestration function: process a student message and return a response.

    Enhanced flow:
    1. Detect intent (lesson, quiz, review, visual, answer, read_aloud,
       vocabulary, write, pronunciation, advance_grade, hint)
    2. Check personalization: confusion, pacing, skip eligibility
    3. RAG retrieval for curriculum questions
    4. Generate response using appropriate prompt template
    5. Update student profile (mastery, session log)
    6. Check grade advancement eligibility
    7. Return structured response

    Args:
        student_input: Text from the student.
        username: Student's username.
        subject: Current subject ("Phonics", "Reading", "Spelling").
        grade: Grade level (1-5).
        current_topic: The current lesson topic string.
        pending_quiz: If a quiz question is awaiting an answer, pass it here.
        session_history: List of previous chat messages (for context).

    Returns:
        Dict with:
          - "message": str — main AI response text
          - "intent": str — detected intent
          - "quiz_questions": list or None
          - "visual_type": str or None
          - "mistake_info": dict or None
          - "performance": dict
          - "next_topic": dict or None
          - "personalization": dict — pacing / confusion info
          - "grade_advancement": dict or None
    """
    # 1. Load student profile
    profile = student_db.get_or_create_student(username)
    stats = student_db.get_stats(username)
    mistake_history = student_db.get_mistake_history(username)
    topic_mastery = student_db.get_topic_mastery(username)
    independence_info = student_db.get_independence_info(username)

    # 2. Detect intent
    intent = determine_intent(student_input)

    # If there's a pending quiz question, the student is answering it
    if pending_quiz and intent == "answer":
        return _handle_quiz_answer(
            student_input=student_input,
            username=username,
            subject=subject,
            grade=grade,
            pending_quiz=pending_quiz,
            stats=stats,
            mistake_history=mistake_history,
            topic_mastery=topic_mastery,
            independence_info=independence_info,
        )

    # 3. Personalization check
    recent_answers = [
        {"is_correct": m.get("type") == "correct",
         "topic": current_topic,
         "student_input": m.get("student_answer", "")}
        for m in mistake_history[-5:]
    ]
    confusion_info = personalization_engine.detect_confusion(profile, recent_answers)
    performance = adaptive_path.evaluate_performance(stats)
    pace_info = personalization_engine.calculate_pace(profile, performance)

    # 4. Retrieve relevant curriculum context via RAG
    rag_query = f"{subject} {current_topic} {student_input}"
    rag_context = ai_teacher.retrieve_context(rag_query, subject=subject)

    # 5. Route to intent handler
    personalization_data = {
        "confusion": confusion_info,
        "pace": pace_info,
    }

    # Check grade advancement
    grade_check = personalization_engine.check_grade_advancement(profile, topic_mastery)

    if intent == "quiz":
        result = _handle_quiz_request(
            username=username, subject=subject, grade=grade,
            current_topic=current_topic, rag_context=rag_context,
            stats=stats, mistake_history=mistake_history,
            independence_info=independence_info,
        )
    elif intent == "review":
        result = _handle_review_request(
            username=username, subject=subject, grade=grade,
            stats=stats, mistake_history=mistake_history, rag_context=rag_context,
        )
    elif intent == "visual":
        result = _handle_visual_request(student_input=student_input, subject=subject)
    elif intent == "read_aloud":
        result = _handle_read_aloud_request(
            student_input=student_input, username=username,
            subject=subject, grade=grade, current_topic=current_topic,
            rag_context=rag_context,
        )
    elif intent == "vocabulary":
        result = _handle_vocabulary_request(
            username=username, subject=subject, grade=grade,
            current_topic=current_topic, stats=stats,
        )
    elif intent == "write":
        result = _handle_writing_request(
            student_input=student_input, username=username,
            subject=subject, grade=grade, current_topic=current_topic,
        )
    elif intent == "pronunciation":
        result = _handle_pronunciation_request(
            student_input=student_input, username=username,
            subject=subject, grade=grade, current_topic=current_topic,
        )
    elif intent == "advance_grade":
        result = _handle_advance_grade_request(
            username=username, grade=grade, grade_check=grade_check,
        )
    elif intent == "hint":
        result = _handle_hint_request(
            student_input=student_input, username=username,
            subject=subject, grade=grade, stats=stats,
            independence_info=independence_info,
        )
    elif intent == "greeting":
        result = {
            "intent": "greeting",
            "response": (
                f"Hi {username}! 👋 It's great to hear from you. "
                f"What would you like to learn in {subject} today?"
            ),
            "visual": None,
            "pending_quiz": None,
            "grade_advanced": False,
        }
    else:
        # Default: lesson/explanation — with confusion-aware strategy
        result = _handle_lesson_request(
            student_input=student_input, username=username,
            subject=subject, grade=grade, current_topic=current_topic,
            rag_context=rag_context, stats=stats, mistake_history=mistake_history,
            confusion_info=confusion_info, independence_info=independence_info,
        )

    # 6. Attach personalization and grade advancement data
    result["personalization"] = personalization_data
    result["grade_advancement"] = grade_check if grade_check.get("should_advance") else None

    return result


# ─────────────────────────────────────────────
# Intent handlers
# ─────────────────────────────────────────────

def _handle_lesson_request(
    student_input: str,
    username: str,
    subject: str,
    grade: int,
    current_topic: str,
    rag_context: str,
    stats: dict,
    mistake_history: list,
    confusion_info: Optional[dict] = None,
    independence_info: Optional[dict] = None,
) -> dict:
    """Handle a lesson explanation request, with confusion-adaptive and Socratic strategy."""
    if independence_info is None:
        independence_info = student_db.get_independence_info(username)

    # Apply confusion strategy if needed
    prompt_modifier = ""
    if confusion_info and confusion_info.get("is_confused"):
        strategy_data = personalization_engine.get_alternative_explanation_strategy(
            current_topic,
            stats.get("confusion_count", 0),
        )
        prompt_modifier = strategy_data["prompt_modifier"]
        student_db.increment_confusion_count(username)

    # Build prompt — Socratic questioning first, not direct explanation
    task = student_input
    if prompt_modifier:
        task = f"{student_input}\n\nTEACHING STRATEGY: {prompt_modifier}"

    prompt = human_engine.build_system_prompt(
        subject=subject,
        grade=grade,
        student_name=username,
        difficulty=stats.get("difficulty_level", "Beginner"),
        recent_mistakes=mistake_history[-3:],
        rag_context=rag_context,
        task=task,
        mode="explain",
        independence_score=independence_info.get("independence_score", 0.5),
        socratic_level=independence_info.get("socratic_level", 1),
    )

    # Call LLM
    response_text = ai_teacher.call_llm(prompt, mode="explain", subject=subject)

    # Increment lesson count
    student_db.increment_lesson(username)

    # Track independence: only penalise when the student asks for help mid-problem.
    # Proactive lesson requests (no recent wrong answer) are exploratory, not dependency.
    recent_wrongs = [m for m in mistake_history[-3:] if m.get("type") not in ("correct",)]
    is_seeking_help = bool(recent_wrongs) or any(
        w in student_input.lower()
        for w in ("help", "don't understand", "confused", "stuck", "what is", "tell me", "give me")
    )
    student_db.update_independence_score(username, solved_independently=not is_seeking_help)

    # Check for badge
    _check_and_award_badges(username, stats)

    # Performance evaluation
    performance = adaptive_path.evaluate_performance(stats)

    return {
        "message": response_text,
        "intent": "lesson",
        "quiz_questions": None,
        "visual_type": _suggest_visual(subject, current_topic),
        "mistake_info": None,
        "performance": performance,
        "next_topic": adaptive_path.recommend_next_topic(
            subject,
            stats.get("current_lesson_index", 0),
            mistake_history,
            stats.get("difficulty_level", "Beginner"),
        ),
    }


def _handle_quiz_request(
    username: str,
    subject: str,
    grade: int,
    current_topic: str,
    rag_context: str,
    stats: dict,
    mistake_history: list,
    independence_info: Optional[dict] = None,
) -> dict:
    """Generate Socratic quiz questions for the current topic."""
    if independence_info is None:
        independence_info = student_db.get_independence_info(username)
    weak_areas = mistake_analyzer.get_repeated_weak_areas(mistake_history)
    prompt = human_engine.build_quiz_prompt(
        subject=subject,
        grade=grade,
        topic=current_topic,
        difficulty=stats.get("difficulty_level", "Beginner"),
        rag_context=rag_context,
        weak_areas=weak_areas,
        num_questions=3,
        independence_score=independence_info.get("independence_score", 0.5),
    )

    raw_response = ai_teacher.call_llm(prompt, mode="quiz", subject=subject)

    # Parse JSON quiz questions
    quiz_questions = _parse_quiz_json(raw_response)

    if quiz_questions:
        return {
            "message": f"📝 Let's test your knowledge of **{current_topic}**! Here are your questions:",
            "intent": "quiz",
            "quiz_questions": quiz_questions,
            "visual_type": None,
            "mistake_info": None,
            "performance": adaptive_path.evaluate_performance(stats),
            "next_topic": None,
        }
    else:
        # Fallback: return as plain text
        return {
            "message": raw_response,
            "intent": "quiz",
            "quiz_questions": None,
            "visual_type": None,
            "mistake_info": None,
            "performance": adaptive_path.evaluate_performance(stats),
            "next_topic": None,
        }


def _handle_quiz_answer(
    student_input: str,
    username: str,
    subject: str,
    grade: int,
    pending_quiz: dict,
    stats: dict,
    mistake_history: list,
    topic_mastery: Optional[dict] = None,
    independence_info: Optional[dict] = None,
) -> dict:
    """
    Check a student's answer to a pending quiz question.

    Uses Socratic guidance: after a wrong answer, guides with a hint question
    rather than revealing the correct answer immediately.
    """
    correct_answer = pending_quiz.get("correct_answer", "")
    explanation = pending_quiz.get("explanation", "")
    socratic_hint = pending_quiz.get("hint", "")

    # Analyse the answer
    analysis = mistake_analyzer.analyze_answer(
        student_answer=student_input,
        correct_answer=correct_answer,
        subject=subject,
    )

    # Update student progress
    student_db.update_progress(username, {"correct": analysis["is_correct"]})

    # Track independence: correct answers without hints indicate growing independence
    student_db.update_independence_score(username, solved_independently=analysis["is_correct"])

    # Update topic mastery in personalization engine
    current_topic = pending_quiz.get("topic", subject)
    if topic_mastery is not None:
        updated_mastery = personalization_engine.update_topic_mastery(
            {"topic_mastery": topic_mastery}, current_topic, analysis["is_correct"]
        )
        student_db.update_topic_mastery(username, current_topic, updated_mastery.get(current_topic, 0.5))

    feedback_message = analysis["explanation"]

    if not analysis["is_correct"]:
        # Record mistake
        student_db.add_mistake(username, {
            "word": correct_answer,
            "correct_answer": correct_answer,
            "student_answer": student_input,
            "type": analysis["type"],
        })
        # Use Socratic correction: guide with a question, not the full answer
        rag_context = ai_teacher.retrieve_context(
            f"{subject} {correct_answer} {analysis['related_rule']}", subject=subject
        )
        correction_prompt = human_engine.build_correction_prompt(
            subject=subject,
            grade=grade,
            student_answer=student_input,
            correct_answer=correct_answer,
            error_type=analysis["type"],
            rule=analysis["related_rule"],
            rag_context=rag_context,
        )
        feedback_message = ai_teacher.call_llm(correction_prompt, mode="explain")
    else:
        feedback_message = f"{analysis['explanation']}\n\n💡 {explanation}"

    # Update difficulty
    updated_stats = student_db.get_stats(username)
    performance = adaptive_path.evaluate_performance(updated_stats)

    if performance["should_increase"] or performance["should_decrease"]:
        new_difficulty = adaptive_path.get_next_difficulty(
            updated_stats.get("difficulty_level", "Beginner"),
            performance["should_increase"],
            performance["should_decrease"],
        )
        student_db.update_student_field(username, "difficulty_level", new_difficulty)

    _check_and_award_badges(username, updated_stats)

    return {
        "message": feedback_message,
        "intent": "answer",
        "quiz_questions": None,
        "visual_type": None,
        "mistake_info": analysis if not analysis["is_correct"] else None,
        "performance": performance,
        "next_topic": None,
    }


def _handle_review_request(
    username: str,
    subject: str,
    grade: int,
    stats: dict,
    mistake_history: list,
    rag_context: str,
) -> dict:
    """Generate a review session for the student."""
    weak_areas = mistake_analyzer.get_repeated_weak_areas(mistake_history)
    review_topics = weak_areas[:3] if weak_areas else [
        adaptive_path.get_current_topic(subject, max(0, stats.get("current_lesson_index", 0) - 1))
    ]

    prompt = human_engine.build_review_prompt(
        subject=subject,
        grade=grade,
        topics=review_topics,
        accuracy=stats.get("accuracy_pct", 0),
        recent_mistakes=mistake_history[-5:],
        rag_context=rag_context,
    )

    response_text = ai_teacher.call_llm(prompt, mode="explain", subject=subject)

    return {
        "message": response_text,
        "intent": "review",
        "quiz_questions": None,
        "visual_type": None,
        "mistake_info": None,
        "performance": adaptive_path.evaluate_performance(stats),
        "next_topic": adaptive_path.recommend_next_topic(
            subject,
            stats.get("current_lesson_index", 0),
            mistake_history,
            stats.get("difficulty_level", "Beginner"),
        ),
    }


def _handle_visual_request(student_input: str, subject: str) -> dict:
    """Determine which visual to show based on the request."""
    text = student_input.lower()

    if "alphabet" in text or "letter" in text:
        visual_type = "alphabet"
    elif "blend" in text:
        visual_type = "blends"
    elif "digraph" in text:
        visual_type = "digraphs"
    elif "word family" in text or "family" in text:
        visual_type = "word_family"
    elif "progress" in text:
        visual_type = "progress"
    else:
        visual_type = "alphabet" if subject == "Phonics" else "vowels"

    return {
        "message": "📊 Here's a visual aid for you! Study it and feel free to ask questions.",
        "intent": "visual",
        "quiz_questions": None,
        "visual_type": visual_type,
        "mistake_info": None,
        "performance": {},
        "next_topic": None,
    }


# ─────────────────────────────────────────────
# New Academic Mode Handlers
# ─────────────────────────────────────────────

def _handle_read_aloud_request(
    student_input: str,
    username: str,
    subject: str,
    grade: int,
    current_topic: str,
    rag_context: str,
) -> dict:
    """Handle a reading aloud / pronunciation practice request."""
    # Use RAG context as the passage to read
    passage = rag_context[:500] if rag_context else (
        f"Practice reading aloud about {current_topic} in {subject}. "
        f"The teacher will help you with any difficult words."
    )

    prompt = human_engine.build_reading_practice_prompt(
        passage=passage,
        student_reading=student_input,
        grade_level=grade,
    )

    response_text = ai_teacher.call_llm(prompt, mode="explain", subject=subject)
    student_db.increment_lesson(username)

    return {
        "message": response_text,
        "intent": "read_aloud",
        "quiz_questions": None,
        "visual_type": None,
        "mistake_info": None,
        "performance": {},
        "next_topic": None,
    }


def _handle_vocabulary_request(
    username: str,
    subject: str,
    grade: int,
    current_topic: str,
    stats: dict,
) -> dict:
    """Handle a vocabulary building request using spaced repetition awareness."""
    difficulty = stats.get("difficulty_level", "Beginner")
    vocab_bank = student_db.get_vocabulary_bank(username)
    known_words = [w.get("word", "") for w in vocab_bank if w.get("times_correct", 0) >= 2]

    word_list = personalization_engine.get_vocabulary_for_lesson(subject, current_topic, difficulty)

    prompt = human_engine.build_vocabulary_prompt(
        words=word_list,
        grade_level=grade,
        known_words=known_words[:10],
    )

    response_text = ai_teacher.call_llm(prompt, mode="explain", subject=subject)

    # Add new words to vocabulary bank
    for word_data in word_list[:3]:
        if word_data.get("word") and word_data["word"] not in known_words:
            student_db.add_vocabulary_word(username, {
                "word": word_data["word"],
                "definition": word_data.get("definition", ""),
                "times_seen": 1,
                "times_correct": 0,
            })

    student_db.increment_lesson(username)

    return {
        "message": response_text,
        "intent": "vocabulary",
        "quiz_questions": None,
        "visual_type": None,
        "mistake_info": None,
        "performance": {},
        "next_topic": None,
    }


def _handle_writing_request(
    student_input: str,
    username: str,
    subject: str,
    grade: int,
    current_topic: str,
) -> dict:
    """Handle a writing / grammar practice request."""
    # Determine if this is a writing submission or a request for a prompt
    words = student_input.lower().split()
    is_submission = len(words) >= 4 and not any(
        kw in student_input.lower() for kw in {"give me", "what should", "can you give", "write about"}
    )

    if is_submission:
        # Student submitted a sentence — evaluate it
        writing_prompt = f"Write a sentence using vocabulary from {current_topic}."
        prompt = human_engine.build_writing_evaluation_prompt(
            student_sentence=student_input,
            prompt_given=writing_prompt,
            grade_level=grade,
        )
        response_text = ai_teacher.call_llm(prompt, mode="explain", subject=subject)

        # Save writing sample
        student_db.add_writing_sample(username, {
            "prompt": writing_prompt,
            "response": student_input,
            "feedback": response_text[:200],
            "score": _DEFAULT_WRITING_SCORE,
        })
    else:
        # Student wants a writing prompt
        writing_prompt = (
            f"As a warm, encouraging Grade {grade} writing teacher, "
            f"give the student a simple, fun writing prompt related to '{current_topic}' in {subject}. "
            f"The prompt should be appropriate for Grade {grade} "
            f"(e.g., 'Write a sentence using the word ___'). "
            f"Keep it short and exciting!"
        )
        response_text = ai_teacher.call_llm(writing_prompt, mode="explain", subject=subject)

    student_db.increment_lesson(username)

    return {
        "message": response_text,
        "intent": "write",
        "quiz_questions": None,
        "visual_type": None,
        "mistake_info": None,
        "performance": {},
        "next_topic": None,
    }


def _handle_pronunciation_request(
    student_input: str,
    username: str,
    subject: str,
    grade: int,
    current_topic: str,
) -> dict:
    """Handle a pronunciation help request."""
    # Extract the word/phrase the student wants to practice
    target_word = _extract_pronunciation_target(student_input, current_topic)

    from voice_engine import evaluate_pronunciation
    pron_result = evaluate_pronunciation(
        student_audio_text=student_input,
        expected_text=target_word,
        grade_level=grade,
    )

    prompt = human_engine.build_pronunciation_feedback_prompt(
        expected=target_word,
        actual=student_input,
        similarity_score=pron_result["similarity"],
        grade_level=grade,
    )

    response_text = ai_teacher.call_llm(prompt, mode="explain", subject=subject)

    # Log pronunciation score
    student_db.add_pronunciation_score(username, {
        "word": target_word,
        "similarity_score": pron_result["similarity"],
    })

    return {
        "message": response_text,
        "intent": "pronunciation",
        "quiz_questions": None,
        "visual_type": None,
        "mistake_info": None,
        "performance": {},
        "next_topic": None,
    }


def _handle_advance_grade_request(
    username: str,
    grade: int,
    grade_check: dict,
) -> dict:
    """Handle a student's request to advance to a harder grade."""
    if grade_check.get("should_advance"):
        message = (
            f"{grade_check['message']}\n\n"
            f"Use the grade selector in the sidebar to move to Grade {grade + 1}. "
            f"Or keep practising here — I'll keep track of your progress! 🌟"
        )
    else:
        mastery = grade_check.get("mastery_pct", 0)
        message = (
            f"I love your ambition! You're currently at {mastery:.0f}% mastery of Grade {grade} content.\n\n"
            f"{grade_check.get('message', '')}\n\n"
            f"Keep going — you're making amazing progress! 💪"
        )

    return {
        "message": message,
        "intent": "advance_grade",
        "quiz_questions": None,
        "visual_type": None,
        "mistake_info": None,
        "performance": {},
        "next_topic": None,
    }


def _handle_hint_request(
    student_input: str,
    username: str,
    subject: str,
    grade: int,
    stats: dict,
    independence_info: Optional[dict] = None,
) -> dict:
    """
    Handle a student's request for a hint.

    Uses Socratic questioning: guides the student toward discovering the answer
    rather than providing it directly. Tracks hint usage to adjust independence score.
    """
    student_db.increment_hint_usage(username)
    # Hint requests reduce independence score — student needed help
    student_db.update_independence_score(username, solved_independently=False)

    if independence_info is None:
        independence_info = student_db.get_independence_info(username)

    indep = independence_info.get("independence_score", 0.5)
    socratic_lv = independence_info.get("socratic_level", 1)

    # Socratic hint: guide toward the answer, not reveal it
    hint_prompt = (
        f"A Grade {grade} student (independence score: {indep:.2f}, socratic level: {socratic_lv}) "
        f"is learning {subject} and asked for a hint. "
        f"Their message: '{student_input}'. "
        f"IMPORTANT: Do NOT give the answer. Instead, ask ONE Socratic guiding question "
        f"that helps them think it through themselves "
        f"(e.g., 'What sound does the first letter make?', "
        f"'Can you think of a word that rhymes with it?', "
        f"'What do you already know about this?'). "
        f"Keep it warm, encouraging, and one question only. "
        f"End with 'I know you can figure it out! 💪'"
    )

    response_text = ai_teacher.call_llm(hint_prompt, mode="explain", subject=subject)

    return {
        "message": f"💡 {response_text}",
        "intent": "hint",
        "quiz_questions": None,
        "visual_type": None,
        "mistake_info": None,
        "performance": {},
        "next_topic": None,
    }


def _extract_pronunciation_target(student_input: str, current_topic: str) -> str:
    """Extract the target word/phrase the student wants to pronounce."""
    text = student_input.lower()
    # Try to find quoted word
    quoted = re.findall(r"[\"'](.+?)[\"']", student_input)
    if quoted:
        return quoted[0]
    # Try "how to say X" / "how do you say X" patterns
    patterns = [
        r"how (?:to|do you) say (.+?)(?:\?|$)",
        r"pronounce (.+?)(?:\?|$)",
        r"say (.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    # Fall back to current topic
    return current_topic


# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────

def parse_quiz_json(raw: str) -> Optional[list]:
    """Parse JSON quiz questions from LLM output, tolerating markdown code fences."""
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and data:
            return data
    except json.JSONDecodeError:
        pass

    return None


# Internal alias
_parse_quiz_json = parse_quiz_json


def _suggest_visual(subject: str, topic: str) -> Optional[str]:
    """Suggest a relevant visual type based on subject and topic."""
    topic_lower = topic.lower()
    if subject == "Phonics":
        if "alphabet" in topic_lower or "letter" in topic_lower:
            return "alphabet"
        if "blend" in topic_lower:
            return "blends"
        if "digraph" in topic_lower:
            return "digraphs"
        if "vowel" in topic_lower:
            return "vowels"
    if subject == "Spelling" and "family" in topic_lower:
        return "word_family"
    return None


def _check_and_award_badges(username: str, stats: dict) -> None:
    """Award badges based on milestones."""
    lessons = stats.get("total_lessons", 0)
    correct = stats.get("correct_answers", 0)
    accuracy = stats.get("accuracy_pct", 0)
    streak = stats.get("streak_days", 0)

    milestones = [
        (lessons >= 1, "First Lesson! 🌟"),
        (lessons >= 10, "10 Lessons Complete! 📚"),
        (correct >= 10, "10 Correct Answers! ✅"),
        (correct >= 50, "50 Correct Answers! 🏆"),
        (accuracy >= 80 and lessons >= 5, "High Achiever! 🎯"),
        (streak >= 3, "3-Day Streak! 🔥"),
        (streak >= 7, "One Week Strong! 💪"),
    ]

    for condition, badge in milestones:
        if condition:
            student_db.award_badge(username, badge)
