"""
learning_orchestrator.py — Central coordinator for learning sessions.

Receives student input, determines intent, orchestrates RAG retrieval,
LLM calls, mistake analysis, student profile updates, and visual generation.

Conversation state machine and intent detection have been extracted into
focused sub-modules for easier testing and maintenance:

  * intent_classifier.py  — keyword-based intent detection
  * badge_service.py      — badge milestone checking and awarding
  * session_manager.py    — voice-first conversation state machine
                            (GREETING → CHECKIN → LESSON → WRAPUP)
"""

import json
import logging
import re
from typing import Optional, TypedDict

from config import APP_CONFIG
import ai_teacher
import human_engine
import student as student_db
import mistake_analyzer
import adaptive_path
import visual_teacher
import personalization_engine
from voice_engine import evaluate_pronunciation

# Sub-module imports — these replace code that previously lived in this file
from intent_classifier import determine_intent  # noqa: F401 (re-exported)
from badge_service import check_and_award_badges
from session_manager import (  # noqa: F401 (re-exported for backward compat)
    ConversationState,
    get_initial_greeting,
    get_wrapup_message,
    handle_student_utterance,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Default writing score assigned when a student submits a sample.
# 0.7 represents a neutral/passing baseline (neither penalising a first
# attempt nor awarding a perfect score without evaluation).  Override in
# APP_CONFIG if you want a different default for your curriculum.
_DEFAULT_WRITING_SCORE: float = APP_CONFIG.get("default_writing_score", 0.7)

# Maximum number of characters accepted from student_input.
# Inputs longer than this are silently truncated before being forwarded to
# LLM prompts, preventing token-spend spikes and API-limit errors.
_MAX_INPUT_LENGTH: int = APP_CONFIG.get("max_input_length", 2_000)


# ─────────────────────────────────────────────
# Response schema
# ─────────────────────────────────────────────

class LessonResult(TypedDict, total=False):
    """
    Typed schema for the dict returned by every intent handler and by
    ``process_student_input``.  Using a TypedDict lets type-checkers catch
    missing keys and makes the expected shape explicit for frontend callers.

    All keys listed under ``total=False`` are optional so that handlers can
    omit fields they do not populate (they will be filled with safe defaults
    by ``process_student_input`` before the dict reaches the caller).
    """
    message:          str
    intent:           str
    quiz_questions:   Optional[list]
    visual_type:      Optional[str]
    mistake_info:     Optional[dict]
    performance:      dict
    next_topic:       Optional[dict]
    personalization:  dict
    grade_advancement: Optional[dict]


# ─────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────

def _sanitize_input(text: str) -> str:
    """
    Sanitise and cap student input before passing it to LLM prompts.

    Strips leading/trailing whitespace and truncates to ``_MAX_INPUT_LENGTH``
    characters to prevent token-spend spikes and API-limit errors.
    """
    return text.strip()[:_MAX_INPUT_LENGTH]


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
    # 0. Sanitise and validate input — truncates to _MAX_INPUT_LENGTH chars
    sanitized_input = _sanitize_input(student_input)

    # 1. Load student profile once — passed to handlers to avoid duplicate DB calls
    profile         = student_db.get_or_create_student(username)
    stats           = student_db.get_stats(username)
    mistake_history = student_db.get_mistake_history(username)
    topic_mastery   = student_db.get_topic_mastery(username)
    independence_info = student_db.get_independence_info(username)

    # 2. Detect intent
    intent = determine_intent(sanitized_input)

    # If there's a pending quiz question, the student is answering it
    if pending_quiz and intent == "answer":
        return _handle_quiz_answer(
            student_input=sanitized_input,
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
    rag_query = f"{subject} {current_topic} {sanitized_input}"
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
        result = _handle_visual_request(student_input=sanitized_input, subject=subject)
    elif intent == "read_aloud":
        result = _handle_read_aloud_request(
            student_input=sanitized_input, username=username,
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
            student_input=sanitized_input, username=username,
            subject=subject, grade=grade, current_topic=current_topic,
        )
    elif intent == "pronunciation":
        result = _handle_pronunciation_request(
            student_input=sanitized_input, username=username,
            subject=subject, grade=grade, current_topic=current_topic,
        )
    elif intent == "advance_grade":
        result = _handle_advance_grade_request(
            username=username, grade=grade, grade_check=grade_check,
        )
    elif intent == "hint":
        result = _handle_hint_request(
            student_input=sanitized_input, username=username,
            subject=subject, grade=grade, stats=stats,
            independence_info=independence_info,
        )
    elif intent == "greeting":
        result = _handle_greeting_request(username=username, subject=subject)
    elif intent == "general":
        result = _handle_general_request(
            student_input=sanitized_input,
            username=username,
            subject=subject,
            grade=grade,
        )
    else:
        # Default: lesson/explanation — with confusion-aware strategy
        result = _handle_lesson_request(
            student_input=sanitized_input, username=username,
            subject=subject, grade=grade, current_topic=current_topic,
            rag_context=rag_context, stats=stats, mistake_history=mistake_history,
            confusion_info=confusion_info, independence_info=independence_info,
        )

    # 6. Ensure the result always has all required keys (fill safe defaults)
    result.setdefault("message", "")
    result.setdefault("intent", intent)
    result.setdefault("quiz_questions", None)
    result.setdefault("visual_type", None)
    result.setdefault("mistake_info", None)
    result.setdefault("performance", {})
    result.setdefault("next_topic", None)

    # 7. Attach personalization and grade advancement data
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

    try:
        response_text = ai_teacher.call_llm(prompt, mode="explain", subject=subject)
    except Exception:
        logger.exception("LLM error in _handle_lesson_request for user=%s", username)
        response_text = (
            f"I'm sorry {username}, I had a little trouble just then! 😊 "
            f"Let's try again — what would you like to learn about {current_topic}?"
        )

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
    check_and_award_badges(username, stats)

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


def _handle_greeting_request(username: str, subject: str) -> dict:
    """Respond to a greeting with a friendly welcome message."""
    return {
        "message": (
            f"Hi {username}! 👋 It's great to hear from you. "
            f"What would you like to learn in {subject} today?"
        ),
        "intent": "greeting",
        "quiz_questions": None,
        "visual_type": None,
        "mistake_info": None,
        "performance": {},
        "next_topic": None,
    }


def _handle_general_request(
    student_input: str,
    username: str,
    subject: str,
    grade: int,
) -> dict:
    """Respond conversationally to general or casual student inputs.

    Does not force lesson continuation — responds like a friendly tutor.
    """
    try:
        response_text = ai_teacher.generate_conversational_reply(
            student_input=student_input,
            username=username,
            grade=grade,
            subject=subject,
        )
    except Exception:
        logger.exception("LLM error in _handle_general_request for user=%s", username)
        response_text = (
            f"That's a great question, {username}! 😊 "
            f"I'm here to help you with {subject} whenever you're ready. "
            f"What would you like to explore?"
        )
    return {
        "message": response_text,
        "intent": "general",
        "quiz_questions": None,
        "visual_type": None,
        "mistake_info": None,
        "performance": {},
        "next_topic": None,
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

    try:
        raw_response = ai_teacher.call_llm(prompt, mode="quiz", subject=subject)
    except Exception:
        logger.exception("LLM error in _handle_quiz_request for user=%s", username)
        return {
            "message": (
                "Sorry, I couldn't generate quiz questions right now. "
                "Let's try again in a moment! 😊"
            ),
            "intent": "quiz",
            "quiz_questions": None,
            "visual_type": None,
            "mistake_info": None,
            "performance": adaptive_path.evaluate_performance(stats),
            "next_topic": None,
        }

    # Parse JSON quiz questions
    quiz_questions = parse_quiz_json(raw_response)

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
        try:
            feedback_message = ai_teacher.call_llm(correction_prompt, mode="explain")
        except Exception:
            logger.exception("LLM error in _handle_quiz_answer for user=%s", username)
            feedback_message = (
                f"That's not quite right — the answer was **{correct_answer}**. "
                f"Don't worry, let's keep practising! 💪"
            )
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

    check_and_award_badges(username, updated_stats)

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

    try:
        response_text = ai_teacher.call_llm(prompt, mode="explain", subject=subject)
    except Exception:
        logger.exception("LLM error in _handle_review_request for user=%s", username)
        response_text = (
            "I had a little trouble loading your review right now. "
            "Let's try again in a moment! 😊"
        )

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

    try:
        response_text = ai_teacher.call_llm(prompt, mode="explain", subject=subject)
    except Exception:
        logger.exception("LLM error in _handle_read_aloud_request for user=%s", username)
        response_text = (
            "I had a little trouble just now — let's try reading practice again! 😊"
        )
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

    try:
        response_text = ai_teacher.call_llm(prompt, mode="explain", subject=subject)
    except Exception:
        logger.exception("LLM error in _handle_vocabulary_request for user=%s", username)
        response_text = (
            "I had a little trouble loading vocabulary just now. "
            "Let's try again in a moment! 😊"
        )

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
        try:
            response_text = ai_teacher.call_llm(prompt, mode="explain", subject=subject)
        except Exception:
            logger.exception("LLM error in _handle_writing_request (eval) for user=%s", username)
            response_text = (
                "I had a little trouble evaluating your writing just now. "
                "Great effort — let's try again! 😊"
            )

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
        try:
            response_text = ai_teacher.call_llm(writing_prompt, mode="explain", subject=subject)
        except Exception:
            logger.exception("LLM error in _handle_writing_request (prompt) for user=%s", username)
            response_text = (
                f"Try writing a sentence about **{current_topic}**. "
                f"I'm sure you'll do great! ✏️"
            )

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

    try:
        pron_result = evaluate_pronunciation(
            student_audio_text=student_input,
            expected_text=target_word,
            grade_level=grade,
        )
    except Exception:
        logger.exception("evaluate_pronunciation failed for user=%s word=%s", username, target_word)
        pron_result = {"similarity": 0.0}

    prompt = human_engine.build_pronunciation_feedback_prompt(
        expected=target_word,
        actual=student_input,
        similarity_score=pron_result["similarity"],
        grade_level=grade,
    )

    try:
        response_text = ai_teacher.call_llm(prompt, mode="explain", subject=subject)
    except Exception:
        logger.exception("LLM error in _handle_pronunciation_request for user=%s", username)
        response_text = (
            f"I had a little trouble just now. "
            f"Try saying **{target_word}** slowly — you're doing great! 😊"
        )

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

    try:
        response_text = ai_teacher.call_llm(hint_prompt, mode="explain", subject=subject)
    except Exception:
        logger.exception("LLM error in _handle_hint_request for user=%s", username)
        response_text = "What do you already know about this? Think carefully — I know you can figure it out! 💪"

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
