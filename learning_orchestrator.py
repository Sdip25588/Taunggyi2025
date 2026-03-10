"""
learning_orchestrator.py — Central coordinator for learning sessions.

Receives student input, determines intent, orchestrates RAG retrieval,
LLM calls, mistake analysis, student profile updates, and visual generation.
"""

import json
import logging
from typing import Optional

from config import APP_CONFIG
import ai_teacher
import human_engine
import student as student_db
import mistake_analyzer
import adaptive_path
import visual_teacher

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Intent detection keywords
# ─────────────────────────────────────────────
QUIZ_KEYWORDS = {"quiz", "test", "question", "practise", "practice", "try", "challenge"}
REVIEW_KEYWORDS = {"review", "revise", "revision", "remind", "recap", "again", "redo"}
LESSON_KEYWORDS = {"teach", "learn", "explain", "what is", "how do", "show me", "tell me"}
VISUAL_KEYWORDS = {"chart", "diagram", "picture", "show", "alphabet", "phonics chart"}


def determine_intent(student_input: str) -> str:
    """
    Classify the student's message intent.

    Returns:
        One of: "quiz", "review", "lesson", "visual", "answer", "greeting".
    """
    text = student_input.lower()

    # Check for quiz intent
    if any(kw in text for kw in QUIZ_KEYWORDS):
        return "quiz"

    # Check for review intent
    if any(kw in text for kw in REVIEW_KEYWORDS):
        return "review"

    # Check for visual request
    if any(kw in text for kw in VISUAL_KEYWORDS):
        return "visual"

    # Check for lesson/explanation request
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

    Args:
        student_input: Text from the student.
        username: Student's username.
        subject: Current subject ("Phonics", "Reading", "Spelling").
        grade: Grade level (1-3).
        current_topic: The current lesson topic string.
        pending_quiz: If a quiz question is awaiting an answer, pass it here.
        session_history: List of previous chat messages (for context).

    Returns:
        Dict with:
          - "message": str — main AI response text
          - "intent": str — detected intent
          - "quiz_questions": list or None — new quiz Qs if intent="quiz"
          - "visual_type": str or None — which visual to show
          - "mistake_info": dict or None — mistake analysis if intent="answer"
          - "performance": dict — current performance evaluation
          - "next_topic": dict or None — topic recommendation
    """
    # 1. Load student profile
    profile = student_db.get_or_create_student(username)
    stats = student_db.get_stats(username)
    mistake_history = student_db.get_mistake_history(username)

    # 2. Determine intent
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
        )

    # 3. Retrieve relevant curriculum context via RAG
    rag_query = f"{subject} {current_topic} {student_input}"
    rag_context = ai_teacher.retrieve_context(rag_query, subject=subject)

    # 4. Build response based on intent
    if intent == "quiz":
        return _handle_quiz_request(
            username=username,
            subject=subject,
            grade=grade,
            current_topic=current_topic,
            rag_context=rag_context,
            stats=stats,
            mistake_history=mistake_history,
        )

    elif intent == "review":
        return _handle_review_request(
            username=username,
            subject=subject,
            grade=grade,
            stats=stats,
            mistake_history=mistake_history,
            rag_context=rag_context,
        )

    elif intent == "visual":
        return _handle_visual_request(
            student_input=student_input,
            subject=subject,
        )

    else:
        # Default: lesson/explanation
        return _handle_lesson_request(
            student_input=student_input,
            username=username,
            subject=subject,
            grade=grade,
            current_topic=current_topic,
            rag_context=rag_context,
            stats=stats,
            mistake_history=mistake_history,
        )


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
) -> dict:
    """Handle a lesson explanation request."""
    # Build prompt
    prompt = human_engine.build_system_prompt(
        subject=subject,
        grade=grade,
        student_name=username,
        difficulty=stats.get("difficulty_level", "Beginner"),
        recent_mistakes=mistake_history[-3:],
        rag_context=rag_context,
        task=student_input,
        mode="explain",
    )

    # Call LLM
    response_text = ai_teacher.call_llm(prompt, mode="explain", subject=subject)

    # Increment lesson count
    student_db.increment_lesson(username)

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
) -> dict:
    """Generate quiz questions for the current topic."""
    weak_areas = mistake_analyzer.get_repeated_weak_areas(mistake_history)
    prompt = human_engine.build_quiz_prompt(
        subject=subject,
        grade=grade,
        topic=current_topic,
        difficulty=stats.get("difficulty_level", "Beginner"),
        rag_context=rag_context,
        weak_areas=weak_areas,
        num_questions=3,
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
) -> dict:
    """Check a student's answer to a pending quiz question."""
    correct_answer = pending_quiz.get("correct_answer", "")
    explanation = pending_quiz.get("explanation", "")

    # Analyse the answer
    analysis = mistake_analyzer.analyze_answer(
        student_answer=student_input,
        correct_answer=correct_answer,
        subject=subject,
    )

    # Update student progress
    student_db.update_progress(username, {"correct": analysis["is_correct"]})

    feedback_message = analysis["explanation"]

    if not analysis["is_correct"]:
        # Record mistake
        student_db.add_mistake(username, {
            "word": correct_answer,
            "correct_answer": correct_answer,
            "student_answer": student_input,
            "type": analysis["type"],
        })
        # Get enhanced correction from LLM
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
