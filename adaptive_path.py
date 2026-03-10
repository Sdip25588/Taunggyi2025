"""
adaptive_path.py — Adaptive difficulty and lesson path adjustment.

Adjusts lesson difficulty based on student performance metrics,
recommends next topics, and generates adaptive quiz focus areas.
"""

from collections import Counter
from typing import Optional

# ─────────────────────────────────────────────
# Difficulty levels within the Grades 1-3 scope
# ─────────────────────────────────────────────
DIFFICULTY_LEVELS: list[str] = ["Beginner", "Elementary", "Intermediate"]

# Curriculum topics per subject (ordered from easiest to hardest)
PHONICS_TOPICS: list[str] = [
    "Alphabet Letters & Sounds",
    "Short Vowels (a, e, i, o, u)",
    "Consonant Sounds",
    "CVC Words (cat, dog, sit)",
    "Long Vowels (Silent-E)",
    "Consonant Blends (bl, br, cl, cr, ...)",
    "Digraphs (ch, sh, th, wh, ph)",
    "Vowel Digraphs (ai, ea, oa, ...)",
    "R-Controlled Vowels (ar, er, ir, or, ur)",
    "Diphthongs (oi, oy, ou, ow)",
]

READING_TOPICS: list[str] = [
    "Letter Recognition",
    "Simple 2-3 Letter Words",
    "Short Sentences",
    "McGuffey Lesson 1-10",
    "McGuffey Lesson 11-20",
    "McGuffey Lesson 21-30",
    "Simple Stories",
    "Comprehension Questions",
]

SPELLING_TOPICS: list[str] = [
    "Simple 3-Letter Words",
    "Short Vowel Families (-at, -an, -ap)",
    "Short Vowel Families (-it, -in, -ig)",
    "Short Vowel Families (-ot, -op, -og)",
    "Short Vowel Families (-ut, -un, -ug)",
    "Long Vowel Patterns",
    "Consonant Blends Spelling",
    "Digraph Spelling",
    "Common Sight Words",
    "Two-Syllable Words",
]

SUBJECT_TOPICS: dict = {
    "Phonics": PHONICS_TOPICS,
    "Reading": READING_TOPICS,
    "Spelling": SPELLING_TOPICS,
}


def evaluate_performance(student_profile: dict) -> dict:
    """
    Evaluate a student's current performance and return an assessment.

    Args:
        student_profile: Dict from student.get_stats().

    Returns:
        Dict with: level, should_increase, should_decrease,
                   rolling_accuracy, recommendation.
    """
    correct = student_profile.get("correct_answers", 0)
    wrong = student_profile.get("wrong_answers", 0)
    total = correct + wrong

    # For adaptive decisions, we need at least 5 questions answered
    if total < 5:
        return {
            "level": student_profile.get("difficulty_level", "Beginner"),
            "should_increase": False,
            "should_decrease": False,
            "rolling_accuracy": None,
            "recommendation": "Keep going! Answer a few more questions so I can personalise your lessons. 😊",
        }

    accuracy = correct / total * 100

    current_level = student_profile.get("difficulty_level", "Beginner")
    level_idx = DIFFICULTY_LEVELS.index(current_level) if current_level in DIFFICULTY_LEVELS else 0

    should_increase = accuracy > 80 and total >= 5 and level_idx < len(DIFFICULTY_LEVELS) - 1
    should_decrease = accuracy < 50 and total >= 5 and level_idx > 0

    if should_increase:
        recommendation = (
            f"🌟 Amazing! Your accuracy is {accuracy:.0f}%. "
            f"You're ready to move to **{DIFFICULTY_LEVELS[level_idx + 1]}** level!"
        )
    elif should_decrease:
        recommendation = (
            f"💪 Don't give up! Let's review some easier material "
            f"to build your confidence. Accuracy: {accuracy:.0f}%."
        )
    else:
        recommendation = (
            f"📚 Great progress! Accuracy: {accuracy:.0f}%. "
            f"Keep up the good work at **{current_level}** level."
        )

    return {
        "level": current_level,
        "should_increase": should_increase,
        "should_decrease": should_decrease,
        "rolling_accuracy": round(accuracy, 1),
        "recommendation": recommendation,
    }


def get_next_difficulty(
    current_difficulty: str, should_increase: bool, should_decrease: bool
) -> str:
    """
    Return the next difficulty level based on performance signals.

    Args:
        current_difficulty: Current difficulty string.
        should_increase: True if accuracy warrants levelling up.
        should_decrease: True if accuracy warrants levelling down.

    Returns:
        New difficulty level string.
    """
    idx = DIFFICULTY_LEVELS.index(current_difficulty) if current_difficulty in DIFFICULTY_LEVELS else 0

    if should_increase:
        idx = min(idx + 1, len(DIFFICULTY_LEVELS) - 1)
    elif should_decrease:
        idx = max(idx - 1, 0)

    return DIFFICULTY_LEVELS[idx]


def recommend_next_topic(
    subject: str,
    current_lesson_index: int,
    mistake_history: list,
    difficulty: str,
) -> dict:
    """
    Recommend the next lesson topic based on progress and weaknesses.

    Args:
        subject: "Phonics", "Reading", or "Spelling".
        current_lesson_index: Index into the topic list for the subject.
        mistake_history: List of mistake dicts from student.py.
        difficulty: Current difficulty level string.

    Returns:
        Dict with: topic, reason, is_review, topic_index.
    """
    topics = SUBJECT_TOPICS.get(subject, PHONICS_TOPICS)

    # Find repeated weak topics from mistakes
    weak_topics = _find_weak_topics(mistake_history, subject)

    # If there are weak topics and difficulty is not Intermediate,
    # suggest a review session first
    if weak_topics and difficulty != "Intermediate":
        weak_topic = weak_topics[0]
        return {
            "topic": weak_topic,
            "reason": f"You've had a few challenges with '{weak_topic}'. Let's review it! 💪",
            "is_review": True,
            "topic_index": current_lesson_index,
        }

    # Advance to next topic
    next_index = min(current_lesson_index + 1, len(topics) - 1)
    next_topic = topics[next_index]

    return {
        "topic": next_topic,
        "reason": f"You're ready for the next lesson: **{next_topic}**! 🚀",
        "is_review": False,
        "topic_index": next_index,
    }


def get_current_topic(subject: str, lesson_index: int) -> str:
    """Return the topic name for a given subject and lesson index."""
    topics = SUBJECT_TOPICS.get(subject, PHONICS_TOPICS)
    idx = max(0, min(lesson_index, len(topics) - 1))
    return topics[idx]


def calculate_topic_mastery(mistake_history: list, subject: str) -> dict:
    """
    Estimate mastery percentage per topic based on mistake history.

    Args:
        mistake_history: List of mistake dicts.
        subject: Current subject.

    Returns:
        Dict mapping topic → estimated mastery %.
    """
    topics = SUBJECT_TOPICS.get(subject, PHONICS_TOPICS)
    mastery = {topic: 50 for topic in topics}  # Start at 50% (not yet assessed)

    # Each mistake on a topic reduces its mastery score
    for mistake in mistake_history:
        mistake_type = mistake.get("type", "")
        for topic in topics:
            keywords = topic.lower().split()
            if any(kw in mistake_type.lower() for kw in keywords):
                mastery[topic] = max(0, mastery[topic] - 10)

    return mastery


def generate_adaptive_quiz_topics(
    subject: str, mistake_history: list, count: int = 5
) -> list[str]:
    """
    Generate a list of quiz topics weighted toward weak areas.

    Args:
        subject: Current subject.
        mistake_history: List of mistake dicts.
        count: How many topic slots to fill.

    Returns:
        List of topic strings (may repeat for weaker areas).
    """
    mastery = calculate_topic_mastery(mistake_history, subject)
    topics = SUBJECT_TOPICS.get(subject, PHONICS_TOPICS)

    # Weight topics inversely by mastery (weaker = more questions)
    weighted: list[str] = []
    for topic in topics:
        m = mastery.get(topic, 100)
        weight = max(1, int((100 - m) / 20) + 1)
        weighted.extend([topic] * weight)

    if not weighted:
        weighted = topics[:5]

    # Select topics, cycling through weighted list
    selected = []
    for i in range(count):
        selected.append(weighted[i % len(weighted)])

    return selected


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _find_weak_topics(mistake_history: list, subject: str) -> list[str]:
    """Return topics with 2+ mistakes, sorted by frequency."""
    topics = SUBJECT_TOPICS.get(subject, PHONICS_TOPICS)
    topic_mistake_count: Counter = Counter()

    for mistake in mistake_history:
        mistake_str = (
            mistake.get("type", "") + " " +
            mistake.get("word", "") + " " +
            mistake.get("correct_answer", "")
        ).lower()
        for topic in topics:
            if any(kw in mistake_str for kw in topic.lower().split()):
                topic_mistake_count[topic] += 1

    return [t for t, c in topic_mistake_count.most_common() if c >= 2]
