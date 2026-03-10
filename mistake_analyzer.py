"""
mistake_analyzer.py — Detect and explain student errors.

Uses difflib for spelling similarity and regex patterns for
phonics-specific error detection (e.g., b/d confusion).
"""

import re
import difflib
from typing import Optional


# ─────────────────────────────────────────────
# Common phonics confusion patterns
# ─────────────────────────────────────────────
PHONICS_CONFUSION_PATTERNS: list[dict] = [
    {
        "pattern": r"\bdat\b|\bdot\b|\bdig\b|\bdog\b",
        "correct_sound": "b",
        "confused_sound": "d",
        "rule": "The letters 'b' and 'd' look similar! 'b' has its bump on the RIGHT, 'd' has its bump on the LEFT. Think of a 'bed' 🛏️ — b is on the left, d is on the right!",
        "type": "letter_reversal",
    },
    {
        "pattern": r"\bqig\b|\bqat\b|\bqin\b",
        "correct_sound": "p",
        "confused_sound": "q",
        "rule": "The letters 'p' and 'q' are mirror images! 'p' has its tail going DOWN and to the RIGHT. Practice: 'p' points right, 'q' points left.",
        "type": "letter_reversal",
    },
    {
        "pattern": r"\bf(?:ink|in|is|at|umb)\b",
        "correct_sound": "th",
        "confused_sound": "f",
        "rule": "The 'th' sound is made by putting your tongue between your teeth. 'f' uses your top teeth on your bottom lip. Try saying 'think' — feel your tongue!",
        "type": "sound_confusion",
    },
    {
        "pattern": r"\bwite\b|\bwit\b|\bwip\b",
        "correct_sound": "wh",
        "confused_sound": "w",
        "rule": "'wh' words like 'white', 'while', 'when' start with 'wh'. When you see 'wh', make a soft blowing sound!",
        "type": "digraph_confusion",
    },
    {
        "pattern": r"\bsh(?:ip|op|e)\b",
        "correct_sound": "sh",
        "confused_sound": "s",
        "rule": "'sh' makes one combined sound — like sshhh! when you want someone to be quiet 🤫. It's different from just 's'.",
        "type": "digraph_confusion",
    },
    {
        "pattern": r"\bck\b",
        "correct_sound": "ck",
        "confused_sound": "k",
        "rule": "After a short vowel, we usually write 'ck' (like 'duck', 'back', 'kick'). 'ck' makes the same /k/ sound.",
        "type": "spelling_pattern",
    },
]

# Common silent-e spelling mistakes
SILENT_E_PATTERN = re.compile(
    r"\b(hop|cap|mat|pin|cub|bit|kit|hat|plan|strip|grip)(e?)\b", re.IGNORECASE
)


def analyze_answer(
    student_answer: str,
    correct_answer: str,
    subject: str = "Spelling",
    context: str = "",
) -> dict:
    """
    Analyze a student's answer and return structured mistake data.

    Args:
        student_answer: What the student typed/said.
        correct_answer: The expected correct answer.
        subject: One of "Phonics", "Reading", "Spelling".
        context: Optional surrounding lesson context.

    Returns:
        Dict with: is_correct, type, explanation, related_rule,
                   similarity_score, student_answer, correct_answer, confidence.
    """
    student_clean = student_answer.strip().lower()
    correct_clean = correct_answer.strip().lower()

    if student_clean == correct_clean:
        return {
            "is_correct": True,
            "type": "correct",
            "explanation": f"✅ That's correct! '{correct_answer}' — well done! 🌟",
            "related_rule": "",
            "similarity_score": 1.0,
            "student_answer": student_answer,
            "correct_answer": correct_answer,
            "confidence": 1.0,
        }

    # Compute similarity
    similarity = difflib.SequenceMatcher(
        None, student_clean, correct_clean
    ).ratio()

    # Detect phonics-specific errors
    phonics_result = _check_phonics_confusion(student_clean, correct_clean)
    if phonics_result:
        return {
            "is_correct": False,
            "type": phonics_result["type"],
            "explanation": _build_explanation(student_answer, correct_answer, phonics_result["rule"]),
            "related_rule": phonics_result["rule"],
            "similarity_score": similarity,
            "student_answer": student_answer,
            "correct_answer": correct_answer,
            "confidence": 0.9,
        }

    # Detect silent-e errors
    silent_e_result = _check_silent_e(student_clean, correct_clean)
    if silent_e_result:
        return {
            "is_correct": False,
            "type": "silent_e",
            "explanation": silent_e_result,
            "related_rule": "Adding a silent 'e' at the end changes the vowel sound from short to long.",
            "similarity_score": similarity,
            "student_answer": student_answer,
            "correct_answer": correct_answer,
            "confidence": 0.85,
        }

    # Generic spelling feedback using similarity
    explanation = _build_spelling_explanation(student_answer, correct_answer, similarity)
    return {
        "is_correct": False,
        "type": "spelling",
        "explanation": explanation,
        "related_rule": _infer_spelling_rule(student_clean, correct_clean),
        "similarity_score": similarity,
        "student_answer": student_answer,
        "correct_answer": correct_answer,
        "confidence": 0.7,
    }


def get_repeated_weak_areas(mistake_history: list) -> list[str]:
    """
    Identify topics/types where the student makes repeated mistakes.

    Args:
        mistake_history: List of mistake dicts from student.py.

    Returns:
        List of weak area labels sorted by frequency.
    """
    from collections import Counter

    types = [m.get("type", "unknown") for m in mistake_history]
    words = [m.get("word", "") for m in mistake_history]

    type_counts = Counter(types)
    word_counts = Counter(words)

    weak_areas = []
    for error_type, count in type_counts.most_common(5):
        if count >= 2:
            weak_areas.append(f"{error_type} (×{count})")

    for word, count in word_counts.most_common(3):
        if count >= 2 and word:
            weak_areas.append(f"word '{word}' (×{count})")

    return weak_areas


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _check_phonics_confusion(student: str, correct: str) -> Optional[dict]:
    """Check if the error matches known phonics confusion patterns."""
    # b/d reversal
    if (student.replace("b", "d") == correct or
            student.replace("d", "b") == correct):
        return PHONICS_CONFUSION_PATTERNS[0]

    # p/q reversal
    if (student.replace("p", "q") == correct or
            student.replace("q", "p") == correct):
        return PHONICS_CONFUSION_PATTERNS[1]

    # th/f confusion
    if (student.replace("f", "th") == correct or
            student.replace("th", "f") == correct):
        return PHONICS_CONFUSION_PATTERNS[2]

    # wh/w confusion
    if student.replace("wh", "w") == correct or student.replace("w", "wh") == correct:
        return PHONICS_CONFUSION_PATTERNS[3]

    return None


def _check_silent_e(student: str, correct: str) -> Optional[str]:
    """Detect missing or extra silent-e errors."""
    if correct.endswith("e") and not student.endswith("e"):
        if correct[:-1] == student:
            return (
                f"Almost! You wrote '{student}' but the answer is '{correct}'. "
                f"Don't forget the silent 'e' at the end! "
                f"The silent 'e' makes the vowel say its name — "
                f"like in 'hope' (the 'o' says its name 'oh'). ✨"
            )
    if student.endswith("e") and not correct.endswith("e"):
        if student[:-1] == correct:
            return (
                f"Good try! You wrote '{student}' but the answer is '{correct}'. "
                f"This word doesn't have a silent 'e' — "
                f"the vowel keeps its short sound. 📖"
            )
    return None


def _build_explanation(student_answer: str, correct_answer: str, rule: str) -> str:
    """Build an encouraging error explanation."""
    return (
        f"Almost! 🌟 You wrote '{student_answer}' but the correct answer is "
        f"'{correct_answer}'.\n\n"
        f"💡 **Remember:** {rule}"
    )


def _build_spelling_explanation(
    student_answer: str, correct_answer: str, similarity: float
) -> str:
    """Build a spelling-specific explanation with diff highlighting."""
    diff = list(difflib.ndiff(student_answer.lower(), correct_answer.lower()))
    changed = [c for c in diff if c.startswith("+ ") or c.startswith("- ")]

    if similarity >= 0.8:
        closeness = "You're very close!"
    elif similarity >= 0.5:
        closeness = "Good try — you got some letters right!"
    else:
        closeness = "Keep practising — you'll get it!"

    return (
        f"Almost! 🌟 You wrote **'{student_answer}'** but the correct spelling is "
        f"**'{correct_answer}'**. {closeness}\n\n"
        f"💡 Look carefully at each letter and try sounding it out: "
        f"{' '.join(correct_answer.upper())}"
    )


def _infer_spelling_rule(student: str, correct: str) -> str:
    """Infer a brief spelling rule based on the difference."""
    if len(correct) > len(student):
        return "Check if any letters are missing — sound out each syllable slowly."
    if len(correct) < len(student):
        return "Check if you added any extra letters — keep it simple!"
    return "Look at the vowel sounds carefully — short and long vowels sound different."
