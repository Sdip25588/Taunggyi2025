"""
intent_classifier.py — Intent detection for student messages.

Classifies a student's free-text input into one of the recognised learning
intents using an ordered keyword-matching strategy.  The ordering is
deliberately strict: more-specific educational intents are evaluated before
broader, catch-all ones, so that "show me how to spell" correctly resolves to
"lesson" rather than "visual" (because "show" was removed from VISUAL_KEYWORDS
— see comment there).

Short greeting words such as "hi" and "hey" are matched with word-boundary
regex (\b) to prevent false positives like "this" containing "hi".

Intents returned
----------------
  "read_aloud"     — student wants to practise reading a passage aloud
  "vocabulary"     — student wants to learn / review vocabulary
  "write"          — writing or grammar practice
  "pronunciation"  — student wants pronunciation guidance
  "advance_grade"  — student wants to move to a harder grade
  "hint"           — student is asking for a hint or clue
  "greeting"       — hello / hi / good morning etc.
  "quiz"           — quiz / test / practice questions
  "review"         — review / revision / recap
  "visual"         — chart / diagram / alphabet picture
  "lesson"         — teach / explain / what is / how do
  "general"        — casual / conversational input
  "answer"         — short response, likely answering a quiz question
"""

import re

# ─────────────────────────────────────────────
# Keyword sets — ordered from most specific to most general
# ─────────────────────────────────────────────

READ_ALOUD_KEYWORDS: frozenset = frozenset({
    "read aloud", "read out", "reading practice", "read this",
    "i'll read", "let me read", "practice reading",
})

VOCABULARY_KEYWORDS: frozenset = frozenset({
    "vocabulary", "vocab", "new words", "word list", "define", "meaning",
    "what does", "word of the day",
})

WRITE_KEYWORDS: frozenset = frozenset({
    "write", "writing", "grammar", "sentence", "compose", "let me write",
    "writing practice", "write a sentence",
})

PRONUNCIATION_KEYWORDS: frozenset = frozenset({
    "pronunciation", "how to say", "how do you say", "say this",
    "how is it said", "sound out", "pronounce",
})

ADVANCE_GRADE_KEYWORDS: frozenset = frozenset({
    "harder", "next grade", "grade 2", "grade 3", "grade 4", "grade 5",
    "advance", "move up", "level up", "i'm ready for", "too easy",
})

HINT_KEYWORDS: frozenset = frozenset({
    "hint", "clue", "help me", "i need a hint", "give me a hint",
})

GREETING_KEYWORDS: frozenset = frozenset({
    "hey", "hi", "hello", "good morning", "good afternoon", "good evening",
})

QUIZ_KEYWORDS: frozenset = frozenset({
    "quiz", "test", "question", "practise", "practice", "try", "challenge",
})

REVIEW_KEYWORDS: frozenset = frozenset({
    "review", "revise", "revision", "remind", "recap", "again", "redo",
})

VISUAL_KEYWORDS: frozenset = frozenset({
    "chart", "diagram", "picture", "alphabet", "phonics chart",
    # NOTE: "show" intentionally omitted — it appears in LESSON_KEYWORDS as the
    # phrase "show me", so including it here would cause "show me how to spell"
    # to match VISUAL before reaching the LESSON intent.
})

LESSON_KEYWORDS: frozenset = frozenset({
    "teach", "learn", "explain", "what is", "how do", "show me", "tell me",
})

GENERAL_KEYWORDS: frozenset = frozenset({
    "how are you", "can we", "i want", "interest", "before",
})


def determine_intent(student_input: str) -> str:
    """
    Classify the student's message into a learning intent.

    Evaluation order ensures specific educational intents take precedence over
    general keyword matches (e.g. "show me how to spell" → "lesson" not
    "visual").

    Args:
        student_input: Raw text from the student (may be pre-sanitised by
                       the caller but this function tolerates any string).

    Returns:
        One of: "read_aloud", "vocabulary", "write", "pronunciation",
        "advance_grade", "hint", "greeting", "quiz", "review", "visual",
        "lesson", "general", "answer".
    """
    text = student_input.lower()

    # Specific academic intents — evaluated before generic ones to avoid
    # false matches (e.g. VISUAL_KEYWORDS contains "show" which would
    # otherwise swallow "show me how to spell").
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

    # Greeting keywords use word-boundary regex to prevent short keywords like
    # "hi" from matching as substrings of longer words (e.g. "hi" in "something").
    if any(re.search(r"\b" + re.escape(kw) + r"\b", text) for kw in GREETING_KEYWORDS):
        return "greeting"

    if any(kw in text for kw in QUIZ_KEYWORDS):
        return "quiz"

    if any(kw in text for kw in REVIEW_KEYWORDS):
        return "review"

    if any(kw in text for kw in VISUAL_KEYWORDS):
        return "visual"

    if any(kw in text for kw in LESSON_KEYWORDS):
        return "lesson"

    # General/conversational — checked after all educational intents so that
    # "can we start a quiz?" still routes to "quiz".
    if any(kw in text for kw in GENERAL_KEYWORDS):
        return "general"

    # Short utterances (≤4 words) are likely quiz answers
    if len(student_input.strip().split()) <= 4:
        return "answer"

    return "lesson"
