"""
personalization_engine.py — Deep adaptive learning personalization.

Handles confusion detection, progressive hints, auto-pacing,
grade advancement recommendations, and topic mastery tracking.
"""

import difflib
from typing import Optional

from config import (
    CONFUSION_THRESHOLD,
    MASTERY_ADVANCE_THRESHOLD,
    MASTERY_SKIP_THRESHOLD,
    HINT_LEVELS,
    PACE_FAST_THRESHOLD,
    PACE_SLOW_THRESHOLD,
)

# ─────────────────────────────────────────────
# Topic taxonomy per subject
# ─────────────────────────────────────────────
TOPIC_TAXONOMY: dict = {
    "Phonics": [
        "short_vowels", "long_vowels", "consonants", "blends",
        "digraphs", "r_controlled_vowels",
    ],
    "Reading": [
        "sight_words", "fluency", "comprehension", "vocabulary_in_context",
    ],
    "Spelling": [
        "CVC_words", "silent_e", "vowel_teams", "suffixes", "prefixes",
    ],
}

# Phrases that signal confusion
CONFUSION_PHRASES: list[str] = [
    "i don't understand", "i dont understand", "i'm confused",
    "im confused", "what do you mean", "i don't get it", "i dont get it",
    "can you explain", "i'm lost", "im lost", "huh?", "what?",
    "i don't know", "i dont know", "help me", "i give up",
    "this is hard", "too hard", "i can't", "i cant",
]


def detect_confusion(student_profile: dict, recent_answers: list) -> dict:
    """
    Detect when a student is confused based on answer patterns and explicit signals.

    Rules:
    - 2+ wrong answers in a row on the same topic → confused
    - Student uses a confusion phrase → confused
    - confusion_count already at/above threshold → confused

    Args:
        student_profile: Full student profile dict from student.py.
        recent_answers: List of recent answer dicts {is_correct, topic, student_input}.

    Returns:
        Dict: {is_confused: bool, topic: str, suggestion: str, strategy: str}
    """
    is_confused = False
    confused_topic = student_profile.get("current_subject", "Phonics")
    suggestion = ""
    strategy = "standard"

    # Check confusion_count in profile
    if student_profile.get("confusion_count", 0) >= CONFUSION_THRESHOLD:
        is_confused = True

    # Check recent_answers for consecutive wrong answers
    if len(recent_answers) >= CONFUSION_THRESHOLD:
        last_n = recent_answers[-CONFUSION_THRESHOLD:]
        if all(not a.get("is_correct", True) for a in last_n):
            is_confused = True
            # Find the common topic
            topics = [a.get("topic", "") for a in last_n if a.get("topic")]
            if topics:
                confused_topic = topics[-1]

    # Check if student explicitly expressed confusion
    for answer in recent_answers[-3:]:
        text = answer.get("student_input", "").lower()
        if any(phrase in text for phrase in CONFUSION_PHRASES):
            is_confused = True
            break

    if is_confused:
        strategy = get_alternative_explanation_strategy(
            confused_topic,
            student_profile.get("confusion_count", 0),
        )["strategy_name"]
        suggestion = (
            f"It looks like '{confused_topic}' needs a fresh approach! "
            f"Let's try a different way — I'll explain it step by step. 🌟"
        )

    return {
        "is_confused": is_confused,
        "topic": confused_topic,
        "suggestion": suggestion,
        "strategy": strategy,
    }


def get_alternative_explanation_strategy(topic: str, attempt_number: int) -> dict:
    """
    When student is confused, try progressively different explanation approaches.

    Attempt 0: Original explanation from RAG/curriculum
    Attempt 1: Simpler words, shorter sentences
    Attempt 2: Analogy or real-life example
    Attempt 3: Trigger a visual aid
    Attempt 4+: Break into micro-steps

    Args:
        topic: The topic the student is struggling with.
        attempt_number: How many times we've tried explaining (0-indexed).

    Returns:
        Dict: {strategy_name: str, prompt_modifier: str}
    """
    strategies = [
        {
            "strategy_name": "curriculum",
            "prompt_modifier": (
                "Explain this using the curriculum content, step by step. "
                "Use numbered steps and simple language."
            ),
        },
        {
            "strategy_name": "simplified",
            "prompt_modifier": (
                "Explain this in the simplest possible words. "
                "Use very short sentences. Imagine teaching a 5-year-old. "
                "One idea per sentence."
            ),
        },
        {
            "strategy_name": "analogy",
            "prompt_modifier": (
                "Use a real-life analogy or everyday example to explain this. "
                f"For example, compare '{topic}' to something familiar like "
                "food, animals, toys, or family. Make it fun and relatable!"
            ),
        },
        {
            "strategy_name": "visual",
            "prompt_modifier": (
                "Suggest a visual aid to explain this. Describe what a "
                "picture or diagram would look like. Use ASCII art or "
                "simple text diagrams if helpful."
            ),
        },
        {
            "strategy_name": "micro_steps",
            "prompt_modifier": (
                "Break this concept into the absolute smallest possible steps. "
                "Each step should be one tiny thing. Number each step 1, 2, 3... "
                "Start with something the student definitely already knows."
            ),
        },
    ]

    idx = min(attempt_number, len(strategies) - 1)
    return strategies[idx]


def generate_progressive_hints(
    question: str, correct_answer: str, hint_level: int
) -> tuple[str, int]:
    """
    Generate a progressive hint instead of giving the answer directly.

    hint_level 1: Very vague — think about the category
    hint_level 2: More specific — partial letter/sound information
    hint_level 3: Almost the answer — fill-in-the-blank style
    hint_level 4: Guided answer — sound it out together

    Args:
        question: The quiz/practice question.
        correct_answer: The correct answer string.
        hint_level: Current hint level (1-4).

    Returns:
        Tuple: (hint_text: str, next_hint_level: int)
    """
    answer = correct_answer.strip()
    answer_lower = answer.lower()

    if hint_level <= 1:
        # Very vague
        hint = (
            f"💡 **Hint 1:** Think about the sounds in this word. "
            f"How many letters does it have? Count with me: {len(answer)} letters!"
        )
        next_level = 2

    elif hint_level == 2:
        # Show first letter and last letter
        if len(answer) >= 2:
            hint = (
                f"💡 **Hint 2:** The answer starts with "
                f"**'{answer[0].upper()}'** and ends with **'{answer[-1].upper()}'**. "
                f"Can you think of the letters in between?"
            )
        else:
            hint = f"💡 **Hint 2:** The answer is a very short word — just {len(answer)} letter(s)!"
        next_level = 3

    elif hint_level == 3:
        # Blanks with some letters revealed
        if len(answer) >= 3:
            blanked = " ".join(
                c.upper() if i % 2 == 0 else "_"
                for i, c in enumerate(answer_lower)
            )
        else:
            blanked = answer[0].upper() + " _" * (len(answer) - 1)
        hint = f"💡 **Hint 3:** Fill in the missing letters: **{blanked}**"
        next_level = 4

    else:
        # Guided answer — sound it out
        letters_spaced = " - ".join(answer.upper())
        hint = (
            f"💡 **Hint 4 (Final):** Let's sound it out together: "
            f"**{letters_spaced}**. "
            f"Say each sound: {' ... '.join(answer_lower)}. "
            f"Put it together — the answer is **'{answer}'**! "
            f"Say it with me! You've got this! 🌟"
        )
        next_level = HINT_LEVELS  # No more hints

    return hint, next_level


def calculate_pace(student_profile: dict, recent_performance: dict) -> dict:
    """
    Determine the appropriate lesson pace based on recent accuracy.

    Args:
        student_profile: Full student profile dict.
        recent_performance: Dict from adaptive_path.evaluate_performance().

    Returns:
        Dict: {pace: str ("fast"/"normal"/"slow"), recommendation: str}
    """
    accuracy_pct = recent_performance.get("rolling_accuracy")
    if accuracy_pct is None:
        # Not enough data yet
        return {
            "pace": student_profile.get("pace_preference", "normal"),
            "recommendation": "Answer a few more questions and I'll personalise your pace! 😊",
        }

    accuracy = accuracy_pct / 100.0

    if accuracy > PACE_FAST_THRESHOLD:
        pace = "fast"
        recommendation = (
            f"🚀 Excellent! {accuracy_pct:.0f}% accuracy — you're flying! "
            f"I'll introduce new content faster and skip easy reviews."
        )
    elif accuracy < PACE_SLOW_THRESHOLD:
        pace = "slow"
        recommendation = (
            f"💪 Let's take our time with {accuracy_pct:.0f}% — "
            f"I'll give you more practice and simpler steps. You're doing great!"
        )
    else:
        pace = "normal"
        recommendation = (
            f"📚 Good progress at {accuracy_pct:.0f}%! "
            f"We'll keep a steady mix of new content and review."
        )

    return {"pace": pace, "recommendation": recommendation}


def check_grade_advancement(student_profile: dict, topic_mastery: dict) -> dict:
    """
    Determine if a student is ready to advance to the next grade level.

    Args:
        student_profile: Full student profile dict.
        topic_mastery: Dict {topic: mastery_float (0-1)}.

    Returns:
        Dict: {should_advance: bool, current_mastery: float, message: str,
               topic_mastery: dict, eligible_topics: list}
    """
    current_grade = student_profile.get("grade_level", 1)
    max_grade = 5  # from GRADE_LEVELS in config

    if current_grade >= max_grade:
        return {
            "should_advance": False,
            "current_mastery": 1.0,
            "message": f"🌟 You've reached the top grade level — Grade {current_grade}! Amazing!",
            "topic_mastery": topic_mastery,
            "eligible_topics": [],
        }

    if not topic_mastery:
        return {
            "should_advance": False,
            "current_mastery": 0.0,
            "message": "Keep learning and I'll track your mastery! 😊",
            "topic_mastery": topic_mastery,
            "eligible_topics": [],
        }

    overall_mastery = sum(topic_mastery.values()) / len(topic_mastery)

    # Topics individually eligible to advance
    eligible_topics = [
        t for t, v in topic_mastery.items() if v >= 0.95
    ]

    should_advance = overall_mastery >= MASTERY_ADVANCE_THRESHOLD

    if should_advance:
        message = (
            f"🎉 Wow! You've mastered {overall_mastery * 100:.0f}% of Grade {current_grade} "
            f"content! You're ready for Grade {current_grade + 1}! "
            f"Would you like to start exploring it? 🌟"
        )
    else:
        needed = round((MASTERY_ADVANCE_THRESHOLD - overall_mastery) * 100, 1)
        message = (
            f"You've mastered {overall_mastery * 100:.0f}% of Grade {current_grade} content. "
            f"Just {needed}% more and you'll be ready for Grade {current_grade + 1}! 💪"
        )

    return {
        "should_advance": should_advance,
        "current_mastery": round(overall_mastery, 4),
        "message": message,
        "topic_mastery": topic_mastery,
        "eligible_topics": eligible_topics,
    }


def update_topic_mastery(
    student_profile: dict, topic: str, is_correct: bool
) -> dict:
    """
    Update mastery for a specific topic using a weighted rolling average.

    Recent performance is weighted more heavily (0.3 new, 0.7 existing).

    Args:
        student_profile: Full student profile dict.
        topic: Topic identifier (e.g. 'short_vowels').
        is_correct: Whether the student answered correctly.

    Returns:
        Updated topic_mastery dict.
    """
    mastery: dict = dict(student_profile.get("topic_mastery", {}))
    current = mastery.get(topic, 0.5)  # Default to 50% for new topics
    new_signal = 1.0 if is_correct else 0.0
    # Weighted rolling average: 30% new signal, 70% existing
    updated = round(0.7 * current + 0.3 * new_signal, 4)
    mastery[topic] = updated
    return mastery


def get_vocabulary_for_lesson(
    subject: str, topic: str, difficulty: str
) -> list[dict]:
    """
    Return a grade-appropriate vocabulary word list for a lesson.

    This provides built-in vocabulary when RAG is not available.

    Args:
        subject: "Phonics", "Reading", or "Spelling".
        topic: Current lesson topic.
        difficulty: "Beginner", "Elementary", or "Intermediate".

    Returns:
        List of dicts: [{word, definition, example, pronunciation_hint}, ...]
    """
    # Built-in vocabulary by subject and difficulty
    vocabulary_bank: dict = {
        "Phonics": {
            "Beginner": [
                {"word": "cat", "definition": "a small furry pet that says 'meow'",
                 "example": "The cat sat on the mat.", "pronunciation_hint": "k - æ - t"},
                {"word": "dog", "definition": "a friendly pet that says 'woof'",
                 "example": "My dog likes to run.", "pronunciation_hint": "d - ɒ - g"},
                {"word": "sun", "definition": "the bright star that gives us light and warmth",
                 "example": "The sun is very bright today.", "pronunciation_hint": "s - ʌ - n"},
                {"word": "hat", "definition": "something you wear on your head",
                 "example": "I wear a hat in the rain.", "pronunciation_hint": "h - æ - t"},
                {"word": "big", "definition": "very large in size",
                 "example": "That is a big elephant!", "pronunciation_hint": "b - ɪ - g"},
            ],
            "Elementary": [
                {"word": "blend", "definition": "two consonants together that each keep their sound",
                 "example": "The word 'blue' starts with the blend 'bl'.", "pronunciation_hint": "b - l - end"},
                {"word": "vowel", "definition": "the letters a, e, i, o, u",
                 "example": "Every word needs a vowel.", "pronunciation_hint": "v - aʊ - el"},
                {"word": "silent", "definition": "making no sound",
                 "example": "The 'e' in 'cake' is silent.", "pronunciation_hint": "s - aɪ - lent"},
            ],
            "Intermediate": [
                {"word": "digraph", "definition": "two letters that make one sound, like 'ch' or 'sh'",
                 "example": "'Ch' in 'chair' is a digraph.", "pronunciation_hint": "d - aɪ - graf"},
                {"word": "diphthong", "definition": "a sound made by combining two vowel sounds",
                 "example": "'oi' in 'coin' is a diphthong.", "pronunciation_hint": "d - ɪ - f - θɒŋ"},
            ],
        },
        "Reading": {
            "Beginner": [
                {"word": "read", "definition": "to look at words and understand them",
                 "example": "I read a book every night.", "pronunciation_hint": "r - iː - d"},
                {"word": "story", "definition": "a written account of events",
                 "example": "Let's read a story together.", "pronunciation_hint": "s - t - ɔː - r - iː"},
                {"word": "word", "definition": "a group of letters that means something",
                 "example": "Can you spell this word?", "pronunciation_hint": "w - ɜː - d"},
            ],
            "Elementary": [
                {"word": "sentence", "definition": "a group of words that makes a complete thought",
                 "example": "A sentence starts with a capital letter.", "pronunciation_hint": "s - en - tence"},
                {"word": "paragraph", "definition": "a group of sentences about one idea",
                 "example": "Each paragraph has a new idea.", "pronunciation_hint": "p - ar - a - graph"},
            ],
            "Intermediate": [
                {"word": "comprehension", "definition": "understanding what you read",
                 "example": "Reading comprehension means you understand the story.",
                 "pronunciation_hint": "com - pre - hen - sion"},
                {"word": "fluency", "definition": "reading smoothly and at a good speed",
                 "example": "Reading fluency takes lots of practice.",
                 "pronunciation_hint": "f - l - u - en - cy"},
            ],
        },
        "Spelling": {
            "Beginner": [
                {"word": "spell", "definition": "to write or say the letters of a word in order",
                 "example": "Can you spell your name?", "pronunciation_hint": "s - p - el"},
                {"word": "letter", "definition": "a symbol in the alphabet",
                 "example": "The letter A is the first in the alphabet.", "pronunciation_hint": "l - et - ter"},
            ],
            "Elementary": [
                {"word": "pattern", "definition": "a repeated arrangement of sounds or letters",
                 "example": "The -at pattern: cat, bat, hat, mat.", "pronunciation_hint": "p - at - tern"},
                {"word": "syllable", "definition": "a part of a word with one vowel sound",
                 "example": "'Happy' has two syllables: hap-py.", "pronunciation_hint": "s - yl - la - ble"},
            ],
            "Intermediate": [
                {"word": "prefix", "definition": "letters added to the beginning of a word to change its meaning",
                 "example": "'Un-' is a prefix: unhappy, undo.", "pronunciation_hint": "p - r - ee - f - ix"},
                {"word": "suffix", "definition": "letters added to the end of a word",
                 "example": "'-ing' is a suffix: running, playing.", "pronunciation_hint": "s - uf - f - ix"},
            ],
        },
    }

    subject_bank = vocabulary_bank.get(subject, vocabulary_bank["Phonics"])
    words = subject_bank.get(difficulty, subject_bank.get("Beginner", []))

    # Filter by topic relevance if possible
    if topic:
        topic_lower = topic.lower()
        relevant = [w for w in words if topic_lower in w.get("definition", "").lower()
                    or topic_lower in w.get("example", "").lower()]
        if relevant:
            return relevant

    return words
