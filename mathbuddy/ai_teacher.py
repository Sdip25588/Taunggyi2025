"""
ai_teacher.py — MathBuddy conversation logic.

MathBuddyTeacher drives a friendly, text-only chat loop:

  1. Greet the learner by name.
  2. Ask an addition / subtraction word problem at the current difficulty.
  3. Accept the learner's typed answer.
  4. Give immediate, encouraging feedback:
       - Correct  → celebrate and possibly increase difficulty.
       - Incorrect → encourage, explain the correct answer, possibly decrease difficulty.
  5. Repeat until the learner ends the session.

Difficulty is adjusted dynamically:
  - 3 correct answers in a row  → advance one difficulty level (if possible).
  - 2 wrong  answers in a row   → drop one difficulty level  (if possible).

This module has **no external dependencies** beyond the standard library
and the sibling ``question_generator`` module, so it is easy to test in
isolation and to wire into any front-end (Streamlit, CLI, etc.).

No audio, voice, text-to-speech, speech-to-text, recording, or
play/listen functionality is included — all interaction is 100% text.

Future extension points
-----------------------
- Swap ``_rule_based_feedback`` for an LLM call once an API key is available.
- Persist session stats via a ``progress_tracker`` object passed at construction.
- Load hint/explanation templates from a ``templates/`` folder.
"""

import random
import re
from typing import Optional

from mathbuddy.question_generator import (
    DIFFICULTY_LEVELS,
    generate_question,
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

_CORRECT_STREAK_TO_ADVANCE = 3   # correct answers in a row → harder
_WRONG_STREAK_TO_RETREAT = 2     # wrong answers in a row → easier

# Friendly response banks (chosen randomly to avoid repetition)
_CORRECT_PHRASES = [
    "Great job! 🎉 That's exactly right!",
    "Brilliant! ⭐ You got it!",
    "Wonderful! 🌟 Perfect answer!",
    "Excellent work! 👏 You're on a roll!",
    "That's correct! 🎊 Keep it up!",
    "Amazing! 💪 You nailed it!",
]

_WRONG_PHRASES = [
    "Good try! Let's work through it together.",
    "Not quite, but that's okay — let's see how we get there.",
    "Nice effort! Let's look at this step by step.",
    "Don't worry — mistakes help us learn! Here's how:",
    "Almost! Let me show you the thinking.",
    "That was a brave try! Let's figure it out together.",
]

_ENCOURAGEMENT_AFTER_WRONG = [
    "You're doing great — keep going! 💪",
    "Every question makes you smarter! 📚",
    "Keep trying — you've got this! 🌟",
    "I believe in you! Next one will be even better. 😊",
]


# ─────────────────────────────────────────────
# MathBuddyTeacher
# ─────────────────────────────────────────────

class MathBuddyTeacher:
    """
    Stateful, text-only math tutor for a single learner session.

    Attributes
    ----------
    learner_name : str
        The learner's name (used in messages).
    difficulty : str
        Current difficulty level (one of ``DIFFICULTY_LEVELS``).
    score : int
        Number of correct answers this session.
    total_questions : int
        Total questions asked this session.
    """

    def __init__(
        self,
        learner_name: str,
        starting_difficulty: str = "Very Easy",
    ) -> None:
        """
        Initialise a new MathBuddy session.

        Args:
            learner_name:        The learner's first name.
            starting_difficulty: Starting level — "Very Easy", "Easy", or "Medium".

        Raises:
            ValueError: If ``starting_difficulty`` is not a valid level.
        """
        if starting_difficulty not in DIFFICULTY_LEVELS:
            raise ValueError(
                f"Unknown difficulty '{starting_difficulty}'. "
                f"Choose from: {DIFFICULTY_LEVELS}"
            )

        self.learner_name: str = learner_name.strip()
        self.difficulty: str = starting_difficulty

        # Session counters
        self.score: int = 0
        self.total_questions: int = 0

        # Streak tracking for adaptive difficulty
        self._correct_streak: int = 0
        self._wrong_streak: int = 0

        # The question currently waiting for an answer (or None)
        self._current_question: Optional[dict] = None

    # ─────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────

    def greet(self) -> str:
        """
        Return the opening greeting message for the session.

        Returns:
            A friendly plain-text greeting that introduces MathBuddy.
        """
        return (
            f"Hi {self.learner_name}! 👋 I'm MathBuddy, your math helper!\n\n"
            f"We're going to practice some fun math problems together. "
            f"I'll ask you a question, and you type your answer — that's it!\n\n"
            f"We're starting at **{self.difficulty}** level. "
            f"Ready? Let's go! 🚀"
        )

    def ask_next_question(self) -> str:
        """
        Generate the next word problem and store it as the current question.

        The generated question is saved internally so that
        :meth:`check_answer` can validate the learner's response.

        Returns:
            The question text to display to the learner.
        """
        self._current_question = generate_question(
            difficulty=self.difficulty,
            learner_name=self.learner_name,
        )
        self.total_questions += 1
        question_num = self.total_questions

        return (
            f"**Question {question_num}** *(Difficulty: {self.difficulty})*\n\n"
            f"{self._current_question['question']}\n\n"
            f"Type your answer as a number."
        )

    def check_answer(self, user_input: str) -> str:
        """
        Validate the learner's typed answer and return feedback text.

        Handles:
        - Non-numeric input (asks the learner to type a number).
        - Correct answer (celebrates, updates streaks, may advance difficulty).
        - Wrong answer (encourages, explains, may retreat difficulty).

        Args:
            user_input: The raw text the learner typed.

        Returns:
            A plain-text feedback message (may include the explanation and
            an announcement if the difficulty level changed).

        Raises:
            RuntimeError: If called before :meth:`ask_next_question`.
        """
        if self._current_question is None:
            raise RuntimeError(
                "No active question. Call ask_next_question() first."
            )

        parsed = _parse_number(user_input)
        if parsed is None:
            return (
                "Hmm, I didn't understand that. 🤔 "
                "Please type your answer as a number — for example: **7** or **12**."
            )

        correct_answer = self._current_question["answer"]
        is_correct = parsed == correct_answer

        if is_correct:
            feedback = self._handle_correct()
        else:
            feedback = self._handle_wrong(correct_answer)

        # Clear the current question so a fresh one must be generated next
        self._current_question = None
        return feedback

    def get_session_summary(self) -> str:
        """
        Return a plain-text summary of the learner's session performance.

        Returns:
            A friendly summary string with score, total questions, and
            percentage, suitable for display at the end of a session.
        """
        if self.total_questions == 0:
            return (
                f"You didn't answer any questions this time, "
                f"{self.learner_name}. Come back whenever you're ready! 😊"
            )

        percentage = round(self.score / self.total_questions * 100)
        stars = _score_to_stars(percentage)

        lines = [
            f"--- Session Summary for {self.learner_name} ---",
            f"Correct answers : {self.score} / {self.total_questions}",
            f"Score           : {percentage}%  {stars}",
            f"Final level     : {self.difficulty}",
        ]

        if percentage == 100:
            lines.append("\nPerfect score! 🏆 You're a math superstar!")
        elif percentage >= 80:
            lines.append("\nFantastic work today! 🌟 Keep it up!")
        elif percentage >= 50:
            lines.append("\nGood effort! 💪 Practice makes perfect!")
        else:
            lines.append(
                "\nNice try! 😊 Every question helps you learn — come back soon!"
            )

        return "\n".join(lines)

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _handle_correct(self) -> str:
        """Update state for a correct answer and return feedback text."""
        self.score += 1
        self._correct_streak += 1
        self._wrong_streak = 0

        praise = random.choice(_CORRECT_PHRASES)
        parts = [praise]

        # Check for difficulty advancement
        if self._correct_streak >= _CORRECT_STREAK_TO_ADVANCE:
            advanced = self._advance_difficulty()
            if advanced:
                self._correct_streak = 0
                parts.append(
                    f"\n🆙 You're doing so well that I'm making it a little harder! "
                    f"We're now at **{self.difficulty}** level. You've got this!"
                )

        return "\n".join(parts)

    def _handle_wrong(self, correct_answer: int) -> str:
        """Update state for a wrong answer and return feedback + explanation."""
        self._wrong_streak += 1
        self._correct_streak = 0

        q = self._current_question
        comfort = random.choice(_WRONG_PHRASES)
        explanation = _build_explanation(q, correct_answer)
        encourage = random.choice(_ENCOURAGEMENT_AFTER_WRONG)

        parts = [comfort, explanation, encourage]

        # Check for difficulty retreat
        if self._wrong_streak >= _WRONG_STREAK_TO_RETREAT:
            retreated = self._retreat_difficulty()
            if retreated:
                self._wrong_streak = 0
                parts.append(
                    f"\n⬇️ Let's try some slightly easier questions for now. "
                    f"We're now at **{self.difficulty}** level — you'll get there!"
                )

        return "\n\n".join(parts)

    def _advance_difficulty(self) -> bool:
        """
        Move to the next difficulty level if possible.

        Returns:
            True if the level actually changed, False if already at max.
        """
        idx = DIFFICULTY_LEVELS.index(self.difficulty)
        if idx < len(DIFFICULTY_LEVELS) - 1:
            self.difficulty = DIFFICULTY_LEVELS[idx + 1]
            return True
        return False

    def _retreat_difficulty(self) -> bool:
        """
        Move to the previous difficulty level if possible.

        Returns:
            True if the level actually changed, False if already at min.
        """
        idx = DIFFICULTY_LEVELS.index(self.difficulty)
        if idx > 0:
            self.difficulty = DIFFICULTY_LEVELS[idx - 1]
            return True
        return False


# ─────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────

def _parse_number(text: str) -> Optional[int]:
    """
    Extract the first integer from a string.

    Returns the integer, or None if no integer is found.
    Handles common input formats like "7", " 7 ", "the answer is 7", "7.0".
    """
    # Strip and search for an integer (including negative, though answers are >=0)
    match = re.search(r"-?\d+", text.strip())
    if match:
        return int(match.group())
    return None


def _build_explanation(question: dict, correct_answer: int) -> str:
    """
    Build a short, child-friendly step-by-step explanation.

    Args:
        question:       The question dict from generate_question().
        correct_answer: The correct numeric answer.

    Returns:
        A plain-text explanation string.
    """
    a = question["num1"]
    b = question["num2"]
    op = question["operation"]

    if op == "addition":
        symbol = "+"
        action_phrase = f"add **{b}** to **{a}**"
    else:
        symbol = "-"
        action_phrase = f"subtract **{b}** from **{a}**"

    return (
        f"Let's think through it:\n"
        f"  We need to {action_phrase}.\n"
        f"  {a} {symbol} {b} = **{correct_answer}**\n"
        f"So the answer is **{correct_answer}**."
    )


def _score_to_stars(percentage: int) -> str:
    """Return a star rating string based on percentage score."""
    if percentage >= 90:
        return "⭐⭐⭐"
    if percentage >= 60:
        return "⭐⭐"
    if percentage >= 30:
        return "⭐"
    return ""
