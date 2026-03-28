"""
badge_service.py — Badge award logic for student milestones.

Centralises the badge-checking logic that was previously scattered inside
learning_orchestrator.py.  All badge thresholds are defined here so they can
be adjusted in one place.

Usage::

    from badge_service import check_and_award_badges

    check_and_award_badges(username, stats)
"""

import student as student_db


def check_and_award_badges(username: str, stats: dict) -> None:
    """
    Evaluate badge milestones against the student's current stats and award
    any that have been reached.

    Args:
        username: Student's unique username.
        stats:    Stats dict as returned by ``student_db.get_stats()``.
    """
    lessons  = stats.get("total_lessons", 0)
    correct  = stats.get("correct_answers", 0)
    accuracy = stats.get("accuracy_pct", 0)
    streak   = stats.get("streak_days", 0)

    # Each tuple is (condition, badge_label).
    milestones = [
        (lessons  >= 1,                     "First Lesson! 🌟"),
        (lessons  >= 10,                    "10 Lessons Complete! 📚"),
        (correct  >= 10,                    "10 Correct Answers! ✅"),
        (correct  >= 50,                    "50 Correct Answers! 🏆"),
        (accuracy >= 80 and lessons >= 5,   "High Achiever! 🎯"),
        (streak   >= 3,                     "3-Day Streak! 🔥"),
        (streak   >= 7,                     "One Week Strong! 💪"),
    ]

    for condition, badge in milestones:
        if condition:
            student_db.award_badge(username, badge)
