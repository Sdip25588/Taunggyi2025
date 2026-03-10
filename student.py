"""
student.py — Student profile management using SQLite.

Handles creating/loading student profiles, tracking progress,
recording mistakes, managing streaks and badges.
"""

import sqlite3
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from config import APP_CONFIG

DB_PATH: str = APP_CONFIG["db_path"]


def _get_connection() -> sqlite3.Connection:
    """Return a SQLite connection, creating the DB file and tables if needed."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_tables(conn)
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create tables if they don't already exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            username            TEXT    UNIQUE NOT NULL,
            grade_level         INTEGER DEFAULT 1,
            created_at          TEXT    DEFAULT (datetime('now')),
            last_active         TEXT    DEFAULT (datetime('now')),
            total_lessons       INTEGER DEFAULT 0,
            total_quizzes       INTEGER DEFAULT 0,
            correct_answers     INTEGER DEFAULT 0,
            wrong_answers       INTEGER DEFAULT 0,
            current_subject     TEXT    DEFAULT 'Phonics',
            current_lesson_index INTEGER DEFAULT 0,
            difficulty_level    TEXT    DEFAULT 'Beginner',
            streak_days         INTEGER DEFAULT 0,
            last_streak_date    TEXT    DEFAULT '',
            badges              TEXT    DEFAULT '[]',
            mistake_history     TEXT    DEFAULT '[]'
        )
    """)
    conn.commit()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def get_or_create_student(username: str) -> dict:
    """
    Load an existing student profile or create a new one.

    Args:
        username: The student's chosen display name.

    Returns:
        A dict with all student fields.
    """
    username = username.strip()
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM students WHERE username = ?", (username,)
        ).fetchone()

        if row is None:
            conn.execute(
                "INSERT INTO students (username) VALUES (?)", (username,)
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM students WHERE username = ?", (username,)
            ).fetchone()

        # Update last_active and handle streak
        today = date.today().isoformat()
        last_streak = row["last_streak_date"]
        streak = row["streak_days"]
        yesterday = (date.today() - timedelta(days=1)).isoformat()

        if last_streak == today:
            pass  # Already logged in today
        elif last_streak == yesterday:
            streak += 1  # Consecutive day
            conn.execute(
                "UPDATE students SET last_active=?, streak_days=?, last_streak_date=? WHERE username=?",
                (datetime.now().isoformat(), streak, today, username),
            )
        else:
            streak = 1  # Reset streak
            conn.execute(
                "UPDATE students SET last_active=?, streak_days=1, last_streak_date=? WHERE username=?",
                (datetime.now().isoformat(), today, username),
            )
        conn.commit()

        return _row_to_dict(conn.execute(
            "SELECT * FROM students WHERE username = ?", (username,)
        ).fetchone())
    finally:
        conn.close()


def update_progress(username: str, quiz_result: dict) -> None:
    """
    Update a student's quiz statistics.

    Args:
        username: Student's username.
        quiz_result: Dict with keys 'correct' (bool) and optionally 'subject'.
    """
    correct: bool = quiz_result.get("correct", False)
    conn = _get_connection()
    try:
        if correct:
            conn.execute(
                """UPDATE students
                   SET total_quizzes = total_quizzes + 1,
                       correct_answers = correct_answers + 1,
                       last_active = datetime('now')
                   WHERE username = ?""",
                (username,),
            )
        else:
            conn.execute(
                """UPDATE students
                   SET total_quizzes = total_quizzes + 1,
                       wrong_answers = wrong_answers + 1,
                       last_active = datetime('now')
                   WHERE username = ?""",
                (username,),
            )
        conn.commit()
    finally:
        conn.close()


def increment_lesson(username: str) -> None:
    """Increment lesson counter when a new lesson is started."""
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE students SET total_lessons = total_lessons + 1 WHERE username = ?",
            (username,),
        )
        conn.commit()
    finally:
        conn.close()


def get_mistake_history(username: str) -> list:
    """
    Return the student's mistake history as a list of dicts.

    Each entry: {word, correct_answer, student_answer, timestamp}
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT mistake_history FROM students WHERE username = ?", (username,)
        ).fetchone()
        if row:
            return json.loads(row["mistake_history"] or "[]")
        return []
    finally:
        conn.close()


def add_mistake(username: str, mistake_data: dict) -> None:
    """
    Append a mistake entry to the student's history.

    Args:
        username: Student's username.
        mistake_data: Dict with keys: word, correct_answer, student_answer, type.
    """
    conn = _get_connection()
    try:
        history = get_mistake_history(username)
        mistake_data["timestamp"] = datetime.now().isoformat()
        history.append(mistake_data)
        # Keep only last 100 mistakes to limit DB size
        history = history[-100:]
        conn.execute(
            "UPDATE students SET mistake_history = ? WHERE username = ?",
            (json.dumps(history), username),
        )
        conn.commit()
    finally:
        conn.close()


def get_stats(username: str) -> dict:
    """
    Return summary statistics for a student.

    Returns:
        Dict with: accuracy_pct, total_lessons, total_quizzes,
                   correct_answers, wrong_answers, streak_days, badges.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM students WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return {}

        total = row["correct_answers"] + row["wrong_answers"]
        accuracy = round((row["correct_answers"] / total * 100), 1) if total > 0 else 0.0

        return {
            "accuracy_pct": accuracy,
            "total_lessons": row["total_lessons"],
            "total_quizzes": row["total_quizzes"],
            "correct_answers": row["correct_answers"],
            "wrong_answers": row["wrong_answers"],
            "streak_days": row["streak_days"],
            "badges": json.loads(row["badges"] or "[]"),
            "difficulty_level": row["difficulty_level"],
            "current_subject": row["current_subject"],
            "grade_level": row["grade_level"],
        }
    finally:
        conn.close()


def award_badge(username: str, badge_name: str) -> bool:
    """
    Award a badge to a student if they don't already have it.

    Args:
        username: Student's username.
        badge_name: Human-readable badge label (e.g., "First Lesson! 🌟").

    Returns:
        True if badge was newly awarded, False if already owned.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT badges FROM students WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return False

        badges: list = json.loads(row["badges"] or "[]")
        if badge_name in badges:
            return False

        badges.append(badge_name)
        conn.execute(
            "UPDATE students SET badges = ? WHERE username = ?",
            (json.dumps(badges), username),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def update_student_field(username: str, field: str, value) -> None:
    """
    Update a single field for a student.

    Args:
        username: Student's username.
        field: Column name to update (validated against allowed fields).
        value: New value.
    """
    # Explicit mapping prevents SQL injection — only these fields are updatable
    _FIELD_QUERIES: dict = {
        "grade_level":          "UPDATE students SET grade_level = ? WHERE username = ?",
        "current_subject":      "UPDATE students SET current_subject = ? WHERE username = ?",
        "current_lesson_index": "UPDATE students SET current_lesson_index = ? WHERE username = ?",
        "difficulty_level":     "UPDATE students SET difficulty_level = ? WHERE username = ?",
    }

    if field not in _FIELD_QUERIES:
        raise ValueError(f"Field '{field}' is not updatable via this function.")

    conn = _get_connection()
    try:
        conn.execute(_FIELD_QUERIES[field], (value, username))
        conn.commit()
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain Python dict."""
    d = dict(row)
    d["badges"] = json.loads(d.get("badges") or "[]")
    d["mistake_history"] = json.loads(d.get("mistake_history") or "[]")
    return d
