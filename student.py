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

from config import APP_CONFIG, MASTERY_ADVANCE_THRESHOLD

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
            mistake_history     TEXT    DEFAULT '[]',
            topic_mastery       TEXT    DEFAULT '{}',
            vocabulary_bank     TEXT    DEFAULT '[]',
            writing_samples     TEXT    DEFAULT '[]',
            pronunciation_scores TEXT   DEFAULT '[]',
            confusion_count     INTEGER DEFAULT 0,
            hint_usage_count    INTEGER DEFAULT 0,
            sessions_log        TEXT    DEFAULT '[]',
            grade_advancement_history TEXT DEFAULT '[]',
            pace_preference     TEXT    DEFAULT 'normal',
            interests           TEXT    DEFAULT '',
            onboarding_done     INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    # Add new columns to existing databases (migration)
    _migrate_columns(conn)


def _migrate_columns(conn: sqlite3.Connection) -> None:
    """Add any missing columns to support schema migrations."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(students)").fetchall()}
    new_columns = {
        "topic_mastery":              "TEXT DEFAULT '{}'",
        "vocabulary_bank":            "TEXT DEFAULT '[]'",
        "writing_samples":            "TEXT DEFAULT '[]'",
        "pronunciation_scores":       "TEXT DEFAULT '[]'",
        "confusion_count":            "INTEGER DEFAULT 0",
        "hint_usage_count":           "INTEGER DEFAULT 0",
        "sessions_log":               "TEXT DEFAULT '[]'",
        "grade_advancement_history":  "TEXT DEFAULT '[]'",
        "pace_preference":            "TEXT DEFAULT 'normal'",
        "interests":                  "TEXT DEFAULT ''",
        "onboarding_done":            "INTEGER DEFAULT 0",
    }
    for col, definition in new_columns.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE students ADD COLUMN {col} {definition}")
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
        "pace_preference":      "UPDATE students SET pace_preference = ? WHERE username = ?",
        "interests":            "UPDATE students SET interests = ? WHERE username = ?",
        "onboarding_done":      "UPDATE students SET onboarding_done = ? WHERE username = ?",
    }

    if field not in _FIELD_QUERIES:
        raise ValueError(f"Field '{field}' is not updatable via this function.")

    conn = _get_connection()
    try:
        conn.execute(_FIELD_QUERIES[field], (value, username))
        conn.commit()
    finally:
        conn.close()


def update_topic_mastery(username: str, topic: str, new_mastery_value: float) -> None:
    """
    Update the mastery score for a specific topic in the student's profile.

    Args:
        username: Student's username.
        topic: Topic identifier (e.g. 'short_vowels').
        new_mastery_value: Float between 0 and 1.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT topic_mastery FROM students WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return
        mastery = json.loads(row["topic_mastery"] or "{}")
        mastery[topic] = round(max(0.0, min(1.0, new_mastery_value)), 4)
        conn.execute(
            "UPDATE students SET topic_mastery = ? WHERE username = ?",
            (json.dumps(mastery), username),
        )
        conn.commit()
    finally:
        conn.close()


def get_topic_mastery(username: str) -> dict:
    """Return the student's topic mastery as a dict {topic: float}."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT topic_mastery FROM students WHERE username = ?", (username,)
        ).fetchone()
        if row:
            return json.loads(row["topic_mastery"] or "{}")
        return {}
    finally:
        conn.close()


def add_vocabulary_word(username: str, word_data: dict) -> None:
    """
    Add or update a word in the student's vocabulary bank.

    Args:
        username: Student's username.
        word_data: Dict with keys: word, definition, times_seen, times_correct, last_seen.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT vocabulary_bank FROM students WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return
        bank: list = json.loads(row["vocabulary_bank"] or "[]")
        # Update existing entry if word already in bank
        for entry in bank:
            if entry.get("word", "").lower() == word_data.get("word", "").lower():
                entry.update(word_data)
                break
        else:
            word_data.setdefault("times_seen", 1)
            word_data.setdefault("times_correct", 0)
            word_data.setdefault("last_seen", datetime.now().isoformat())
            bank.append(word_data)
        # Keep last 500 words
        bank = bank[-500:]
        conn.execute(
            "UPDATE students SET vocabulary_bank = ? WHERE username = ?",
            (json.dumps(bank), username),
        )
        conn.commit()
    finally:
        conn.close()


def get_vocabulary_bank(username: str) -> list:
    """Return the student's vocabulary bank as a list of dicts."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT vocabulary_bank FROM students WHERE username = ?", (username,)
        ).fetchone()
        if row:
            return json.loads(row["vocabulary_bank"] or "[]")
        return []
    finally:
        conn.close()


def add_writing_sample(username: str, sample_data: dict) -> None:
    """
    Append a writing sample to the student's history.

    Args:
        username: Student's username.
        sample_data: Dict with keys: prompt, response, feedback, score, timestamp.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT writing_samples FROM students WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return
        samples: list = json.loads(row["writing_samples"] or "[]")
        sample_data.setdefault("timestamp", datetime.now().isoformat())
        samples.append(sample_data)
        samples = samples[-50:]  # Keep last 50 writing samples
        conn.execute(
            "UPDATE students SET writing_samples = ? WHERE username = ?",
            (json.dumps(samples), username),
        )
        conn.commit()
    finally:
        conn.close()


def add_pronunciation_score(username: str, score_data: dict) -> None:
    """
    Append a pronunciation score entry to the student's history.

    Args:
        username: Student's username.
        score_data: Dict with keys: word, similarity_score, timestamp.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT pronunciation_scores FROM students WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return
        scores: list = json.loads(row["pronunciation_scores"] or "[]")
        score_data.setdefault("timestamp", datetime.now().isoformat())
        scores.append(score_data)
        scores = scores[-200:]  # Keep last 200 scores
        conn.execute(
            "UPDATE students SET pronunciation_scores = ? WHERE username = ?",
            (json.dumps(scores), username),
        )
        conn.commit()
    finally:
        conn.close()


def log_session(username: str, session_data: dict) -> None:
    """
    Append a session summary to the student's session log.

    Args:
        username: Student's username.
        session_data: Dict with keys: date, duration, topics, accuracy, words_learned.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT sessions_log FROM students WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return
        sessions: list = json.loads(row["sessions_log"] or "[]")
        session_data.setdefault("date", datetime.now().isoformat())
        sessions.append(session_data)
        sessions = sessions[-90:]  # Keep last 90 sessions (~3 months)
        conn.execute(
            "UPDATE students SET sessions_log = ? WHERE username = ?",
            (json.dumps(sessions), username),
        )
        conn.commit()
    finally:
        conn.close()


def get_weekly_progress(username: str) -> list:
    """
    Return accuracy data for the last 7 days from session logs.

    Returns:
        List of dicts: [{date, accuracy, topics_count}, ...]
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT sessions_log FROM students WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return []
        sessions: list = json.loads(row["sessions_log"] or "[]")
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        recent = [s for s in sessions if s.get("date", "") >= cutoff]
        return recent[-7:]
    finally:
        conn.close()


def increment_confusion_count(username: str) -> None:
    """Increment the confusion detection counter for the student."""
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE students SET confusion_count = confusion_count + 1 WHERE username = ?",
            (username,),
        )
        conn.commit()
    finally:
        conn.close()


def increment_hint_usage(username: str) -> None:
    """Increment the hint usage counter for the student."""
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE students SET hint_usage_count = hint_usage_count + 1 WHERE username = ?",
            (username,),
        )
        conn.commit()
    finally:
        conn.close()


def check_grade_readiness(username: str) -> dict:
    """
    Calculate how close the student is to advancing to the next grade.

    Returns:
        Dict with: mastery_pct, ready_to_advance, current_grade, next_grade, message.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT grade_level, topic_mastery FROM students WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return {"mastery_pct": 0.0, "ready_to_advance": False, "current_grade": 1}
        mastery: dict = json.loads(row["topic_mastery"] or "{}")
        current_grade = row["grade_level"]
        if mastery:
            avg_mastery = sum(mastery.values()) / len(mastery)
        else:
            avg_mastery = 0.0
        ready = avg_mastery >= MASTERY_ADVANCE_THRESHOLD
        pct = round(avg_mastery * 100, 1)
        if ready:
            message = (
                f"🎉 Amazing! You've mastered {pct}% of Grade {current_grade} content! "
                f"Would you like to explore Grade {current_grade + 1}? 🌟"
            )
        else:
            needed = round((MASTERY_ADVANCE_THRESHOLD - avg_mastery) * 100, 1)
            message = (
                f"You've mastered {pct}% of Grade {current_grade} content. "
                f"Keep going — just {needed}% more to reach Grade {current_grade + 1}! 🚀"
            )
        return {
            "mastery_pct": pct,
            "ready_to_advance": ready,
            "current_grade": current_grade,
            "next_grade": current_grade + 1,
            "message": message,
        }
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain Python dict."""
    d = dict(row)
    d["badges"] = json.loads(d.get("badges") or "[]")
    d["mistake_history"] = json.loads(d.get("mistake_history") or "[]")
    d["topic_mastery"] = json.loads(d.get("topic_mastery") or "{}")
    d["vocabulary_bank"] = json.loads(d.get("vocabulary_bank") or "[]")
    d["writing_samples"] = json.loads(d.get("writing_samples") or "[]")
    d["pronunciation_scores"] = json.loads(d.get("pronunciation_scores") or "[]")
    d["sessions_log"] = json.loads(d.get("sessions_log") or "[]")
    d["grade_advancement_history"] = json.loads(d.get("grade_advancement_history") or "[]")
    return d
