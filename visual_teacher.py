"""
visual_teacher.py — Generate educational visuals using Matplotlib.

All functions return a Matplotlib Figure that can be displayed
in Streamlit via `st.pyplot(fig)`.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────
# Shared styling helpers
# ─────────────────────────────────────────────
COLORS = {
    "primary":   "#4A90D9",
    "secondary": "#F5A623",
    "success":   "#7ED321",
    "vowel":     "#FF6B6B",
    "consonant": "#4ECDC4",
    "bg":        "#FAFAFA",
    "text":      "#333333",
}

VOWELS = set("aeiouAEIOU")

EXAMPLE_WORDS = {
    "A": "Apple", "B": "Ball", "C": "Cat", "D": "Dog",
    "E": "Egg", "F": "Fish", "G": "Goat", "H": "Hat",
    "I": "Igloo", "J": "Jar", "K": "Kite", "L": "Lion",
    "M": "Moon", "N": "Net", "O": "Orange", "P": "Pig",
    "Q": "Queen", "R": "Rain", "S": "Sun", "T": "Tree",
    "U": "Umbrella", "V": "Van", "W": "Wind", "X": "X-ray",
    "Y": "Yarn", "Z": "Zebra",
}


def create_alphabet_chart(highlight_letter: Optional[str] = None) -> plt.Figure:
    """
    Generate a colorful A–Z alphabet chart with example words.

    Args:
        highlight_letter: If given, that letter cell is highlighted in yellow.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(5, 6, figsize=(14, 10))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("🔤 The Alphabet", fontsize=20, fontweight="bold",
                 color=COLORS["text"], y=1.01)

    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    for i, ax in enumerate(axes.flat):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        if i < len(letters):
            letter = letters[i]
            is_vowel = letter in VOWELS
            is_highlight = highlight_letter and letter.upper() == highlight_letter.upper()

            bg_color = "#FFD700" if is_highlight else (COLORS["vowel"] if is_vowel else COLORS["consonant"])
            ax.set_facecolor(bg_color)

            # Letter (large)
            ax.text(0.5, 0.65, f"{letter}/{letter.lower()}",
                    ha="center", va="center",
                    fontsize=18, fontweight="bold", color="white")
            # Example word (small)
            word = EXAMPLE_WORDS.get(letter, "")
            ax.text(0.5, 0.25, word, ha="center", va="center",
                    fontsize=9, color="white", style="italic")

            for spine in ax.spines.values():
                spine.set_edgecolor("white")
                spine.set_linewidth(2)
        else:
            ax.set_visible(False)

    # Legend
    vowel_patch = mpatches.Patch(color=COLORS["vowel"], label="Vowels (a e i o u)")
    cons_patch = mpatches.Patch(color=COLORS["consonant"], label="Consonants")
    fig.legend(handles=[vowel_patch, cons_patch],
               loc="lower center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    return fig


def create_phonics_sound_chart(chart_type: str = "vowels") -> plt.Figure:
    """
    Generate a phonics sound reference chart.

    Args:
        chart_type: "vowels", "consonant_blends", or "digraphs".

    Returns:
        Matplotlib Figure.
    """
    data = {
        "vowels": {
            "title": "🔊 Short & Long Vowel Sounds",
            "rows": [
                ["Vowel", "Short Sound", "Example", "Long Sound", "Example"],
                ["A a", "/æ/", "cat, hat, map", "/eɪ/", "cake, name, rain"],
                ["E e", "/ɛ/", "bed, pet, hen", "/iː/", "eve, these, see"],
                ["I i", "/ɪ/", "sit, big, win", "/aɪ/", "kite, time, ride"],
                ["O o", "/ɒ/", "dog, hot, top", "/oʊ/", "rope, note, boat"],
                ["U u", "/ʌ/", "cup, bug, run", "/juː/", "cube, mule, cute"],
            ],
            "col_colors": ["#FF6B6B", "#FFB3B3", "#FFD9D9", "#FFB3B3", "#FFD9D9"],
        },
        "consonant_blends": {
            "title": "🔊 Consonant Blends",
            "rows": [
                ["Blend", "Examples", "Blend", "Examples"],
                ["bl", "blue, black, blow", "br", "bread, bridge, brush"],
                ["cl", "clap, clock, clean", "cr", "crab, cry, crown"],
                ["fl", "flag, fly, flower", "fr", "frog, free, front"],
                ["gl", "glad, glue, glow", "gr", "grass, great, green"],
                ["pl", "play, plan, plate", "pr", "pray, price, proud"],
                ["sl", "slap, slow, slim", "sk", "sky, skip, skin"],
                ["sm", "smile, smoke, small", "sn", "snap, snow, snake"],
                ["sp", "spin, spot, spell", "st", "stop, star, stamp"],
                ["sw", "swim, swing, sweet", "tr", "tree, train, truck"],
            ],
            "col_colors": ["#4ECDC4", "#A8EDEA", "#4ECDC4", "#A8EDEA"],
        },
        "digraphs": {
            "title": "🔊 Digraphs (2 Letters → 1 Sound)",
            "rows": [
                ["Digraph", "Sound", "Examples"],
                ["ch", "/tʃ/", "chair, cheese, chicken, church"],
                ["sh", "/ʃ/", "ship, shop, shell, brush"],
                ["th", "/θ/ or /ð/", "think, this, three, the"],
                ["wh", "/w/", "when, what, where, while"],
                ["ph", "/f/", "phone, photo, elephant"],
                ["ck", "/k/", "duck, back, kick, lock"],
                ["ng", "/ŋ/", "sing, ring, strong, king"],
                ["qu", "/kw/", "queen, quiet, quick, quiz"],
            ],
            "col_colors": ["#F5A623", "#FFD9A0", "#FFECD0"],
        },
    }

    chart = data.get(chart_type, data["vowels"])
    rows = chart["rows"]
    title = chart["title"]
    col_colors = chart["col_colors"]

    n_cols = len(rows[0])
    n_rows = len(rows)

    fig, ax = plt.subplots(figsize=(13, max(4, n_rows * 0.7 + 1.5)))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, fontweight="bold",
                 color=COLORS["text"], pad=15)

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(col_colors[j % len(col_colors)])
        cell.set_text_props(fontweight="bold", color="white")

    # Alternate row shading
    for i in range(1, n_rows - 1):
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_facecolor("#F0F9FF" if i % 2 == 0 else "white")

    plt.tight_layout()
    return fig


def create_word_family_diagram(pattern: str = "-at") -> plt.Figure:
    """
    Generate a word family diagram showing words with a shared rime.

    Args:
        pattern: The rime pattern, e.g. "-at", "-ig", "-ot".

    Returns:
        Matplotlib Figure.
    """
    WORD_FAMILIES: dict = {
        "-at": ["cat", "bat", "hat", "mat", "rat", "sat", "fat", "pat"],
        "-an": ["can", "ban", "fan", "man", "pan", "ran", "tan", "van"],
        "-ig": ["big", "dig", "fig", "jig", "pig", "rig", "wig", "twig"],
        "-ot": ["dot", "got", "hot", "lot", "not", "pot", "rot", "tot"],
        "-un": ["bun", "fun", "gun", "nun", "run", "sun", "pun", "spun"],
        "-it": ["bit", "fit", "hit", "kit", "lit", "pit", "sit", "wit"],
        "-op": ["cop", "hop", "mop", "pop", "top", "crop", "drop", "shop"],
        "-in": ["bin", "fin", "kin", "pin", "sin", "tin", "win", "chin"],
    }

    words = WORD_FAMILIES.get(pattern, WORD_FAMILIES["-at"])
    rime = pattern.lstrip("-")
    title = f"📝 Word Family: '{pattern}' Words"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16, fontweight="bold", color=COLORS["text"], pad=12)

    # Center hub
    hub = plt.Circle((5, 3), 0.9, color=COLORS["secondary"], zorder=5)
    ax.add_patch(hub)
    ax.text(5, 3, f"_{rime}", ha="center", va="center",
            fontsize=18, fontweight="bold", color="white", zorder=6)

    # Arrange words in a circle
    n = len(words)
    for i, word in enumerate(words):
        angle = 2 * np.pi * i / n
        radius = 2.2
        x = 5 + radius * np.cos(angle)
        y = 3 + radius * np.sin(angle)

        # Draw connection line
        ax.plot([5 + 0.9 * np.cos(angle), x - 0.5 * np.cos(angle)],
                [3 + 0.9 * np.sin(angle), y - 0.5 * np.sin(angle)],
                color=COLORS["primary"], linewidth=1.5, zorder=3)

        # Onset (consonant part)
        onset = word[:-len(rime)] if word.endswith(rime) else word
        ax.text(x - 0.15, y, onset, ha="right", va="center",
                fontsize=13, fontweight="bold", color=COLORS["primary"])
        ax.text(x - 0.1, y, rime, ha="left", va="center",
                fontsize=13, color=COLORS["secondary"])

    plt.tight_layout()
    return fig


def create_progress_chart(
    sessions: list[dict],
    student_name: str = "Student",
) -> plt.Figure:
    """
    Generate a line chart showing accuracy over recent sessions.

    Args:
        sessions: List of dicts with 'session' (label) and 'accuracy' (float 0-100).
        student_name: Name displayed in the chart title.

    Returns:
        Matplotlib Figure.
    """
    if not sessions:
        sessions = [{"session": "Start", "accuracy": 0}]

    labels = [s.get("session", f"#{i+1}") for i, s in enumerate(sessions)]
    scores = [s.get("accuracy", 0) for s in sessions]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    ax.plot(labels, scores, marker="o", linewidth=2.5,
            markersize=8, color=COLORS["primary"])
    ax.fill_between(range(len(labels)), scores, alpha=0.15, color=COLORS["primary"])

    ax.set_ylim(0, 105)
    ax.axhline(80, color=COLORS["success"], linestyle="--",
               linewidth=1.5, label="Target (80%)")

    ax.set_title(f"📈 {student_name}'s Accuracy Over Time",
                 fontsize=14, fontweight="bold", color=COLORS["text"])
    ax.set_xlabel("Session", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig


def create_mastery_bar_chart(mastery_dict: dict, subject: str = "Phonics") -> plt.Figure:
    """
    Create a horizontal bar chart showing topic mastery.

    Args:
        mastery_dict: {topic_name: mastery_pct} from adaptive_path.
        subject: Subject label for the title.

    Returns:
        Matplotlib Figure.
    """
    if not mastery_dict:
        return _placeholder_fig("No mastery data yet. Start a lesson!")

    topics = list(mastery_dict.keys())
    values = [mastery_dict[t] for t in topics]
    colors = [COLORS["success"] if v >= 80 else
              COLORS["secondary"] if v >= 50 else
              COLORS["vowel"] for v in values]

    fig, ax = plt.subplots(figsize=(10, max(3, len(topics) * 0.55)))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    bars = ax.barh(topics, values, color=colors, height=0.6)
    ax.set_xlim(0, 110)
    ax.axvline(80, color=COLORS["success"], linestyle="--",
               linewidth=1.5, label="Mastery Target (80%)")
    ax.set_title(f"🏆 {subject} Topic Mastery", fontsize=14,
                 fontweight="bold", color=COLORS["text"])
    ax.set_xlabel("Mastery %", fontsize=11)
    ax.legend(fontsize=10)

    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=9, color=COLORS["text"])

    plt.tight_layout()
    return fig


def _placeholder_fig(message: str) -> plt.Figure:
    """Return a simple placeholder figure with a text message."""
    fig, ax = plt.subplots(figsize=(6, 2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=13, color=COLORS["text"], style="italic",
            transform=ax.transAxes)
    return fig


# ─────────────────────────────────────────────
# Learning Analytics Visualizations
# ─────────────────────────────────────────────

def generate_topic_mastery_chart(mastery_data: dict, subject: str = "") -> plt.Figure:
    """
    Horizontal bar chart of topic mastery percentages, color-coded green/yellow/red.

    Args:
        mastery_data: {topic_name: mastery_float (0-1)} dict.
        subject: Subject label for title.

    Returns:
        Matplotlib Figure.
    """
    if not mastery_data:
        return _placeholder_fig("No mastery data yet. Start a lesson!")

    topics = list(mastery_data.keys())
    values = [mastery_data[t] * 100 for t in topics]
    bar_colors = [
        COLORS["success"] if v >= 80 else
        COLORS["secondary"] if v >= 50 else
        COLORS["vowel"]
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(10, max(3, len(topics) * 0.6)))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    bars = ax.barh(topics, values, color=bar_colors, height=0.55)
    ax.set_xlim(0, 115)
    ax.axvline(80, color=COLORS["success"], linestyle="--",
               linewidth=1.5, label="Mastery Target (80%)")
    ax.axvline(50, color=COLORS["secondary"], linestyle=":",
               linewidth=1, label="Minimum (50%)")

    title = f"🏆 Topic Mastery" + (f" — {subject}" if subject else "")
    ax.set_title(title, fontsize=14, fontweight="bold", color=COLORS["text"])
    ax.set_xlabel("Mastery %", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")

    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=9, color=COLORS["text"])

    plt.tight_layout()
    return fig


def generate_weekly_progress_chart(
    daily_accuracy_data: list,
    student_name: str = "Student",
) -> plt.Figure:
    """
    Line chart showing accuracy trend over past 7 days/sessions.

    Args:
        daily_accuracy_data: List of dicts [{date, accuracy}, ...].
        student_name: For chart title.

    Returns:
        Matplotlib Figure.
    """
    if not daily_accuracy_data:
        daily_accuracy_data = [{"date": "Today", "accuracy": 0}]

    labels = [s.get("date", f"Day {i+1}")[:10] for i, s in enumerate(daily_accuracy_data)]
    scores = [s.get("accuracy", 0) for s in daily_accuracy_data]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    ax.plot(range(len(labels)), scores, marker="o", linewidth=2.5,
            markersize=8, color=COLORS["primary"], zorder=3)
    ax.fill_between(range(len(labels)), scores, alpha=0.12, color=COLORS["primary"])

    ax.set_ylim(0, 105)
    ax.axhline(80, color=COLORS["success"], linestyle="--",
               linewidth=1.5, label="Target (80%)")

    if len(scores) >= 2:
        trend = "📈 Improving!" if scores[-1] > scores[0] else (
            "📉 Let's push harder!" if scores[-1] < scores[0] else "➡️ Steady progress"
        )
        ax.set_title(
            f"📊 {student_name}'s Weekly Progress  {trend}",
            fontsize=13, fontweight="bold", color=COLORS["text"]
        )
    else:
        ax.set_title(f"📊 {student_name}'s Weekly Progress",
                     fontsize=13, fontweight="bold", color=COLORS["text"])

    ax.set_xlabel("Session / Day", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    return fig


def generate_grade_readiness_gauge(mastery_percentage: float, current_grade: int) -> plt.Figure:
    """
    Linear progress gauge showing how close student is to advancing grades.

    Args:
        mastery_percentage: 0-100 float.
        current_grade: Current grade number.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(9, 2.5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 90, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "90% ★", "100%"], fontsize=10)

    # Background bar
    ax.barh(0.5, 100, height=0.35, color="#E0E0E0", left=0, zorder=1)

    # Progress bar
    bar_color = (COLORS["success"] if mastery_percentage >= 90 else
                 COLORS["secondary"] if mastery_percentage >= 50 else
                 COLORS["vowel"])
    ax.barh(0.5, mastery_percentage, height=0.35, color=bar_color, left=0, zorder=2)

    # Threshold marker
    ax.axvline(90, color=COLORS["success"], linestyle="--", linewidth=1.5,
               label="Advance threshold (90%)", zorder=3)

    ax.text(mastery_percentage / 2, 0.5, f"{mastery_percentage:.0f}%",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="white", zorder=4)

    if mastery_percentage >= 90:
        title = f"🎉 Grade {current_grade} Mastery: Ready for Grade {current_grade + 1}!"
    else:
        title = f"🎓 Grade {current_grade} Readiness: {mastery_percentage:.0f}%"

    ax.set_title(title, fontsize=13, fontweight="bold", color=COLORS["text"], pad=10)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    return fig


def generate_session_summary_card(session_data: dict) -> plt.Figure:
    """
    Visual summary card of a learning session.

    Args:
        session_data: Dict with keys: topics, correct, total, words_learned,
                      duration_min, encouragement.

    Returns:
        Matplotlib Figure.
    """
    topics = session_data.get("topics", [])
    correct = session_data.get("correct", 0)
    total = session_data.get("total", 0)
    words_learned = session_data.get("words_learned", 0)
    duration = session_data.get("duration_min", 0)
    pct = int(correct / total * 100) if total > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#F0F8FF")
    ax.set_facecolor("#F0F8FF")
    ax.set_axis_off()

    # Title
    ax.text(0.5, 0.92, "✨ Session Complete! ✨", ha="center", va="top",
            fontsize=16, fontweight="bold", color=COLORS["primary"],
            transform=ax.transAxes)

    stats_text = (
        f"Questions:  {correct}/{total} correct ({pct}%)\n"
        f"Topics covered:  {', '.join(topics) if topics else 'General practice'}\n"
        f"New vocabulary:  {words_learned} words\n"
        f"Time spent:  {duration} minutes"
    )
    ax.text(0.5, 0.60, stats_text, ha="center", va="center",
            fontsize=11, color=COLORS["text"], transform=ax.transAxes,
            linespacing=2.0, family="monospace")

    encouragement = session_data.get("encouragement", "Great work today! Keep it up! 🌟")
    ax.text(0.5, 0.10, encouragement, ha="center", va="bottom",
            fontsize=12, color=COLORS["success"], fontweight="bold",
            transform=ax.transAxes, style="italic")

    plt.tight_layout()
    return fig


def generate_vocabulary_word_cloud(vocabulary_list: list) -> plt.Figure:
    """
    Display learned vocabulary as a styled grid (word cloud alternative without wordcloud lib).

    Args:
        vocabulary_list: List of dicts [{word, times_seen, times_correct}, ...].

    Returns:
        Matplotlib Figure.
    """
    if not vocabulary_list:
        return _placeholder_fig("No vocabulary learned yet. Start a vocabulary lesson!")

    # Sort by times_seen descending (most practiced = larger)
    sorted_words = sorted(vocabulary_list, key=lambda w: w.get("times_seen", 1), reverse=True)[:30]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_axis_off()
    ax.set_title("📖 Your Vocabulary Bank", fontsize=14, fontweight="bold",
                 color=COLORS["text"])

    max_seen = max((w.get("times_seen", 1) for w in sorted_words), default=1)

    x_positions = np.linspace(0.05, 0.95, 6)
    y_positions = np.linspace(0.85, 0.10, 5)
    positions = [(x, y) for y in y_positions for x in x_positions]

    for i, word_data in enumerate(sorted_words[:len(positions)]):
        word = word_data.get("word", "")
        times_seen = word_data.get("times_seen", 1)
        times_correct = word_data.get("times_correct", 0)
        accuracy = times_correct / times_seen if times_seen > 0 else 0
        font_size = 9 + int(times_seen / max_seen * 10)
        color = (COLORS["success"] if accuracy >= 0.8 else
                 COLORS["secondary"] if accuracy >= 0.5 else
                 COLORS["vowel"])
        if i < len(positions):
            ax.text(positions[i][0], positions[i][1], word,
                    ha="center", va="center", fontsize=font_size,
                    color=color, fontweight="bold" if times_seen > 3 else "normal",
                    transform=ax.transAxes)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Onboarding Interest Visual
# ─────────────────────────────────────────────

def create_interest_visual(
    interest: str,
    emoji: str,
    careers: list,
    matched_keyword: str,
) -> plt.Figure:
    """
    Create a colourful career-pathway visual for the onboarding interest step.

    Shows the student's interest in the centre, connected to possible future
    careers by a spoke diagram, with an English-learning bridge note.

    Args:
        interest: The student's raw interest text.
        emoji: An emoji representing the interest.
        careers: List of career strings.
        matched_keyword: The normalised keyword matched from the interest.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#FFF8F0")
    ax.set_facecolor("#FFF8F0")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(
        0.5, 0.96,
        f"Your Interest \u2192 Your Future! {emoji}",
        ha="center", va="top",
        fontsize=16, fontweight="bold",
        color=COLORS["primary"],
    )

    # ── Centre circle (interest) ─────────────────────────────────────────────
    centre_x, centre_y = 0.5, 0.52
    centre_circle = plt.Circle(
        (centre_x, centre_y), 0.13,
        color=COLORS["secondary"], zorder=3,
    )
    ax.add_patch(centre_circle)
    # Emoji on top of circle
    ax.text(
        centre_x, centre_y + 0.02,
        emoji,
        ha="center", va="center",
        fontsize=26, zorder=4,
    )
    ax.text(
        centre_x, centre_y - 0.06,
        matched_keyword.title(),
        ha="center", va="center",
        fontsize=10, fontweight="bold",
        color="white", zorder=4,
    )

    # ── Career spokes ────────────────────────────────────────────────────────
    num_careers = min(len(careers), 5)
    angles = np.linspace(0, 2 * np.pi, num_careers, endpoint=False)
    spoke_len = 0.30
    career_colors = [
        COLORS["primary"], COLORS["success"], "#9B59B6",
        "#E74C3C", "#1ABC9C",
    ]

    for i, angle in enumerate(angles):
        cx = centre_x + spoke_len * np.cos(angle)
        cy = centre_y + spoke_len * np.sin(angle)

        # Line from centre to career bubble
        ax.plot(
            [centre_x, cx], [centre_y, cy],
            color="#CCCCCC", linewidth=1.5, zorder=1,
        )

        # Career bubble
        bubble = plt.Circle(
            (cx, cy), 0.09,
            color=career_colors[i % len(career_colors)],
            alpha=0.85, zorder=2,
        )
        ax.add_patch(bubble)

        career_label = careers[i]
        # Split long labels onto two lines for better fit in bubble
        if len(career_label) > 12:
            words = career_label.split()
            mid = max(1, len(words) // 2)
            career_label = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])

        ax.text(
            cx, cy,
            career_label,
            ha="center", va="center",
            fontsize=8, fontweight="bold",
            color="white", zorder=3,
        )

    # ── English bridge banner ─────────────────────────────────────────────────
    banner_y = 0.08
    banner_rect = mpatches.FancyBboxPatch(
        (0.05, banner_y - 0.04), 0.90, 0.09,
        boxstyle="round,pad=0.01",
        facecolor=COLORS["primary"], alpha=0.15,
        edgecolor=COLORS["primary"], linewidth=1.5,
    )
    ax.add_patch(banner_rect)
    ax.text(
        0.5, banner_y,
        "\U0001f4d6 Learning English \u2192 Phonics \u00b7 Reading \u00b7 Spelling \u2192 Unlocks ALL these futures!",
        ha="center", va="center",
        fontsize=9, color=COLORS["primary"],
        fontweight="bold",
    )

    plt.tight_layout()
    return fig
