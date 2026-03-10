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
