"""
question_generator.py — Word-problem generator for MathBuddy.

Generates simple, real-life addition and subtraction word problems
using random numbers and pre-defined templates.

Supported difficulty levels:
    "Very Easy"  — numbers 1–10, addition only
    "Easy"       — numbers 1–20, addition and subtraction
    "Medium"     — numbers 1–50, addition and subtraction

Each generated question is returned as a plain dict so it is easy to
store, display, and test without any external dependencies.

Future extension points
-----------------------
- Add a ``templates/`` folder and load YAML/JSON template files from it.
- Add a ``progress_tracker`` parameter to weight templates by past errors.
- Expand to multiplication / division for harder difficulty levels.
"""

import random
from typing import Optional

# ─────────────────────────────────────────────
# Difficulty configuration
# ─────────────────────────────────────────────

DIFFICULTY_LEVELS = ["Very Easy", "Easy", "Medium"]

_DIFFICULTY_CONFIG: dict = {
    "Very Easy": {
        "min_num": 1,
        "max_num": 10,
        "operations": ["addition"],
    },
    "Easy": {
        "min_num": 1,
        "max_num": 20,
        "operations": ["addition", "subtraction"],
    },
    "Medium": {
        "min_num": 1,
        "max_num": 50,
        "operations": ["addition", "subtraction"],
    },
}

# ─────────────────────────────────────────────
# Word-problem templates
#
# Placeholders:
#   {name}   – learner's name or a character name
#   {name2}  – a second character name
#   {a}      – first operand
#   {b}      – second operand
#   {item}   – a countable noun (plural)
# ─────────────────────────────────────────────

_NAMES = [
    "Aung", "Mya", "Ko Ko", "Hla", "Zin",
    "Nay", "Min", "Su", "Kyaw", "Thida",
]

_ITEMS = [
    "apples", "oranges", "mangoes", "bananas", "cookies",
    "pencils", "books", "stickers", "marbles", "flowers",
    "balloons", "toys", "fish", "eggs", "coins",
]

_ADDITION_TEMPLATES = [
    "{name} has {a} {item}. {name2} gives {name} {b} more {item}. "
    "How many {item} does {name} have now?",

    "There are {a} {item} on the table. Someone puts {b} more {item} on the table. "
    "How many {item} are there in all?",

    "{name} picks {a} {item} from the garden. Then {name} picks {b} more {item}. "
    "How many {item} did {name} pick altogether?",

    "A shop has {a} {item} in the morning. By noon, {b} more {item} arrive. "
    "How many {item} does the shop have now?",

    "{name} collects {a} {item} on Monday and {b} {item} on Tuesday. "
    "How many {item} does {name} have in total?",

    "{name} bakes {a} {item}. {name2} bakes {b} more {item}. "
    "How many {item} are there together?",
]

_SUBTRACTION_TEMPLATES = [
    "{name} has {a} {item}. {name} gives {b} {item} to a friend. "
    "How many {item} does {name} have left?",

    "There are {a} {item} in a basket. {b} {item} are eaten. "
    "How many {item} are still in the basket?",

    "{name} has {a} {item}. {name} uses {b} {item} for a project. "
    "How many {item} are left?",

    "A box has {a} {item}. {name} takes out {b} {item}. "
    "How many {item} remain in the box?",

    "{name} buys {a} {item} at the market. On the way home, {b} {item} fall out of the bag. "
    "How many {item} does {name} have when they get home?",

    "There are {a} {item} in the classroom. {b} {item} are given away as prizes. "
    "How many {item} are left in the classroom?",
]


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def generate_question(
    difficulty: str = "Very Easy",
    learner_name: Optional[str] = None,
) -> dict:
    """
    Generate a single word-problem question.

    Args:
        difficulty:    One of ``DIFFICULTY_LEVELS`` ("Very Easy", "Easy", "Medium").
                       Defaults to "Very Easy".
        learner_name:  If provided, may appear as the main character in the problem.

    Returns:
        A dict with the following keys:

        ``question``   (str)  – Full text of the word problem.
        ``answer``     (int)  – Correct numeric answer.
        ``operation``  (str)  – "addition" or "subtraction".
        ``difficulty`` (str)  – The difficulty level used.
        ``num1``       (int)  – First operand.
        ``num2``       (int)  – Second operand.

    Raises:
        ValueError: If ``difficulty`` is not a recognised level.
    """
    if difficulty not in _DIFFICULTY_CONFIG:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Choose from: {DIFFICULTY_LEVELS}"
        )

    cfg = _DIFFICULTY_CONFIG[difficulty]
    operation = random.choice(cfg["operations"])

    num1 = random.randint(cfg["min_num"], cfg["max_num"])
    num2 = random.randint(cfg["min_num"], cfg["max_num"])

    # For subtraction, ensure the result is non-negative (keeps it child-friendly)
    if operation == "subtraction" and num2 > num1:
        num1, num2 = num2, num1

    answer = num1 + num2 if operation == "addition" else num1 - num2

    # Pick characters and item
    name1 = learner_name if learner_name else random.choice(_NAMES)
    name2 = random.choice([n for n in _NAMES if n != name1])
    item = random.choice(_ITEMS)

    # Select and fill in a template
    templates = (
        _ADDITION_TEMPLATES if operation == "addition" else _SUBTRACTION_TEMPLATES
    )
    template = random.choice(templates)
    question_text = template.format(
        name=name1,
        name2=name2,
        a=num1,
        b=num2,
        item=item,
    )

    return {
        "question": question_text,
        "answer": answer,
        "operation": operation,
        "difficulty": difficulty,
        "num1": num1,
        "num2": num2,
    }


def get_difficulty_levels() -> list:
    """Return the ordered list of supported difficulty levels."""
    return list(DIFFICULTY_LEVELS)
