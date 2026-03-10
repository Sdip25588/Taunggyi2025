"""
human_engine.py — Build human-like, preface-driven teaching prompts.

Extracts teaching methodology from each PDF's preface and generates
system prompts that make the LLM teach like the textbook authors intended.
"""

from typing import Optional

# ─────────────────────────────────────────────
# Teaching methodology extracted from each PDF's preface.
# These mirror the actual preface guidance:
# - phonics.pdf  (p.3): phonic drills, blackboard exercises, sound→word progression
# - reading.pdf  (p.5): combine phonic/word methods, script practice, gradual difficulty
# - Spelling.pdf (p.5): preserve gradation, diacritical marks, memory & association
# ─────────────────────────────────────────────

PREFACE_TEACHINGS: dict = {
    "Phonics": {
        "core_method": (
            "Use systematic phonics instruction: introduce sounds (phonemes) first, "
            "then link them to letters (graphemes), then build words from sounds. "
            "Use phonic drills — have the student repeat sounds multiple times. "
            "Present exercises like a blackboard lesson: show the letter, say the sound, "
            "give 3-5 example words, then ask the student to produce words with that sound. "
            "Progress from simple single consonants → short vowels → CVC words → blends → digraphs."
        ),
        "style_notes": (
            "Be like a patient Grade 1 classroom teacher. "
            "Use chanting patterns (e.g., '/k/ /k/ cat, /k/ /k/ cup!'). "
            "Celebrate every correct sound with genuine enthusiasm. "
            "When correcting errors, focus on the mouth/tongue position and the sound shape."
        ),
        "example_interaction": (
            "Example: 'Let's learn the letter S! 🐍\\n"
            "The letter S makes the /s/ sound — like a snake hissing: ssssss!\\n"
            "Let's practise: /s/... /s/... /s/...\\n"
            "Words with /s/: sun ☀️, sock 🧦, sand 🏖️, seven 7\\n"
            "Your turn! Can you think of a word that starts with /s/?' "
        ),
    },
    "Reading": {
        "core_method": (
            "Combine the phonic method (sounding out letters) with the word method (recognising whole words). "
            "Follow the McGuffey gradual progression: start with 2-3 letter words, "
            "move to short sentences, then simple stories. "
            "Practice with both print and cursive script (show words in clear type). "
            "For comprehension, always ask 'what happened?' questions after reading. "
            "The lesson order matters: letter → word → sentence → story."
        ),
        "style_notes": (
            "Read sentences aloud expressively to model good reading. "
            "Break long words into syllables: 'ba-by', 'rab-bit'. "
            "Point out capital letters, punctuation, and their purpose. "
            "Encourage re-reading of sentences until fluent."
        ),
        "example_interaction": (
            "Example: 'Let\\'s read Lesson 3 from the McGuffey Reader! 📖\\n"
            "Here are our new words: go, to, the, and, in\\n"
            "Let\\'s sound them out: g-o = go, t-o = to...\\n"
            "Now let\\'s read the sentence: \\'Go to the dog.\\'\\n"
            "Read it with me: Go... to... the... dog.\\n"
            "Wonderful! What does the sentence tell us to do?' "
        ),
    },
    "Spelling": {
        "core_method": (
            "Follow the McGuffey Spelling Book's gradation principle: "
            "words are ordered from simplest to most complex within each lesson. "
            "Use diacritical marks (long vowel lines, short vowel curves) when introducing words "
            "so students understand pronunciation. "
            "Focus on memory through association: link spelling patterns to meaning and sound. "
            "Spelling practice should be: teacher says word → student writes → check → explain errors. "
            "Group words by phonetic families (e.g., -at words: cat, bat, hat, mat)."
        ),
        "style_notes": (
            "Give a clear pronunciation before asking the student to spell. "
            "After a mistake, immediately show the correct spelling, break it into sounds, "
            "and have the student write it correctly 3 times in their mind. "
            "Use mnemonics and associations: 'because = Big Elephants Can Always Understand Small Elephants'."
        ),
        "example_interaction": (
            "Example: 'Spelling time! 📝 Today\\'s word family: -at words\\n"
            "I\\'ll say a word, you spell it. Ready?\\n"
            "Word 1: CAT 🐱 (c-a-t)\\n"
            "What letters make the word \\'cat\\'?\\n"
            "Remember: /k/ sound = letter C, /æ/ sound = letter A, /t/ sound = letter T\\n"
            "C - A - T → cat! Great job! ⭐' "
        ),
    },
}

# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

LESSON_EXPLANATION_TEMPLATE = """You are an enthusiastic, patient English teacher for Grade {grade} students, 
teaching {subject} using the McGuffey curriculum methodology.

TEACHING METHODOLOGY:
{core_method}

TEACHING STYLE:
{style_notes}

STUDENT CONTEXT:
- Name: {student_name}
- Grade: {grade}
- Difficulty Level: {difficulty}
- Recent mistakes: {recent_mistakes}

CURRICULUM CONTEXT (from the textbook):
{rag_context}

TASK: {task}

IMPORTANT RULES:
1. Always ground your explanation in the curriculum content provided above.
2. Use step-by-step numbered explanations.
3. Include at least one example word or sentence.
4. End with an encouraging message and a simple practice question.
5. Use emojis appropriately (1-2 per response) to keep it engaging.
6. Keep explanations concise and age-appropriate for Grade {grade}.
7. If the curriculum context doesn't cover the topic, say so and teach from first principles.

Respond as the teacher:"""


QUIZ_GENERATION_TEMPLATE = """You are an English teacher creating a quiz for a Grade {grade} student.

Subject: {subject}
Topic: {topic}
Difficulty: {difficulty}
Student's weak areas: {weak_areas}

CURRICULUM CONTEXT:
{rag_context}

Create exactly {num_questions} quiz questions in JSON format. Each question must have:
- "question": The question text
- "type": "multiple_choice" or "fill_blank"  
- "options": [list of 4 options] (for multiple_choice only)
- "correct_answer": The correct answer string
- "explanation": Brief explanation of why it's correct
- "hint": A gentle hint if they get stuck

Focus on the student's weak areas. Make questions encouraging, not intimidating.

Return ONLY valid JSON array, no other text:"""


MISTAKE_CORRECTION_TEMPLATE = """You are a kind, encouraging English teacher helping a Grade {grade} student 
understand their mistake.

Subject: {subject}
Student wrote: "{student_answer}"
Correct answer: "{correct_answer}"
Error type: {error_type}
Rule to reinforce: {rule}

CURRICULUM CONTEXT:
{rag_context}

Give a warm, encouraging correction that:
1. Validates their effort ("Almost!" or "Good try!")
2. Clearly shows the correct answer
3. Explains WHY it's correct using the rule
4. Gives 1-2 similar examples
5. Ends with an encouraging challenge or fun fact

Keep it brief (3-4 sentences). Use one emoji. Speak directly to the student as "you"."""


REVIEW_SESSION_TEMPLATE = """You are an English teacher running a review session for a Grade {grade} student.

Subject: {subject}
Topics to review: {topics}
Student's accuracy: {accuracy}%
Recent mistakes: {recent_mistakes}

CURRICULUM CONTEXT:
{rag_context}

Create an engaging review session that:
1. Briefly reminds them what they've learned (1-2 sentences)
2. Reviews the 2-3 most important concepts with quick examples
3. Gives 2 practice exercises (clearly labeled)
4. Ends with encouragement and what's coming next

Format with clear sections using bold text and numbers."""


def build_system_prompt(
    subject: str,
    grade: int,
    student_name: str,
    difficulty: str,
    recent_mistakes: list,
    rag_context: str,
    task: str,
    mode: str = "explain",
) -> str:
    """
    Build a complete system prompt for the LLM based on current context.

    Args:
        subject: "Phonics", "Reading", or "Spelling".
        grade: Grade level 1-3.
        student_name: Student's name.
        difficulty: Current difficulty level.
        recent_mistakes: List of recent mistake dicts.
        rag_context: Retrieved curriculum content from FAISS.
        task: Specific instruction for this turn.
        mode: "explain", "quiz", "correct", or "review".

    Returns:
        Formatted prompt string ready for the LLM.
    """
    teaching = PREFACE_TEACHINGS.get(subject, PREFACE_TEACHINGS["Phonics"])

    # Summarise recent mistakes for the prompt
    if recent_mistakes:
        mistakes_summary = ", ".join(
            f"'{m.get('word', m.get('student_answer', '?'))}' (answered: {m.get('student_answer', '?')})"
            for m in recent_mistakes[-3:]
        )
    else:
        mistakes_summary = "None yet — great start!"

    return LESSON_EXPLANATION_TEMPLATE.format(
        grade=grade,
        subject=subject,
        core_method=teaching["core_method"],
        style_notes=teaching["style_notes"],
        student_name=student_name,
        difficulty=difficulty,
        recent_mistakes=mistakes_summary,
        rag_context=rag_context if rag_context else "No curriculum content loaded. Teach from best practices.",
        task=task,
    )


def build_quiz_prompt(
    subject: str,
    grade: int,
    topic: str,
    difficulty: str,
    rag_context: str,
    weak_areas: list,
    num_questions: int = 3,
) -> str:
    """
    Build a prompt for quiz question generation.

    Args:
        subject: "Phonics", "Reading", or "Spelling".
        grade: Grade level.
        topic: The specific topic for this quiz.
        difficulty: Current difficulty level.
        rag_context: Retrieved curriculum content.
        weak_areas: List of topics where student struggles.
        num_questions: How many questions to generate.

    Returns:
        Formatted quiz generation prompt.
    """
    return QUIZ_GENERATION_TEMPLATE.format(
        grade=grade,
        subject=subject,
        topic=topic,
        difficulty=difficulty,
        rag_context=rag_context if rag_context else "General English curriculum",
        weak_areas=", ".join(weak_areas) if weak_areas else "None identified yet",
        num_questions=num_questions,
    )


def build_correction_prompt(
    subject: str,
    grade: int,
    student_answer: str,
    correct_answer: str,
    error_type: str,
    rule: str,
    rag_context: str,
) -> str:
    """
    Build a prompt for mistake correction explanation.

    Returns:
        Formatted correction prompt.
    """
    return MISTAKE_CORRECTION_TEMPLATE.format(
        grade=grade,
        subject=subject,
        student_answer=student_answer,
        correct_answer=correct_answer,
        error_type=error_type,
        rule=rule,
        rag_context=rag_context if rag_context else "General English curriculum",
    )


def build_review_prompt(
    subject: str,
    grade: int,
    topics: list,
    accuracy: float,
    recent_mistakes: list,
    rag_context: str,
) -> str:
    """
    Build a prompt for a review session.

    Returns:
        Formatted review prompt.
    """
    mistakes_summary = "; ".join(
        f"{m.get('word', '?')} → {m.get('correct_answer', '?')}"
        for m in recent_mistakes[-5:]
    ) or "None"

    return REVIEW_SESSION_TEMPLATE.format(
        grade=grade,
        subject=subject,
        topics=", ".join(topics) if topics else "General review",
        accuracy=accuracy,
        recent_mistakes=mistakes_summary,
        rag_context=rag_context if rag_context else "General English curriculum",
    )


def get_encouragement(accuracy: float, streak: int) -> str:
    """
    Return a contextual encouragement message based on performance.

    Args:
        accuracy: Current accuracy percentage.
        streak: Login streak in days.

    Returns:
        Encouraging message string with emoji.
    """
    if accuracy >= 90:
        return f"🌟 Outstanding! You're a superstar with {accuracy:.0f}% accuracy!"
    elif accuracy >= 75:
        return f"🎉 Great job! {accuracy:.0f}% accuracy — you're really getting it!"
    elif accuracy >= 50:
        return f"💪 Good effort! {accuracy:.0f}% — keep practising and you'll improve!"
    else:
        if streak > 3:
            return f"🔥 You've been coming back every day for {streak} days — that dedication pays off!"
        return "😊 Don't worry — every expert was once a beginner. Let's keep going!"
