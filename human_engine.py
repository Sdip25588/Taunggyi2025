"""
human_engine.py — Build human-like, preface-driven teaching prompts.

Extracts teaching methodology from each PDF's preface and generates
system prompts that make the LLM teach like the textbook authors intended.
"""

import datetime
import random
from typing import Optional

# ─────────────────────────────────────────────
# Professor mode: varied greeting bank
# ─────────────────────────────────────────────

GREETING_BANK: list[str] = [
    "Hello {name}! I'm happy to see you today. How are you doing?",
    "Hi {name}! Great to have you back. How are you feeling today?",
    "Good to see you, {name}! Before we start, how's your day going?",
    "Hey {name}! Welcome back to class. How are you today?",
    "Hello {name}! I've been looking forward to our session. How are you?",
    "Hi {name}! Ready for some English practice? How are you feeling?",
    "Welcome back, {name}! It's always wonderful to see you. How are you today?",
    "Good day, {name}! I'm glad you're here. How has your day been so far?",
    "Hello {name}! Wonderful to see you again. How are you doing today?",
    "Hi there, {name}! So happy you're here. How are you feeling right now?",
]

MORNING_GREETINGS: list[str] = [
    "Good morning, {name}! I hope you had a great start to your day. How are you?",
    "Good morning, {name}! The best time to learn is in the morning. How are you feeling?",
]

AFTERNOON_GREETINGS: list[str] = [
    "Good afternoon, {name}! Hope your day is going well. How are you?",
    "Good afternoon, {name}! Let's make this afternoon a great learning session. How are you doing?",
]

EVENING_GREETINGS: list[str] = [
    "Good evening, {name}! Thanks for taking time to learn today. How are you?",
    "Good evening, {name}! Let's end the day on a great note. How are you feeling?",
]


def get_varied_greeting(name: str) -> str:
    """
    Return a varied, friendly greeting for the student.

    Chooses from a bank of greetings, optionally influenced by time of day.
    Guaranteed to feel different across sessions.

    Args:
        name: The student's first name.

    Returns:
        Greeting string addressed to the student.
    """
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        time_pool = MORNING_GREETINGS
    elif 12 <= hour < 17:
        time_pool = AFTERNOON_GREETINGS
    elif 17 <= hour < 21:
        time_pool = EVENING_GREETINGS
    else:
        time_pool = []

    # Combine time-of-day greetings with general bank; weight general bank higher
    pool = GREETING_BANK + time_pool
    greeting_template = random.choice(pool)
    return greeting_template.format(name=name)


# ─────────────────────────────────────────────
# Professor mode prompt templates
# ─────────────────────────────────────────────

CHECKIN_FOLLOWUP_TEMPLATE = """You are a warm, friendly English teacher talking with {name} (Grade {grade} student).

The student just said: "{student_response}"

This seems like a short or unclear reply. Your job is to gently follow up to:
1. Show you heard them and care about how they feel.
2. Ask one simple follow-up question to help them open up (e.g. about their day, what they did, how they're feeling).
3. Keep it very brief, natural, and encouraging — like a caring teacher.
4. End by transitioning smoothly toward starting the lesson (e.g. "When you're ready, we can begin!").

Respond as the teacher (2-3 sentences maximum):"""


DOUBT_HANDLER_TEMPLATE = """You are a warm, patient English teacher for Grade {grade} students.

Student name: {name}
The student asked: "{question}"

Your job:
1. Answer their question clearly and kindly — this is their doubt/question and it must be answered FIRST.
2. Use simple language appropriate for Grade {grade}.
3. Give a short example if helpful.
4. After answering, smoothly transition back: "Does that make sense? Great! Now, let's continue with today's lesson."

Keep the answer focused (3-5 sentences). Use 1 emoji. Respond as the teacher:"""


PROFESSOR_LESSON_INTRO_TEMPLATE = """You are an enthusiastic English professor introducing today's lesson to {name} (Grade {grade}).

Today's chosen lesson:
- Subject: {subject}
- Topic: {topic}
- Reason the professor chose this: {reason}

Your job:
1. Briefly announce today's topic in an exciting, encouraging way.
2. Explain WHY you chose it in student-friendly language (use the reason above as a guide).
3. Tell the student what they'll be doing: a quick warm-up, some teaching, and a little practice.
4. Invite them to begin with an encouraging phrase.

Keep it short (3-4 sentences). Use 1-2 emojis. Speak directly to the student. Respond as the professor:"""


WARMUP_TEMPLATE = """You are an encouraging English teacher giving {name} (Grade {grade}) a quick confidence-building warm-up.

Subject: {subject}
Topic being introduced today: {topic}
Student's current difficulty level: {difficulty}

Your job:
1. Start with ONE simple question or activity that the student almost certainly knows the answer to.
2. Frame it positively: "Let's start with something you already know well!"
3. Wait for their answer (end with the question for them to respond to).

Keep it to 2-3 sentences. End with the warm-up question. Use 1 emoji:"""


LESSON_WRAPUP_TEMPLATE = """You are a warm English teacher wrapping up today's lesson with {name} (Grade {grade}).

What was covered today:
- Subject: {subject}
- Topic: {topic}
- Student performance: {performance_summary}

Your job:
1. Praise the student genuinely and specifically for something they did well today.
2. Give a short summary of what was learned (1 sentence).
3. Tell them what exciting thing is coming in the next session.
4. End with a warm, motivating goodbye.

Keep it to 3-4 sentences. Use 1-2 emojis. Respond as the teacher:"""

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


# ─────────────────────────────────────────────
# Academic Mode Prompt Templates
# ─────────────────────────────────────────────

READING_PRACTICE_TEMPLATE = """You are a warm, encouraging English teacher helping a Grade {grade} student practice reading aloud.

PASSAGE FROM CURRICULUM:
{passage}

STUDENT'S READING (what they said):
{student_reading}

YOUR JOB:
1. Compare what the student read with the passage.
2. Celebrate what they got right first — always find something to praise.
3. For any words they struggled with, say "Let's try that word together!" and model it.
4. Never say "wrong" or "incorrect" — use "almost there!", "let's try together", "you're so close!".
5. Give 1-2 specific tips (e.g., "The word 'through' sounds like 'throo' — the 'ough' is silent here!").
6. End with a specific encouraging message and invite them to try again.

RULES:
- Grade {grade} appropriate language only.
- Warm, patient, enthusiastic teacher voice.
- If student_reading is empty or unclear, gently ask them to try reading aloud.

Respond as the teacher:"""


VOCABULARY_PRACTICE_TEMPLATE = """You are an enthusiastic English teacher introducing vocabulary to a Grade {grade} student.

VOCABULARY WORDS FOR THIS LESSON:
{words_with_definitions}

WORDS THE STUDENT ALREADY KNOWS:
{known_words}

YOUR JOB:
1. Introduce 2-3 NEW words the student doesn't know yet (not in known_words).
2. For each word: say the word, give a simple definition, use it in a fun example sentence.
3. Ask the student to use the word in their own sentence.
4. If they already know most words, celebrate and introduce a bonus challenge word.
5. Use emojis and enthusiasm to make vocabulary fun!

RULES:
- Grade {grade} vocabulary and sentence complexity.
- One word at a time if the student seems overwhelmed.
- Always celebrate when they use a word correctly.

Respond as the teacher:"""


WRITING_EVALUATION_TEMPLATE = """You are a kind, encouraging writing teacher for a Grade {grade} student.

WRITING PROMPT GIVEN TO STUDENT:
{prompt_given}

STUDENT'S RESPONSE:
{student_sentence}

EVALUATE and RESPOND:
1. Find something genuinely great about their writing — always start positive.
2. Check: grammar, spelling, vocabulary usage, sentence structure.
3. Give ONE gentle suggestion to improve (not multiple — keep it manageable).
4. Model the improvement: "What if we tried: [improved version]? That sounds great, right?"
5. Ask them to try the improved version or write another sentence.

IMPORTANT RULES:
- NEVER say "wrong", "incorrect", "mistake", or "error".
- Always use: "almost there!", "what if we tried...", "I love how you...", "let's make it even better".
- Grade {grade} expectations — celebrate every effort.
- Keep feedback to 3-4 sentences maximum.

Respond as the teacher:"""


PRONUNCIATION_FEEDBACK_TEMPLATE = """You are a patient, encouraging pronunciation coach for a Grade {grade} student.

EXPECTED TEXT (what they should have said):
{expected}

WHAT STUDENT ACTUALLY SAID:
{actual}

SIMILARITY SCORE: {similarity_score:.0%}

RESPOND based on the similarity score:
- If score >= 90%: "Perfect pronunciation! 🌟 You said it beautifully!"
- If score 70-89%: "Almost there! Let's try the tricky word together. [identify the word that was different]. 
  Say it with me: [word]. You can do it!"  
- If score < 70%: "Let's practice this together step by step. I'll break it down slowly: 
  [break the difficult words into syllables]. Say each part with me!"

RULES:
- Be warm and specific — identify which word/sound was challenging.
- Model the correct pronunciation by breaking words into syllables.
- Always invite them to try again.
- Grade {grade} appropriate encouragement.

Respond as the teacher:"""


def build_reading_practice_prompt(
    passage: str,
    student_reading: str,
    grade_level: int,
) -> str:
    """
    Build a prompt for reading aloud practice and pronunciation feedback.

    Args:
        passage: The passage from curriculum the student should read.
        student_reading: What the student actually said/typed.
        grade_level: Student's grade level.

    Returns:
        Formatted reading practice prompt.
    """
    return READING_PRACTICE_TEMPLATE.format(
        grade=grade_level,
        passage=passage or "No passage provided — teach a short appropriate reading passage.",
        student_reading=student_reading or "(Student has not yet read aloud)",
    )


def build_vocabulary_prompt(
    words: list,
    grade_level: int,
    known_words: list,
) -> str:
    """
    Build a prompt for vocabulary building practice.

    Args:
        words: List of dicts {word, definition} for the current lesson.
        grade_level: Student's grade level.
        known_words: List of words the student already knows.

    Returns:
        Formatted vocabulary prompt.
    """
    if words:
        words_text = "\n".join(
            f"- {w.get('word', w) if isinstance(w, dict) else w}"
            + (f": {w.get('definition', '')}" if isinstance(w, dict) and w.get("definition") else "")
            for w in words
        )
    else:
        words_text = "No specific word list — introduce grade-appropriate vocabulary."

    return VOCABULARY_PRACTICE_TEMPLATE.format(
        grade=grade_level,
        words_with_definitions=words_text,
        known_words=", ".join(known_words) if known_words else "None yet",
    )


def build_writing_evaluation_prompt(
    student_sentence: str,
    prompt_given: str,
    grade_level: int,
) -> str:
    """
    Build a prompt for evaluating a student's writing.

    Args:
        student_sentence: What the student wrote.
        prompt_given: The writing prompt they were given.
        grade_level: Student's grade level.

    Returns:
        Formatted writing evaluation prompt.
    """
    return WRITING_EVALUATION_TEMPLATE.format(
        grade=grade_level,
        prompt_given=prompt_given or "Write a sentence using a word you learned today.",
        student_sentence=student_sentence or "(Student has not yet written anything)",
    )


def build_pronunciation_feedback_prompt(
    expected: str,
    actual: str,
    similarity_score: float,
    grade_level: int = 1,
) -> str:
    """
    Build a prompt for pronunciation coaching feedback.

    Args:
        expected: The text the student should have said.
        actual: What the student actually said (from STT).
        similarity_score: Float 0-1 from difflib.SequenceMatcher.
        grade_level: Student's grade level.

    Returns:
        Formatted pronunciation feedback prompt.
    """
    return PRONUNCIATION_FEEDBACK_TEMPLATE.format(
        grade=grade_level,
        expected=expected,
        actual=actual or "(No speech detected)",
        similarity_score=similarity_score,
    )


# ─────────────────────────────────────────────
# Professor mode builder functions
# ─────────────────────────────────────────────

def build_checkin_followup_prompt(
    name: str,
    grade: int,
    student_response: str,
) -> str:
    """
    Build a prompt to follow up on a short/unclear student check-in response.

    Args:
        name: Student's name.
        grade: Grade level.
        student_response: What the student said in response to the greeting.

    Returns:
        Formatted check-in follow-up prompt.
    """
    return CHECKIN_FOLLOWUP_TEMPLATE.format(
        name=name,
        grade=grade,
        student_response=student_response or "(no response)",
    )


def build_doubt_handler_prompt(
    name: str,
    grade: int,
    question: str,
) -> str:
    """
    Build a prompt that answers a student's doubt, then transitions back to the lesson.

    Args:
        name: Student's name.
        grade: Grade level.
        question: The student's question/doubt.

    Returns:
        Formatted doubt-handling prompt.
    """
    return DOUBT_HANDLER_TEMPLATE.format(
        name=name,
        grade=grade,
        question=question,
    )


def build_professor_lesson_intro_prompt(
    name: str,
    grade: int,
    subject: str,
    topic: str,
    reason: str,
) -> str:
    """
    Build a prompt for the professor to introduce today's chosen lesson.

    Args:
        name: Student's name.
        grade: Grade level.
        subject: Today's subject (Phonics/Reading/Spelling).
        topic: Today's specific topic.
        reason: Why the professor chose this topic.

    Returns:
        Formatted professor lesson intro prompt.
    """
    return PROFESSOR_LESSON_INTRO_TEMPLATE.format(
        name=name,
        grade=grade,
        subject=subject,
        topic=topic,
        reason=reason,
    )


def build_warmup_prompt(
    name: str,
    grade: int,
    subject: str,
    topic: str,
    difficulty: str,
) -> str:
    """
    Build a prompt for a quick warm-up activity at the start of a lesson.

    Args:
        name: Student's name.
        grade: Grade level.
        subject: Today's subject.
        topic: Today's topic.
        difficulty: Student's current difficulty level.

    Returns:
        Formatted warm-up prompt.
    """
    return WARMUP_TEMPLATE.format(
        name=name,
        grade=grade,
        subject=subject,
        topic=topic,
        difficulty=difficulty,
    )


def build_wrapup_prompt(
    name: str,
    grade: int,
    subject: str,
    topic: str,
    performance_summary: str,
) -> str:
    """
    Build a prompt for the lesson wrap-up with praise and a preview of what's next.

    Args:
        name: Student's name.
        grade: Grade level.
        subject: Today's subject.
        topic: Today's topic.
        performance_summary: Brief description of how the student did.

    Returns:
        Formatted wrap-up prompt.
    """
    return LESSON_WRAPUP_TEMPLATE.format(
        name=name,
        grade=grade,
        subject=subject,
        topic=topic,
        performance_summary=performance_summary,
    )
