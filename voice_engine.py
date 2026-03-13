"""
voice_engine.py — Voice conversation engine for Azure Speech-to-Text and TTS.

Handles:
  - Speech-to-Text (student speaks → text) via Azure STT or SpeechRecognition fallback
  - Text-to-Speech (AI text → audio) via Azure Neural voices with SSML
  - Pronunciation evaluation using difflib similarity
  - Graceful degradation when Azure is not configured
"""

import difflib
import logging
import os
import re
import tempfile
import unicodedata
from typing import Optional

from config import (
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    AZURE_STT_LANGUAGE,
    PRONUNCIATION_EXCELLENT_THRESHOLD,
    PRONUNCIATION_GOOD_THRESHOLD,
    TTS_CONFIG,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Emoji → tone mapping for SSML voice styles
# ─────────────────────────────────────────────
EMOJI_TONE_STYLES: dict = {
    # Celebratory / praise emojis → cheerful
    "🌟": "cheerful", "⭐": "cheerful", "🎉": "cheerful", "🏆": "cheerful",
    "✅": "cheerful", "🎊": "cheerful", "🥳": "cheerful", "👏": "cheerful",
    # Encouragement / warm emojis → friendly
    "😊": "friendly", "💪": "friendly", "🤗": "friendly", "😄": "friendly",
    "❤️": "friendly", "💖": "friendly", "🙌": "friendly",
    # Thinking / learning emojis → calm
    "🤔": "calm", "📚": "calm", "📖": "calm", "💡": "calm", "🧠": "calm",
    # Gentle correction → gentle
    "⚠️": "gentle", "💬": "gentle", "🔄": "gentle",
    # Fun / playful
    "🎮": "cheerful", "🎨": "cheerful", "🚀": "cheerful",
}

# Azure Neural voice SSML style map
_AZURE_VOICE_STYLES: dict = {
    "cheerful": "cheerful",
    "friendly": "friendly",
    "calm": "calm",
    "gentle": "gentle",
    "default": "friendly",
}


def _strip_emojis(text: str) -> str:
    """
    Remove emoji and pictographic characters from text using Unicode category detection.

    Uses unicodedata.category to identify symbol characters (category 'So' = Other Symbol)
    and other non-letter, non-digit pictographic characters, avoiding overly broad regex ranges.
    """
    result = []
    for char in text:
        cat = unicodedata.category(char)
        cp = ord(char)
        # Keep standard ASCII and letter/number/punctuation categories
        # Remove: So (Other Symbol), Mn if high codepoint (combining marks in emoji sequences)
        # and characters in the Supplementary Multilingual Plane (emoji range > U+1F000)
        if cp > 0x1F000:
            continue  # Skip emoji and pictographic characters in SMP
        if cat in ("So", "Sk") and cp > 0x2000:
            continue  # Skip other symbols that are emoji-like
        result.append(char)
    # Also remove variation selector characters (U+FE0F etc.) that follow emoji
    cleaned = "".join(result)
    # Remove variation selectors (U+FE00 to U+FE0F)
    cleaned = re.sub(r"[\uFE00-\uFE0F]", "", cleaned)
    return cleaned


# ─────────────────────────────────────────────
# Text preparation
# ─────────────────────────────────────────────

def prepare_text_for_tts(text: str) -> tuple[str, str]:
    """
    Strip emojis and markdown from text for TTS, and determine voice style.

    The style is inferred from the dominant emoji type in the original text.
    This matches the 'Synthesis-style' approach: voice tone conveys emotion,
    not literal emoji names.

    Args:
        text: Raw text that may contain emojis and markdown.

    Returns:
        Tuple of (clean_text: str, voice_style: str).
    """
    # Detect dominant emotion from emojis before stripping
    voice_style = _detect_voice_style(text)

    # Strip markdown formatting
    clean = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    clean = re.sub(r"\*(.+?)\*", r"\1", clean)
    clean = re.sub(r"#{1,6}\s+", "", clean)
    clean = re.sub(r"`(.+?)`", r"\1", clean)
    clean = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", clean)
    clean = re.sub(r"^\s*[-*+]\s+", "", clean, flags=re.MULTILINE)

    # Strip all emojis (voice tone handles the emotion)
    clean = _strip_emojis(clean)

    # Normalise whitespace
    clean = re.sub(r"\s+", " ", clean).strip()

    return clean, voice_style


def _detect_voice_style(text: str) -> str:
    """
    Determine the best SSML voice style from emojis in the text.

    Returns the most frequently occurring style, defaulting to 'friendly'.
    """
    style_counts: dict = {}
    for emoji_char, style in EMOJI_TONE_STYLES.items():
        count = text.count(emoji_char)
        if count > 0:
            style_counts[style] = style_counts.get(style, 0) + count

    if not style_counts:
        return "friendly"
    return max(style_counts, key=lambda x: style_counts[x])


# ─────────────────────────────────────────────
# Speech-to-Text
# ─────────────────────────────────────────────

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Convenience wrapper: convert raw audio bytes to a transcription string.

    Returns the transcribed text, or an empty string on failure.
    Suitable for use in the conversation loop without handling the full
    dict result from listen_to_student.

    Args:
        audio_bytes: Raw audio bytes (WAV or MP3).

    Returns:
        Transcribed text string, or "" if recognition failed.
    """
    result = listen_to_student(audio_bytes)
    return result.get("text", "") if result.get("success") else ""


def synthesize_speech(text: str, voice_style: str = "friendly") -> Optional[str]:
    """
    Convenience wrapper: convert text to speech and return the audio file path.

    Strips emojis and markdown before synthesis. Delegates to speak_response.

    Args:
        text: Text to speak (may contain emojis/markdown — they will be stripped).
        voice_style: SSML style hint ("cheerful", "friendly", "calm", "gentle").

    Returns:
        Path to the generated MP3 file, or None on failure.
    """
    return speak_response(text, voice_style=voice_style)


def listen_to_student(audio_bytes: Optional[bytes] = None) -> dict:
    """
    Convert student audio to text using Azure STT or SpeechRecognition fallback.

    For Streamlit, audio_bytes comes from audio_recorder_streamlit or file upload.
    Falls back gracefully to text input when Azure is not configured.

    Args:
        audio_bytes: Raw audio bytes (WAV or MP3) from the browser recorder.

    Returns:
        Dict: {success: bool, text: str, confidence: float, error: str}
    """
    if not audio_bytes:
        return {"success": False, "text": "", "confidence": 0.0,
                "error": "No audio data provided."}

    # Try Azure STT first
    if AZURE_SPEECH_KEY and AZURE_SPEECH_REGION:
        result = _azure_stt(audio_bytes)
        if result["success"]:
            return result

    # Fallback: SpeechRecognition library
    return _sr_fallback_stt(audio_bytes)


def _azure_stt(audio_bytes: bytes) -> dict:
    """Speech-to-text via Azure Cognitive Services Speech SDK."""
    try:
        import azure.cognitiveservices.speech as speechsdk

        # Write audio to a temp file for Azure SDK
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            speech_config = speechsdk.SpeechConfig(
                subscription=AZURE_SPEECH_KEY,
                region=AZURE_SPEECH_REGION,
            )
            speech_config.speech_recognition_language = AZURE_STT_LANGUAGE

            audio_input = speechsdk.AudioConfig(filename=tmp_path)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_input,
            )

            result = recognizer.recognize_once()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return {
                    "success": True,
                    "text": result.text,
                    "confidence": 0.95,
                    "error": "",
                }
            else:
                return {
                    "success": False,
                    "text": "",
                    "confidence": 0.0,
                    "error": f"Azure STT: {result.reason.name}",
                }
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    except ImportError:
        logger.warning("azure-cognitiveservices-speech not installed.")
        return {"success": False, "text": "", "confidence": 0.0,
                "error": "Azure Speech SDK not installed."}
    except Exception as exc:
        logger.warning("Azure STT error: %s", exc)
        return {"success": False, "text": "", "confidence": 0.0, "error": str(exc)}


def _sr_fallback_stt(audio_bytes: bytes) -> dict:
    """Fallback STT using the SpeechRecognition library with Google free API."""
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            with sr.AudioFile(tmp_path) as source:
                audio = recognizer.record(source)

            text = recognizer.recognize_google(audio, language=AZURE_STT_LANGUAGE)
            return {"success": True, "text": text, "confidence": 0.80, "error": ""}

        except sr.UnknownValueError:
            return {"success": False, "text": "", "confidence": 0.0,
                    "error": "Could not understand the audio. Please try again."}
        except sr.RequestError as exc:
            return {"success": False, "text": "", "confidence": 0.0, "error": str(exc)}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    except ImportError:
        logger.warning("SpeechRecognition not installed.")
        return {"success": False, "text": "", "confidence": 0.0,
                "error": "SpeechRecognition not installed."}
    except Exception as exc:
        logger.warning("SpeechRecognition fallback error: %s", exc)
        return {"success": False, "text": "", "confidence": 0.0, "error": str(exc)}


# ─────────────────────────────────────────────
# Text-to-Speech
# ─────────────────────────────────────────────

def speak_response(text: str, voice_style: str = "friendly") -> Optional[str]:
    """
    Convert AI response text to speech audio, returning path to MP3 file.

    Strips emojis before TTS — voice tone conveys the emotion via SSML.
    Uses Azure Neural voices with expressive SSML styles when available.
    Falls back to edge-tts (free) if Azure is not configured.

    Args:
        text: AI response text (may contain emojis and markdown).
        voice_style: SSML style ("cheerful", "friendly", "calm", "gentle").

    Returns:
        Path to the generated MP3 file, or None on failure.
    """
    clean_text, detected_style = prepare_text_for_tts(text)
    # Use detected style unless an explicit style was passed (not default)
    effective_style = detected_style if voice_style == "friendly" else voice_style
    clean_text = clean_text[:800]  # Limit length for TTS

    if not clean_text:
        return None

    # Try Azure TTS with SSML
    if TTS_CONFIG.get("provider") == "azure" and AZURE_SPEECH_KEY and AZURE_SPEECH_REGION:
        result = _azure_tts_ssml(clean_text, effective_style)
        if result:
            return result

    # Fallback to edge-tts
    return _edge_tts(clean_text)


def _azure_tts_ssml(text: str, voice_style: str = "friendly") -> Optional[str]:
    """
    Generate speech via Azure Neural voice with SSML expressive styles.

    Uses SSML <mstts:express-as> for natural emotion without reading emoji names.
    """
    try:
        import azure.cognitiveservices.speech as speechsdk

        ssml_style = _AZURE_VOICE_STYLES.get(voice_style, "friendly")
        voice = TTS_CONFIG.get("azure_voice", "en-US-JennyNeural")

        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">'
            f'<voice name="{voice}">'
            f'<mstts:express-as style="{ssml_style}">'
            f'{_escape_xml(text)}'
            f'</mstts:express-as>'
            f'</voice>'
            f'</speak>'
        )

        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_SPEECH_REGION,
        )

        tmp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        audio_config = speechsdk.audio.AudioOutputConfig(filename=tmp_path)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config
        )
        result = synthesizer.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return tmp_path
        else:
            logger.warning("Azure TTS SSML failed: %s", result.reason)
            return None

    except ImportError:
        logger.warning("azure-cognitiveservices-speech not installed.")
        return None
    except Exception as exc:
        logger.warning("Azure TTS SSML error: %s", exc)
        return None


def _edge_tts(text: str) -> Optional[str]:
    """Generate speech using edge-tts (free, no API key required)."""
    try:
        import asyncio
        import edge_tts

        voice = TTS_CONFIG.get("edge_voice", "en-US-AriaNeural")
        tmp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        async def _run():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(tmp_path)

        asyncio.run(_run())
        return tmp_path

    except ImportError:
        logger.warning("edge-tts not installed.")
        return None
    except Exception as exc:
        logger.warning("edge-tts error: %s", exc)
        return None


def _escape_xml(text: str) -> str:
    """Escape special XML characters for safe SSML embedding."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
    )


# ─────────────────────────────────────────────
# Pronunciation Evaluation
# ─────────────────────────────────────────────

def evaluate_pronunciation(
    student_audio_text: str, expected_text: str, grade_level: int = 1
) -> dict:
    """
    Compare what student said vs what they should have said.

    Uses difflib.SequenceMatcher to calculate similarity.
    Returns structured feedback with warm encouragement.

    Args:
        student_audio_text: Text from STT (what student actually said).
        expected_text: The target text the student should have read.
        grade_level: For appropriate feedback complexity.

    Returns:
        Dict: {similarity: float, feedback_text: str, level: str,
               tts_word: str, should_retry: bool}
    """
    if not student_audio_text:
        return {
            "similarity": 0.0,
            "feedback_text": "I didn't catch that! Could you try speaking again? 🎤",
            "level": "no_audio",
            "tts_word": expected_text,
            "should_retry": True,
        }

    student_clean = student_audio_text.strip().lower()
    expected_clean = expected_text.strip().lower()

    similarity = difflib.SequenceMatcher(
        None, student_clean, expected_clean
    ).ratio()

    if similarity >= PRONUNCIATION_EXCELLENT_THRESHOLD:
        feedback = (
            f"Perfect pronunciation! You said it exactly right! "
            f"Keep up the amazing work!"
        )
        level = "excellent"
        should_retry = False

    elif similarity >= PRONUNCIATION_GOOD_THRESHOLD:
        # Find the differing word
        difficult_word = _find_different_word(student_clean, expected_clean)
        if difficult_word:
            feedback = (
                f"Almost there! You're so close! "
                f"Let's try the word '{difficult_word}' together. "
                f"Listen carefully, then say it with me: {difficult_word}!"
            )
        else:
            feedback = (
                f"Almost there! You've got most of it right! "
                f"Let's say the whole thing together one more time!"
            )
        level = "good"
        should_retry = True

    else:
        # Find the first different part
        first_words = expected_text.split()
        syllables = _break_into_syllables(first_words[0] if first_words else expected_text)
        feedback = (
            f"Let's practice this together step by step. "
            f"I'll say it slowly: {syllables}. "
            f"Now say each part with me! You can do it!"
        )
        level = "needs_practice"
        should_retry = True

    return {
        "similarity": round(similarity, 4),
        "feedback_text": feedback,
        "level": level,
        "tts_word": expected_text,
        "should_retry": should_retry,
    }


def _find_different_word(student: str, expected: str) -> str:
    """Find the first word that differs between student and expected text."""
    student_words = student.split()
    expected_words = expected.split()

    for s_word, e_word in zip(student_words, expected_words):
        if s_word != e_word:
            return e_word

    # If different lengths, return the extra/missing word
    if len(expected_words) > len(student_words):
        return expected_words[len(student_words)]

    return expected_words[0] if expected_words else ""


def _break_into_syllables(word: str) -> str:
    """
    Simple syllable breakdown for pronunciation guidance.
    Uses vowel clusters as syllable boundaries.
    """
    vowels = "aeiouAEIOU"
    if len(word) <= 3:
        return " - ".join(word.upper())

    syllables = []
    current = ""
    for i, char in enumerate(word):
        current += char
        # Break after vowel followed by consonant followed by vowel
        if (i < len(word) - 2 and char.lower() in vowels
                and word[i + 1].lower() not in vowels
                and i + 2 < len(word) and word[i + 2].lower() in vowels):
            syllables.append(current)
            current = ""

    if current:
        syllables.append(current)

    return " - ".join(syllables) if len(syllables) > 1 else word
