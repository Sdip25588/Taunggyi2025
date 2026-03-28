"""
ai_teacher.py — Core LLM and RAG pipeline.

Handles:
  - Loading curriculum PDFs and building a FAISS vector index
  - Retrieving relevant chunks for a student query
  - Calling the active LLM provider (Gemini → Groq → OpenRouter) with
    automatic fallback and per-provider retry logic
  - Structured model routing dict for easy future expansion

Provider fallback order (configured in config.py):
  1. Gemini (primary)   — Google Generative AI
  2. Groq   (backup)    — fast open-source models (LLaMA 3 etc.)
  3. OpenRouter (second backup) — wide model selection via unified API
  4. Friendly error message — if all providers fail
"""

import logging
import time
from pathlib import Path
from typing import Optional

import requests
from google import genai
from google.genai import types as genai_types

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    MODELS,
    ACTIVE_MODEL,
    PDF_PATHS,
    RAG_CONFIG,
    GEMINI_SAFETY_SETTINGS,
    validate_gemini_key,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Lazy imports for heavy ML libraries
# (avoids slow startup when RAG not needed)
# ─────────────────────────────────────────────
_faiss = None
_embeddings_model = None
_vectorstore = None

RAG_AVAILABLE = False  # Set True after successful index load/build

# ─────────────────────────────────────────────
# Retry / fallback constants
# ─────────────────────────────────────────────
_MAX_RETRIES: int = 3          # attempts per provider before giving up on that provider
_RETRY_DELAY: float = 1.0      # seconds to wait between retries (doubles each attempt)

# Error substrings that signal a hard quota/rate-limit — no point retrying
_RATE_LIMIT_SIGNALS: tuple = (
    "429",
    "RESOURCE_EXHAUSTED",
    "rate_limit",
    "rate limit",
    "quota",
    "Too Many Requests",
)

# Error substrings that signal the model/key is simply not available
_NOT_FOUND_SIGNALS: tuple = (
    "NOT_FOUND",
    "not found",
    "API_KEY_INVALID",
    "API key not valid",
    "model not found",
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if *exc* indicates a quota / rate-limit problem."""
    msg = str(exc).lower()
    return any(s.lower() in msg for s in _RATE_LIMIT_SIGNALS)


def _is_not_found_error(exc: Exception) -> bool:
    """Return True if *exc* indicates a model or key not-found problem."""
    msg = str(exc)
    return any(s.lower() in msg.lower() for s in _NOT_FOUND_SIGNALS)


def _import_rag_dependencies():
    """Import LangChain/FAISS dependencies lazily."""
    global _faiss
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return True
    except ImportError as exc:
        logger.warning("RAG dependencies not available: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Provider-specific call functions
# Each function accepts (prompt, temperature, max_tokens) and returns the
# model's text response, or raises an exception on failure.
# Retry logic is contained here so call_llm() stays clean.
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """
    Call the Google Gemini API and return the response text.

    Retries up to ``_MAX_RETRIES`` times on transient errors.
    Raises immediately on rate-limit or quota errors (429 / RESOURCE_EXHAUSTED)
    so the caller can fall back to the next provider without wasting time.

    Args:
        prompt:      Full prompt string to send to the model.
        temperature: Sampling temperature (0.0–1.0).
        max_tokens:  Maximum output tokens.

    Returns:
        The model's text response.

    Raises:
        Exception: On unrecoverable error after all retries are exhausted.
    """
    last_exc: Exception = RuntimeError("Gemini: no attempts made")
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    safety_settings=[
                        genai_types.SafetySetting(
                            category=s["category"],
                            threshold=s["threshold"],
                        )
                        for s in GEMINI_SAFETY_SETTINGS
                    ],
                ),
            )
            logger.info("Gemini responded successfully (attempt %d).", attempt)
            return response.text

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Gemini attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc
            )
            # Hard stop: rate-limit or not-found errors won't recover with retries
            if _is_rate_limit_error(exc) or _is_not_found_error(exc):
                logger.info(
                    "Gemini: non-retryable error (%s). Skipping remaining retries.", exc
                )
                break
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * attempt)  # simple linear back-off

    raise last_exc


def call_groq(prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """
    Call the Groq API and return the response text.

    Uses the ``groq`` Python SDK with the model configured in ``GROQ_MODEL``
    (default: ``llama3-8b-8192``).  Retries on transient errors; raises
    immediately on rate-limit errors.

    Args:
        prompt:      Full prompt string.
        temperature: Sampling temperature (0.0–1.0).
        max_tokens:  Maximum output tokens.

    Returns:
        The model's text response.

    Raises:
        Exception: On unrecoverable error after all retries are exhausted.
        ImportError: If the ``groq`` package is not installed.
    """
    import groq as groq_sdk  # lazy import — only needed when falling back

    last_exc: Exception = RuntimeError("Groq: no attempts made")
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            client = groq_sdk.Groq(api_key=GROQ_API_KEY)
            chat_completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.info("Groq responded successfully (attempt %d).", attempt)
            return chat_completion.choices[0].message.content

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Groq attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc
            )
            if _is_rate_limit_error(exc) or _is_not_found_error(exc):
                logger.info(
                    "Groq: non-retryable error (%s). Skipping remaining retries.", exc
                )
                break
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * attempt)

    raise last_exc


def call_openrouter(
    prompt: str, temperature: float = 0.7, max_tokens: int = 2048
) -> str:
    """
    Call the OpenRouter API and return the response text.

    OpenRouter exposes an OpenAI-compatible REST endpoint, so this function
    calls it directly with the ``requests`` library — no extra SDK needed.
    Uses the model configured in ``OPENROUTER_MODEL``
    (default: ``mistralai/mistral-7b-instruct``).

    Args:
        prompt:      Full prompt string.
        temperature: Sampling temperature (0.0–1.0).
        max_tokens:  Maximum output tokens.

    Returns:
        The model's text response.

    Raises:
        Exception: On unrecoverable error after all retries are exhausted.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Recommended by OpenRouter for rate-limit transparency
        "HTTP-Referer": "https://github.com/Sdip25588/Taunggyi2025",
        "X-Title": "Taunggyi English Tutor",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_exc: Exception = RuntimeError("OpenRouter: no attempts made")
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            # Raise for 4xx/5xx HTTP errors so the except block can handle them
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            logger.info("OpenRouter responded successfully (attempt %d).", attempt)
            return text

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "OpenRouter attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc
            )
            if _is_rate_limit_error(exc) or _is_not_found_error(exc):
                logger.info(
                    "OpenRouter: non-retryable error (%s). Skipping remaining retries.",
                    exc,
                )
                break
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * attempt)

    raise last_exc


# ─────────────────────────────────────────────
# Gemini key validation helper
# ─────────────────────────────────────────────

def _configure_gemini() -> tuple[bool, str]:
    """
    Validate the Gemini API key configuration.

    Returns:
        (ok, message) — ok is False when a problem is detected;
        message contains a human-readable description with remediation steps.
    """
    return validate_gemini_key(GEMINI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Main LLM entry point — tries providers in order with automatic fallback
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    mode: str = "explain",
    subject: str = "Phonics",
) -> str:
    """
    Call an LLM with automatic provider fallback.

    Tries providers in this order:
      1. Gemini   (primary)       — requires ``GEMINI_API_KEY``
      2. Groq     (first backup)  — requires ``GROQ_API_KEY``
      3. OpenRouter (second backup) — requires ``OPENROUTER_API_KEY``
      4. Friendly error message   — if *all* providers fail

    Each provider is given up to ``_MAX_RETRIES`` attempts on transient
    errors.  Rate-limit / quota errors skip retries immediately and
    move on to the next provider.

    To add a new provider in the future:
      1. Write a ``call_<provider>()`` function following the pattern above.
      2. Add its key/model to ``config.py``.
      3. Append it to the ``_providers`` list below.

    Args:
        prompt:  The full prompt (including system instructions and context).
        mode:    ``"explain"`` (temperature 0.7) or ``"quiz"`` (temperature 0.3).
        subject: Current subject — used only for logging / error messages.

    Returns:
        The model's text response, or a friendly fallback message string.
    """
    model_config = MODELS[ACTIVE_MODEL]
    temperature = (
        model_config["temperature_quiz"]
        if mode == "quiz"
        else model_config["temperature_explain"]
    )
    max_tokens = model_config["max_tokens"]

    # ── ordered list of (name, callable, api_key) ──────────────────────────
    _providers = [
        ("Gemini",      call_gemini,      GEMINI_API_KEY),
        ("Groq",        call_groq,        GROQ_API_KEY),
        ("OpenRouter",  call_openrouter,  OPENROUTER_API_KEY),
    ]

    errors: list[str] = []  # collect per-provider error summaries for logging

    for provider_name, provider_fn, api_key in _providers:
        # Skip providers whose API key is not configured
        if not api_key or api_key in ("your_gemini_api_key_here",
                                       "your_groq_api_key_here",
                                       "your_openrouter_api_key_here"):
            logger.info("Skipping %s: API key not configured.", provider_name)
            continue

        try:
            logger.info("Trying provider: %s (subject=%s, mode=%s).", provider_name, subject, mode)
            return provider_fn(prompt, temperature, max_tokens)

        except Exception as exc:
            error_summary = f"{provider_name}: {exc}"
            errors.append(error_summary)
            logger.warning("Provider %s failed — moving to next. Error: %s", provider_name, exc)

    # ── All providers exhausted ─────────────────────────────────────────────
    logger.error(
        "All LLM providers failed for subject=%s mode=%s. Errors: %s",
        subject, mode, " | ".join(errors),
    )
    return (
        "⚠️ **AI tutor is temporarily unavailable.**\n\n"
        "All AI providers are currently unreachable (rate limits or configuration issues). "
        "Please try again in a moment.\n\n"
        "**Quick fixes:**\n"
        "1. Check that at least one of `GEMINI_API_KEY`, `GROQ_API_KEY`, or "
        "`OPENROUTER_API_KEY` is set correctly in your `.env` or `config_secrets.json`.\n"
        "2. If you see a 429 error, you have hit a quota limit — wait a minute and retry, "
        "or add a backup API key.\n"
        "3. See README → Configure API Keys for step-by-step instructions."
    )


def generate_conversational_reply(
    student_input: str,
    username: str = "Student",
    grade: int = 1,
    subject: str = "English",
) -> str:
    """
    Generate a friendly, conversational reply to a non-lesson student message.

    Builds a tutor-style prompt that instructs the LLM to respond warmly and
    naturally—without pushing lesson content—then delegates to ``call_llm``
    so the same provider-fallback logic applies.

    Args:
        student_input: The student's casual or conversational message.
        username:      Student's name (for personalisation).
        grade:         Grade level (1–5) for age-appropriate tone.
        subject:       Current subject — included in the prompt so the LLM can
                       gently reference it if the student asks what they're studying.

    Returns:
        A short, friendly response string.
    """
    prompt = (
        f"You are a friendly and encouraging {subject} tutor helping a Grade {grade} student "
        f"named {username}. The student said: '{student_input}'. "
        f"Respond in a warm, conversational way — like a supportive teacher chatting with a "
        f"student. Do NOT continue a lesson or push new lesson content. "
        f"Keep your response short, friendly, and age-appropriate."
    )
    return call_llm(prompt, mode="explain", subject=subject)


# ─────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────

def build_faiss_index(force_rebuild: bool = False) -> bool:
    """
    Load curriculum PDFs, create embeddings, and save FAISS index.

    Skips rebuild if the index already exists on disk (unless force_rebuild=True).

    Args:
        force_rebuild: If True, always rebuild even if index exists.

    Returns:
        True if index is ready, False if no PDFs found or dependencies missing.
    """
    global _vectorstore, RAG_AVAILABLE

    index_path = RAG_CONFIG["faiss_index_path"]
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    # Try loading existing index
    if not force_rebuild and Path(index_path).exists():
        return _load_faiss_index()

    if not _import_rag_dependencies():
        return False

    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # Load all available PDFs
    all_docs = []
    for subject, pdf_path in PDF_PATHS.items():
        if pdf_path.exists():
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                # Tag each chunk with its source subject
                for doc in docs:
                    doc.metadata["subject"] = subject
                all_docs.extend(docs)
                logger.info("Loaded %d pages from %s", len(docs), pdf_path.name)
            except Exception as exc:
                logger.warning("Could not load %s: %s", pdf_path, exc)

    if not all_docs:
        logger.warning("No curriculum PDFs found. RAG disabled.")
        RAG_AVAILABLE = False
        return False

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CONFIG["chunk_size"],
        chunk_overlap=RAG_CONFIG["chunk_overlap"],
    )
    chunks = splitter.split_documents(all_docs)
    logger.info("Created %d text chunks from %d pages", len(chunks), len(all_docs))

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name=RAG_CONFIG["embedding_model"],
        model_kwargs={"device": "cpu"},
    )
    _vectorstore = FAISS.from_documents(chunks, embeddings)
    _vectorstore.save_local(index_path)
    logger.info("FAISS index saved to %s", index_path)

    RAG_AVAILABLE = True
    return True


def load_faiss_index() -> bool:
    """Load an existing FAISS index from disk."""
    global _vectorstore, RAG_AVAILABLE

    if not _import_rag_dependencies():
        return False

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    index_path = RAG_CONFIG["faiss_index_path"]
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=RAG_CONFIG["embedding_model"],
            model_kwargs={"device": "cpu"},
        )
        _vectorstore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded from %s", index_path)
        RAG_AVAILABLE = True
        return True
    except Exception as exc:
        logger.warning("Could not load FAISS index: %s", exc)
        RAG_AVAILABLE = False
        return False


# Keep the private alias for internal use
_load_faiss_index = load_faiss_index


def retrieve_context(
    query: str,
    subject: Optional[str] = None,
    top_k: Optional[int] = None,
) -> str:
    """
    Retrieve relevant curriculum chunks for a student query.

    Args:
        query: The student's question or current lesson topic.
        subject: If given, prefer chunks from this subject.
        top_k: Number of chunks to retrieve (defaults to config value).

    Returns:
        Concatenated context string, or empty string if RAG unavailable.
    """
    if not RAG_AVAILABLE or _vectorstore is None:
        return ""

    k = top_k or RAG_CONFIG["top_k"]

    try:
        # Build filter if subject is specified
        if subject:
            filter_dict = {"subject": subject.lower()}
            docs = _vectorstore.similarity_search(query, k=k, filter=filter_dict)
            # Fall back to unfiltered if no subject-specific results
            if not docs:
                docs = _vectorstore.similarity_search(query, k=k)
        else:
            docs = _vectorstore.similarity_search(query, k=k)

        if not docs:
            return ""

        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "curriculum")
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"[Source: {Path(source).name}, Page {page}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)
    except Exception as exc:
        logger.warning("RAG retrieval error: %s", exc)
        return ""


def check_pdfs_available() -> dict:
    """
    Check which curriculum PDFs exist on disk.

    Returns:
        Dict mapping subject name → True/False.
    """
    return {
        subject: path.exists()
        for subject, path in PDF_PATHS.items()
    }


def get_rag_status() -> dict:
    """
    Return a status summary for display in the UI.

    Returns:
        Dict with 'available' (bool), 'index_exists' (bool), 'pdfs' (dict).
    """
    index_path = RAG_CONFIG["faiss_index_path"]
    return {
        "available": RAG_AVAILABLE,
        "index_exists": Path(index_path).exists(),
        "pdfs": check_pdfs_available(),
    }
