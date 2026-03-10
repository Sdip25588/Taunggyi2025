"""
ai_teacher.py — Core LLM and RAG pipeline.

Handles:
  - Loading curriculum PDFs and building a FAISS vector index
  - Retrieving relevant chunks for a student query
  - Calling Google Gemini 1.5 Flash with a grounded prompt
  - Structured model routing dict for future expansion
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types as genai_types

from config import (
    GEMINI_API_KEY,
    MODELS,
    ACTIVE_MODEL,
    PDF_PATHS,
    RAG_CONFIG,
    GEMINI_SAFETY_SETTINGS,
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


# ─────────────────────────────────────────────
# Gemini LLM setup
# ─────────────────────────────────────────────

def _configure_gemini() -> bool:
    """Configure the Gemini SDK. Returns True if successful."""
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set — LLM calls will fail.")
        return False
    return True


def call_llm(
    prompt: str,
    mode: str = "explain",
    subject: str = "Phonics",
) -> str:
    """
    Call the active LLM (Gemini 1.5 Flash) with the given prompt.

    Args:
        prompt: The full prompt including system instructions and context.
        mode: "explain" (temperature 0.7) or "quiz" (temperature 0.3).
        subject: Current subject (for safety filtering context).

    Returns:
        The model's text response, or an error message string.
    """
    if not _configure_gemini():
        return (
            "⚠️ **Gemini API key not configured.**\n\n"
            "Please add your `GEMINI_API_KEY` to the `.env` file.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    model_config = MODELS[ACTIVE_MODEL]
    temperature = (
        model_config["temperature_quiz"]
        if mode == "quiz"
        else model_config["temperature_explain"]
    )

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=model_config["model_id"],
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=model_config["max_tokens"],
                safety_settings=[
                    genai_types.SafetySetting(
                        category=s["category"],
                        threshold=s["threshold"],
                    )
                    for s in GEMINI_SAFETY_SETTINGS
                ],
            ),
        )
        return response.text
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        return f"⚠️ AI response error: {exc}\n\nPlease check your API key and try again."


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
