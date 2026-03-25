#!/usr/bin/env python3
"""
check_api_key.py — Quick CLI diagnostic for Gemini API key setup.

Run this script from the project root to verify that the app will be
able to find and use your Gemini API key before you start the app:

    python check_api_key.py

Exit codes:
    0 — Key looks valid (format check only; does not make a live API call).
    1 — Key is missing, a placeholder, or has a formatting problem.

This script loads keys exactly the same way the main app does, so a
"✅ OK" result means the app should start without a key-related error.
"""

import os
import sys

# Load .env and config_secrets.json the same way config.py does
from dotenv import load_dotenv

load_dotenv()

# Import after load_dotenv so the same values are visible
from config import (  # noqa: E402  (import after load_dotenv is intentional)
    GEMINI_API_KEY,
    validate_gemini_key,
    _GEMINI_KEY_TYPOS,
    _SECRETS,
)

_SEP = "─" * 60


def _check_typos() -> None:
    """Warn if a common GEMINI_API_KEY misspelling is set."""
    typos_found = []
    for typo in _GEMINI_KEY_TYPOS:
        if os.environ.get(typo):
            typos_found.append(typo)
    if typos_found:
        print("\n⚠️  Found misspelled environment variable(s):")
        for typo in typos_found:
            print(f"   • {typo}  (the app does not read this — use GEMINI_API_KEY)")


def _source_description() -> str:
    """Describe where the key was loaded from."""
    if os.environ.get("GEMINI_API_KEY"):
        return "environment variable"
    if _SECRETS.get("GEMINI_API_KEY"):
        return "config_secrets.json"
    return "not found in any source"


def main() -> int:
    print(_SEP)
    print("  Taunggyi English Tutor — API Key Diagnostic")
    print(_SEP)

    source = _source_description()
    print(f"\n📂 Key source : {source}")

    # Mask the key for display (show first 8 chars + asterisks)
    if GEMINI_API_KEY:
        display = GEMINI_API_KEY[:8] + "*" * max(0, len(GEMINI_API_KEY) - 8)
    else:
        display = "(not set)"
    print(f"🔑 Key preview: {display}")

    is_valid, message = validate_gemini_key(GEMINI_API_KEY)

    if is_valid:
        print(f"\n✅ Key status  : OK — the app should connect without a key error.")
        _check_typos()
        print()
        print("You do NOT need to re-add the API key.")
        print("Start the app with:  streamlit run main.py")
        print(_SEP)
        return 0
    else:
        print(f"\n❌ Key status  : PROBLEM DETECTED")
        print(f"\n   {message}")
        _check_typos()
        print()
        print("Fix the issue above, then re-run this script to confirm.")
        print("Setup instructions: see README.md → Configure API Keys")
        print(_SEP)
        return 1


if __name__ == "__main__":
    sys.exit(main())
