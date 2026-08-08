"""Environment-backed configuration (.env is loaded once on import)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / ".qe_runs"


def llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").lower()


def llm_model() -> str:
    return os.getenv("LLM_MODEL", "gemini-3.6-flash")


def sut_base_url() -> str:
    return os.getenv("SUT_BASE_URL", "http://127.0.0.1:8000")
