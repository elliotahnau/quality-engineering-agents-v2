"""LLM factory.

Stage nodes call get_llm(role) lazily inside the node function, so unit tests
can monkeypatch this module without any API key and the graph can be compiled
offline. temperature=0 everywhere: the eval harness (Phase 3) needs runs to be
as reproducible as an LLM allows.
"""

import os
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from qe_agent import config

Role = str  # "planner" | "generator" | "triager"


@lru_cache(maxsize=8)
def get_llm(role: Role = "default") -> BaseChatModel:
    provider = config.llm_provider()
    model = config.llm_model()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
        return ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0)
    raise RuntimeError(f"unknown LLM_PROVIDER: {provider}")
