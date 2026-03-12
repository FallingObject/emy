from __future__ import annotations

import logging
from typing import Any

import numpy as np
from openai import OpenAI

from .config import EmyConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around Ollama via the OpenAI-compatible API."""

    def __init__(self, config: EmyConfig):
        self.config = config
        self.client = OpenAI(
            base_url=f"{config.ollama_base_url}/v1",
            api_key="ollama",
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
    ):
        kwargs: dict[str, Any] = dict(
            model=model or self.config.llm_model,
            messages=messages,
            temperature=temperature,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs)

    def chat_plain(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
    ):
        """Chat without tool calling — guaranteed to work with any model."""
        return self.client.chat.completions.create(
            model=model or self.config.llm_model,
            messages=messages,
            temperature=temperature,
        )

    def embed(self, texts: list[str] | str, model: str | None = None) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self.config.embedding_dim), dtype=np.float32)
        response = self.client.embeddings.create(
            model=model or self.config.embedding_model,
            input=texts,
        )
        vecs = [e.embedding for e in response.data]
        return np.array(vecs, dtype=np.float32)

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception as e:
            logger.warning("Ollama health check failed: %s", e)
            return False

    def available_models(self) -> list[str]:
        try:
            return [m.id for m in self.client.models.list().data]
        except Exception:
            return []
