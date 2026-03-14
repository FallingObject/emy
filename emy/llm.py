from __future__ import annotations

import logging
from typing import Any

import numpy as np
import requests
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

    def vision_extract(
        self,
        base64_images: list[str],
        model: str | None = None,
        prompt: str = (
            "Extract ALL text content from this image. "
            "Preserve the original structure (headings, lists, tables) as closely as possible. "
            "Return only the extracted text, no commentary."
        ),
    ) -> str:
        """Send base64-encoded images to a vision model (e.g. LLaVA) via Ollama's native API.

        The OpenAI-compatible /v1 endpoint does not reliably support multi-image
        vision, so we call the Ollama ``/api/chat`` endpoint directly.
        """
        vision_model = model or self.config.vision_model
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": base64_images,
            }
        ]
        try:
            resp = requests.post(
                f"{self.config.ollama_base_url}/api/chat",
                json={"model": vision_model, "messages": messages, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as exc:
            logger.warning("Vision extraction failed: %s", exc)
            return ""
