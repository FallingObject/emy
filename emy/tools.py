from __future__ import annotations

import logging
import re
from typing import Literal, Optional

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from pydantic import BaseModel

from .config import EmyConfig
from .embeddings import EmbeddingIndex
from .retriever import HybridRetriever

logger = logging.getLogger(__name__)

# ── Structured web-fetch result (used by research jobs) ──────────────


class WebFetchResult(BaseModel):
    status: Literal["ok", "captcha", "paywall", "blocked", "error"]
    url: str
    http_status: Optional[int] = None
    title: Optional[str] = None
    text: str = ""
    error: Optional[str] = None


_CAPTCHA_PATTERNS = [
    r"captcha", r"are you a human", r"verify you are human",
    r"unusual traffic", r"robot check", r"i am not a robot",
    r"human verification",
]
_PAYWALL_PATTERNS = [
    r"subscribe to (read|continue|access)", r"subscription required",
    r"premium content", r"sign in to read", r"members only",
    r"unlock this article", r"create an account to continue",
]


def _detect_interstitial(text: str) -> Optional[str]:
    """Return 'captcha', 'paywall', or None."""
    lo = text.lower()
    for pat in _CAPTCHA_PATTERNS:
        if re.search(pat, lo):
            return "captcha"
    for pat in _PAYWALL_PATTERNS:
        if re.search(pat, lo):
            return "paywall"
    return None


# ── tool schemas (OpenAI function-calling format) ────────────────────

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search the agent's persistent memory vault for relevant past "
                "interactions, stored facts, and reflections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search indexed local documents for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use when local docs "
                "and memory are insufficient or the question requires recent data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch and read the text content of a web page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
]


class ToolExecutor:
    """Runs tool calls requested by the LLM."""

    def __init__(
        self,
        config: EmyConfig,
        memory_index: EmbeddingIndex,
        retriever: HybridRetriever,
    ):
        self.config = config
        self.memory_index = memory_index
        self.retriever = retriever

    def execute(self, name: str, arguments: dict) -> str:
        handlers = {
            "search_memory": self._search_memory,
            "search_documents": self._search_documents,
            "web_search": self._web_search,
            "web_fetch": self._web_fetch,
        }
        handler = handlers.get(name)
        if not handler:
            return f"Unknown tool: {name}"
        try:
            return handler(**arguments)
        except Exception as exc:
            logger.error("Tool %s failed: %s", name, exc)
            return f"Tool error: {exc}"

    # ── handlers ──────────────────────────────────────────────────────

    def _search_memory(self, query: str) -> str:
        results = self.memory_index.search(query, top_k=self.config.max_memory_hits)
        if not results:
            return "No relevant memories found."
        lines: list[str] = []
        for r in results:
            lines.append(
                f"[{r['source']}/{r['entry_id']}] (score={r['score']:.3f}): "
                f"{r['text'][:300]}"
            )
        return "\n".join(lines)

    def _search_documents(self, query: str) -> str:
        results = self.retriever.search(query)
        if not results:
            return "No relevant documents found."
        lines: list[str] = []
        for r in results:
            lines.append(f"[{r.source}] (score={r.score:.3f}): {r.text[:300]}")
        return "\n".join(lines)

    def _web_search(self, query: str) -> str:
        try:
            results = list(DDGS().text(query, max_results=self.config.max_web_results))
        except Exception as exc:
            return f"Web search failed: {exc}"
        if not results:
            return "No web results found."
        lines: list[str] = []
        for r in results:
            lines.append(f"[{r['title']}]({r['href']}): {r['body'][:200]}")
        return "\n".join(lines)

    def _web_fetch(self, url: str) -> str:
        """Simple text fetch used by the chat agent (unchanged behaviour)."""
        try:
            resp = httpx.get(
                url,
                timeout=10,
                follow_redirects=True,
                headers={"User-Agent": "Emy/3.0"},
            )
            resp.raise_for_status()
        except Exception as exc:
            return f"Web fetch failed: {exc}"
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[: self.config.max_fetch_chars]

    def web_fetch_structured(self, url: str) -> WebFetchResult:
        """Structured fetch used by the research job runner.

        Returns a :class:`WebFetchResult` with CAPTCHA/paywall detection so the
        runner can transition the job to ``blocked`` instead of silently failing.
        """
        try:
            resp = httpx.get(
                url,
                timeout=10,
                follow_redirects=True,
                headers={"User-Agent": "Emy/3.0"},
            )
        except Exception as exc:
            return WebFetchResult(status="error", url=url, error=str(exc))

        http_status = resp.status_code
        if http_status >= 400:
            return WebFetchResult(
                status="error", url=url, http_status=http_status,
                error=f"HTTP {http_status}",
            )

        soup = BeautifulSoup(resp.text, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)[: self.config.max_fetch_chars]

        interstitial = _detect_interstitial(text)
        if interstitial:
            return WebFetchResult(
                status=interstitial,  # type: ignore[arg-type]
                url=url,
                http_status=http_status,
                title=title,
                text=text[:200],
            )

        return WebFetchResult(
            status="ok",
            url=url,
            http_status=http_status,
            title=title,
            text=text,
        )
