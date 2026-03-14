from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from .config import EmyConfig
from .embeddings import EmbeddingIndex
from .retriever import HybridRetriever

logger = logging.getLogger(__name__)

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
