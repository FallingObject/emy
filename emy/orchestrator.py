from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from .config import EmyConfig
from .embeddings import EmbeddingIndex
from .llm import LLMClient
from .retriever import HybridRetriever
from .scoring import ReflectionScorer
from .tools import TOOL_DEFINITIONS, ToolExecutor
from .types import AgentResponse, ScoringResult
from .utils import append_jsonl, is_greeting, new_id
from .vault import MarkdownVault

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are Emy, a memory-first agentic RAG assistant running locally via Ollama.

You have tools for searching your memory vault, searching indexed documents,
searching the web, and fetching web pages.

Guidelines:
- Use tools to gather evidence before answering substantive questions.
- For simple greetings, respond directly without tools.
- Cite your sources when using retrieved information.
- If you don't have enough information, say so honestly.
- Keep responses clear, practical, and grounded in evidence.
{memory_context}"""


class Emy:
    """Main Emy v2 orchestrator — bounded ReAct agent loop."""

    def __init__(self, config: Optional[EmyConfig] = None):
        self.config = config or EmyConfig()
        self._setup_dirs()

        self.llm = LLMClient(self.config)
        self.vault = MarkdownVault(self.config.vault_dir)

        self.memory_index = EmbeddingIndex(self.config, self.llm, namespace="memory")
        self.doc_index = EmbeddingIndex(self.config, self.llm, namespace="documents")
        self.retriever = HybridRetriever(self.config, self.llm, self.doc_index)

        self.tools = ToolExecutor(self.config, self.memory_index, self.retriever)
        self.scorer = ReflectionScorer(self.llm)

        self.sessions: dict[str, list[dict]] = defaultdict(list)
        self.mode: str = self.config.mode

    def _setup_dirs(self) -> None:
        for d in (
            self.config.workdir,
            self.config.vault_dir,
            self.config.logs_dir,
            self.config.indexes_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ── main entry point ──────────────────────────────────────────────

    def respond(
        self,
        query: str,
        session_id: str = "default",
        mode: str | None = None,
    ) -> AgentResponse:
        mode = mode or self.mode
        trace: list[str] = [f"mode={mode}"]

        # Greeting fast-path
        if is_greeting(query):
            answer = "Hello! I'm Emy, your memory-first research assistant. How can I help you?"
            trace.append("route=greeting (heuristic)")
            self._log_episode(session_id, query, answer, "greeting", [], None, mode)
            return AgentResponse(answer=answer, trace=trace)

        # Build memory context for the system prompt
        memory_ctx = self._build_memory_context(query)
        system = SYSTEM_PROMPT.format(memory_context=memory_ctx)

        # Conversation history
        history = self.sessions[session_id][-(self.config.session_history_turns * 2) :]
        messages: list[dict] = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})

        # ── bounded tool-calling loop ─────────────────────────────────
        tools_used: list[str] = []
        evidence_parts: list[str] = []
        answer = ""

        for step in range(self.config.max_tool_calls):
            try:
                response = self.llm.chat(messages, tools=TOOL_DEFINITIONS)
            except Exception as exc:
                # Model may not support tool calling — fall back to plain chat
                logger.warning("Tool-calling failed, falling back to plain chat: %s", exc)
                trace.append(f"tool_fallback={exc}")
                response = self.llm.chat_plain(messages)

            choice = response.choices[0]

            # If the model wants to call tools
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(choice.message)
                for tc in choice.message.tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    trace.append(f"tool[{step}]: {tool_name}({json.dumps(args)[:80]})")
                    result = self.tools.execute(tool_name, args)
                    tools_used.append(tool_name)
                    evidence_parts.append(result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
            else:
                # Final answer
                answer = choice.message.content or ""
                break
        else:
            # Ran out of steps — try to get a final answer
            try:
                final = self.llm.chat_plain(messages)
                answer = final.choices[0].message.content or ""
            except Exception:
                answer = "I ran out of steps while researching. Please try a more specific question."

        trace.append(f"tools_used={tools_used}")

        # ── reflection scoring ────────────────────────────────────────
        score: ScoringResult | None = None
        if mode in ("train", "deploy") and tools_used:
            try:
                evidence_text = "\n".join(evidence_parts)
                score = self.scorer.score(query, answer, evidence_text, tools_used)
                trace.append(f"score={score.overall:.2f}")
                if mode != "locked" and score.lesson and score.lesson.lower() != "none":
                    self._store_reflection(query, tools_used, score)
            except Exception as exc:
                trace.append(f"scoring_error={exc}")

        # ── persist episode ───────────────────────────────────────────
        self._log_episode(session_id, query, answer, "agent", tools_used, score, mode)
        self.sessions[session_id].append({"role": "user", "content": query})
        self.sessions[session_id].append({"role": "assistant", "content": answer})

        # ── retrieve source chunks for the response model ─────────────
        sources = []
        if "search_documents" in tools_used:
            sources = self.retriever.search(query)

        return AgentResponse(
            answer=answer,
            sources=sources,
            tools_used=tools_used,
            trace=trace,
            score=score,
        )

    # ── memory context builder ────────────────────────────────────────

    def _build_memory_context(self, query: str) -> str:
        parts: list[str] = []

        # Vault facts
        facts = self.vault.all_entries("facts.md")
        if facts:
            parts.append("\nKnown facts:")
            for e in facts[:10]:
                parts.append(f"  - [{e.category}] {e.fields.get('content', '')}")

        # Top reflections by similarity
        try:
            reflections = self.memory_index.search(query, top_k=3)
            reflection_lines = [
                r["text"][:200]
                for r in reflections
                if r["source"] == "reflections"
            ]
            if reflection_lines:
                parts.append("\nRelevant reflections:")
                for line in reflection_lines:
                    parts.append(f"  - {line}")
        except Exception:
            pass

        return "\n".join(parts) if parts else ""

    # ── persistence helpers ───────────────────────────────────────────

    def _store_reflection(
        self,
        query: str,
        tools_used: list[str],
        score: ScoringResult,
    ) -> None:
        entry_id = new_id("refl")
        self.vault.write_entry(
            "reflections.md",
            "auto",
            entry_id,
            {
                "trigger": query[:100],
                "score": f"{score.overall:.2f}",
                "lesson": score.lesson,
                "tools": ", ".join(tools_used),
                "created": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.memory_index.add(
            [f"Reflection: {score.lesson}"],
            [entry_id],
            source="reflections",
        )

    def _log_episode(
        self,
        session_id: str,
        query: str,
        answer: str,
        route: str,
        tools_used: list[str],
        score: ScoringResult | None,
        mode: str,
    ) -> None:
        if mode == "locked":
            return
        episode = {
            "id": new_id("ep"),
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "answer": answer[:500],
            "route": route,
            "tools_used": tools_used,
            "score": score.model_dump() if score else None,
            "mode": mode,
        }
        append_jsonl(self.config.logs_dir / "episodes.jsonl", episode)

    # ── training / management API ─────────────────────────────────────

    def ingest(self, corpus_dir: str) -> int:
        return self.retriever.ingest(corpus_dir)

    def add_fact(self, category: str, content: str, confidence: float = 0.9) -> str:
        entry_id = new_id("fact")
        self.vault.write_entry(
            "facts.md",
            category,
            entry_id,
            {
                "content": content,
                "confidence": str(confidence),
                "created": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.memory_index.add([content], [entry_id], source="facts")
        return entry_id

    def add_training_example(
        self, text: str, label: str, source: str = "api"
    ) -> str:
        entry_id = new_id("intent")
        self.vault.write_entry(
            "intents.md",
            label,
            entry_id,
            {
                "term": text,
                "source": source,
                "created": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.memory_index.add([text], [entry_id], source="intents")
        return entry_id

    def add_reflection(
        self, lesson: str, trigger: str = "manual", score: float = 0.5
    ) -> str:
        entry_id = new_id("refl")
        self.vault.write_entry(
            "reflections.md",
            "manual",
            entry_id,
            {
                "trigger": trigger,
                "score": str(score),
                "lesson": lesson,
                "created": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.memory_index.add(
            [f"Reflection: {lesson}"], [entry_id], source="reflections"
        )
        return entry_id

    def add_entity(self, name: str, entity_type: str, description: str) -> str:
        entry_id = new_id("ent")
        self.vault.write_entry(
            "entities.md",
            entity_type,
            entry_id,
            {
                "name": name,
                "description": description,
                "created": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.memory_index.add([f"{name}: {description}"], [entry_id], source="entities")
        return entry_id

    def set_mode(self, mode: str) -> None:
        if mode in ("train", "deploy", "locked"):
            self.mode = mode

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "ollama_healthy": self.llm.health_check(),
            "llm_model": self.config.llm_model,
            "embedding_model": self.config.embedding_model,
            "memory_entries": self.memory_index.total,
            "doc_entries": self.doc_index.total,
            "vault_files": {
                "facts": len(self.vault.all_entries("facts.md")),
                "intents": len(self.vault.all_entries("intents.md")),
                "reflections": len(self.vault.all_entries("reflections.md")),
                "entities": len(self.vault.all_entries("entities.md")),
            },
        }
