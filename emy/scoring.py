from __future__ import annotations

import json
import logging

from .llm import LLMClient
from .types import ScoringResult

logger = logging.getLogger(__name__)

SCORE_PROMPT = """\
Score this RAG response on 3 dimensions (each 0.0-1.0).
Return ONLY a JSON object: {{"faithfulness": ..., "relevance": ..., "completeness": ..., "lesson": "..."}}

User query: {query}

Answer given: {answer}

Evidence used: {evidence}

Tools used: {tools}

Scoring criteria:
- faithfulness: Does the answer only use information from the evidence? 1.0 = fully grounded.
- relevance: Does the answer address the user's question? 1.0 = perfectly relevant.
- completeness: Does the answer fully cover the question? 1.0 = complete.
- lesson: One sentence about what could be improved (or "none" if the response was ideal).
"""


class ReflectionScorer:
    """LLM-as-judge scoring for RAG quality, inspired by RAGAS + G-Eval."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def score(
        self,
        query: str,
        answer: str,
        evidence: str,
        tools_used: list[str],
    ) -> ScoringResult:
        prompt = SCORE_PROMPT.format(
            query=query,
            answer=answer[:500],
            evidence=evidence[:500] if evidence else "None",
            tools=", ".join(tools_used) if tools_used else "None",
        )
        try:
            response = self.llm.chat_plain(
                [
                    {
                        "role": "system",
                        "content": "You are a response quality evaluator. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            text = response.choices[0].message.content or ""
            text = text.strip()
            # Strip markdown code fences if present
            if "```" in text:
                parts = text.split("```")
                text = parts[1] if len(parts) >= 2 else text
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            data = json.loads(text)
            f = float(data.get("faithfulness", 0.5))
            r = float(data.get("relevance", 0.5))
            c = float(data.get("completeness", 0.5))
            return ScoringResult(
                faithfulness=f,
                relevance=r,
                completeness=c,
                overall=round((f + r + c) / 3, 3),
                lesson=data.get("lesson", ""),
            )
        except Exception as exc:
            logger.warning("Scoring failed: %s", exc)
            return ScoringResult(lesson=f"Scoring error: {exc}")
