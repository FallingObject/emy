from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RouteResult(BaseModel):
    label: str
    confidence: float = 0.0
    should_retrieve: bool = False
    should_web_search: bool = False
    reason: str = ""


class RetrievalChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoringResult(BaseModel):
    faithfulness: float = 0.5
    relevance: float = 0.5
    completeness: float = 0.5
    overall: float = 0.5
    lesson: str = ""


class AgentResponse(BaseModel):
    answer: str
    route: RouteResult | None = None
    sources: list[RetrievalChunk] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    score: ScoringResult | None = None
    needs_label: bool = False


class VaultEntry(BaseModel):
    id: str
    category: str
    fields: dict[str, str] = Field(default_factory=dict)
