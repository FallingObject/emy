from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# Literals used as string fields (Pydantic v2 accepts plain str fields too,
# but we keep the Literals for documentation clarity and runtime validation).
JobStatus = Literal[
    "queued", "running", "paused", "blocked",
    "completed", "failed", "canceled", "expired",
]
JobPhase = Literal[
    "planning", "research", "writing", "critiquing", "refining", "finalizing",
]


class ResearchJobSpec(BaseModel):
    objective: str
    deliverable_type: str = "markdown_report"
    deadline_at: str  # ISO-8601
    time_budget_s: int = 3600
    # role -> model name override (empty = use config.llm_model)
    models: dict[str, str] = Field(default_factory=dict)
    # budget knobs: max_cycles, max_web_searches_per_cycle, etc.
    budgets: dict[str, Any] = Field(default_factory=dict)


class InterventionRequest(BaseModel):
    kind: Literal["captcha", "paywall", "rate_limit", "blocked", "other"]
    url: str
    message: str
    created_at: str


class JobMetrics(BaseModel):
    grounding: float = 0.0
    coverage: float = 0.0
    clarity: float = 0.0
    timeliness: float = 0.0
    overall: float = 0.0
    cycle_history: list[dict[str, Any]] = Field(default_factory=list)


class ResearchJobState(BaseModel):
    job_id: str
    spec: ResearchJobSpec
    status: str = "queued"   # JobStatus
    phase: str = "planning"  # JobPhase
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    last_checkpoint_at: Optional[str] = None
    worker_lease: Optional[dict[str, str]] = None

    cycle: int = 0
    outline: dict[str, Any] = Field(default_factory=dict)
    questions_backlog: list[dict[str, Any]] = Field(default_factory=list)
    metrics: JobMetrics = Field(default_factory=JobMetrics)
    blocked: Optional[InterventionRequest] = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


# ── API request/response models ───────────────────────────────────────

class JobCreateRequest(BaseModel):
    objective: str
    deliverable_type: str = "markdown_report"
    time_budget_s: int = 3600
    deadline_at: Optional[str] = None  # derived from time_budget_s if omitted
    models: dict[str, str] = Field(default_factory=dict)
    budgets: dict[str, Any] = Field(default_factory=dict)


class JobSummary(BaseModel):
    job_id: str
    status: str
    phase: str
    objective: str
    created_at: str
    deadline_at: str
    finished_at: Optional[str] = None
    cycle: int = 0
    overall_score: float = 0.0


class InterventionResolveRequest(BaseModel):
    action: Literal["skip", "resume"] = "resume"
    # optional manual evidence text (user pasted content from blocked URL)
    manual_content: Optional[str] = None
    manual_source_name: Optional[str] = None
