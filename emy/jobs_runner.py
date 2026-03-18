"""TRM-style iterative research job runner.

Each cycle runs: Planner → Researcher → (tool execution) → Writer → Critic → Refiner.
The loop continues until the deadline, a quality threshold, or a max-cycles cap is hit.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from .config import EmyConfig
from .jobs_roles import CriticRole, PlannerRole, RefinerRole, ResearcherRole, WriterRole
from .jobs_store import JobStore
from .jobs_types import InterventionRequest, JobMetrics, ResearchJobState
from .llm import LLMClient
from .tools import ToolExecutor
from .utils import new_id, now_iso

logger = logging.getLogger(__name__)

# ── SSRF protection ────────────────────────────────────────────────────

_BLOCKED_HOST_FRAGMENTS = [
    "localhost", "127.", "0.0.0.0", "::1",
    "10.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.",
    "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", "172.26.",
    "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
    "192.168.", "169.254.",
]


def _is_safe_url(url: str) -> bool:
    lo = url.lower()
    if not lo.startswith(("http://", "https://")):
        return False
    return not any(frag in lo for frag in _BLOCKED_HOST_FRAGMENTS)


# ── helpers ────────────────────────────────────────────────────────────

def _time_remaining_s(deadline_at: str) -> float:
    try:
        dl = datetime.fromisoformat(deadline_at.replace("Z", "+00:00"))
        return (dl - datetime.now(timezone.utc)).total_seconds()
    except Exception:
        return 0.0


def _compact_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "(none)"
    lines = []
    for s in sources[:20]:
        sid = s.get("source_id", "?")
        title = (s.get("title") or s.get("url") or "unknown")[:60]
        lines.append(f"[{sid}] {title}")
    return "\n".join(lines)


def _build_evidence_pack(results: list[dict[str, Any]]) -> str:
    if not results:
        return "(no evidence collected this cycle)"
    parts = []
    for r in results:
        sid = r.get("source_id", "?")
        source = r.get("url") or r.get("query") or r.get("source", "unknown")
        text = r.get("text", "")[:600]
        parts.append(f"[{sid}] Source: {source}\n{text}")
    return "\n\n".join(parts)


# ── Runner ─────────────────────────────────────────────────────────────

class ResearchJobRunner:
    """Runs a single ResearchJob to completion (or until blocked/expired)."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolExecutor,
        store: JobStore,
        config: EmyConfig,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.store = store
        self.config = config

    # ── public API ────────────────────────────────────────────────────

    def run_until_done(self, state: ResearchJobState) -> ResearchJobState:
        """Main entry point: run the TRM loop until a terminal condition."""
        state.status = "running"
        state.started_at = state.started_at or now_iso()
        self.store.save(state)
        self.store.append_event(state.job_id, {"event": "job_started", "cycle": state.cycle})

        # Load or initialise artifacts
        report_md = self.store.read_artifact(state.job_id, "report.md") or self._init_report(state)
        raw = self.store.read_artifact(state.job_id, "sources.json")
        sources: list[dict[str, Any]] = json.loads(raw) if raw else []

        # Inject any manual attachment evidence (user-supplied content when blocked)
        att_dir = self.store.attachments_dir(state.job_id)
        attachment_evidence: list[dict[str, Any]] = []
        for att_file in sorted(att_dir.iterdir()):
            if att_file.is_file():
                sid = new_id("S")
                attachment_evidence.append({
                    "source_id": sid,
                    "type": "manual_attachment",
                    "source": att_file.name,
                    "text": att_file.read_text(encoding="utf-8", errors="ignore")[:1200],
                })
                sources.append({"source_id": sid, "title": att_file.name, "url": None})

        # Build roles (per-job model overrides fall back to config defaults)
        mm = state.spec.models
        _m = lambda key, cfg_val: mm.get(key) or cfg_val or self.config.llm_model
        planner  = PlannerRole (self.llm, _m("planner",    self.config.job_planner_model))
        researcher = ResearcherRole(self.llm, _m("researcher", self.config.llm_model))
        writer   = WriterRole  (self.llm, _m("writer",     self.config.job_writer_model))
        critic   = CriticRole  (self.llm, _m("critic",     self.config.job_critic_model))
        refiner  = RefinerRole (self.llm, _m("refiner",    self.config.llm_model))

        # Budget defaults
        b = state.spec.budgets
        max_cycles       = int(b.get("max_cycles",                  self.config.job_max_cycles))
        max_searches     = int(b.get("max_web_searches_per_cycle",   self.config.max_web_results))
        max_fetches      = int(b.get("max_web_fetches_per_cycle",    3))
        checkpoint_ivl   = int(b.get("checkpoint_interval_s",        self.config.job_checkpoint_interval_s))
        grace_s          = int(b.get("finalization_grace_s",          self.config.job_finalization_grace_s))

        last_ckpt_ts  = time.monotonic()
        last_critic: dict[str, Any] = {}
        prev_overall  = 0.0
        plateau_cnt   = 0

        # ── main loop ─────────────────────────────────────────────────
        while state.cycle < max_cycles:
            remaining = _time_remaining_s(state.spec.deadline_at)
            if remaining <= grace_s:
                self.store.append_event(state.job_id, {"event": "deadline_approaching", "remaining_s": remaining})
                break

            state.cycle += 1
            self.store.append_event(state.job_id, {"event": "cycle_start", "cycle": state.cycle, "remaining_s": round(remaining)})

            # ── 1. Planner ────────────────────────────────────────────
            state.phase = "planning"
            try:
                plan = planner.run(
                    objective=state.spec.objective,
                    deliverable_type=state.spec.deliverable_type,
                    time_remaining_min=remaining / 60.0,
                    cycle=state.cycle,
                    outline=state.outline,
                    questions=state.questions_backlog,
                    critic_feedback=last_critic,
                )
                if plan.get("outline"):
                    state.outline = {"sections": plan["outline"]}
                if plan.get("questions"):
                    state.questions_backlog = plan["questions"]
                self.store.append_event(state.job_id, {"event": "planner_done", "cycle": state.cycle})
            except Exception as exc:
                logger.warning("Planner cycle %d: %s", state.cycle, exc)
                self.store.append_event(state.job_id, {"event": "planner_error", "cycle": state.cycle, "error": str(exc)})

            # ── 2. Researcher ─────────────────────────────────────────
            state.phase = "research"
            try:
                research_plan = researcher.run(
                    objective=state.spec.objective,
                    questions=state.questions_backlog[:6],
                    source_index_compact=_compact_sources(sources),
                    max_searches=max_searches,
                    max_fetches=max_fetches,
                )
            except Exception as exc:
                logger.warning("Researcher cycle %d: %s", state.cycle, exc)
                research_plan = {"web_search": [], "web_fetch": [], "local_doc_queries": []}

            # ── 3. Tool execution (Python-controlled) ─────────────────
            new_evidence: list[dict[str, Any]] = list(attachment_evidence)
            attachment_evidence = []  # consume once
            intervention: InterventionRequest | None = None

            for item in research_plan.get("web_search", []):
                q = (item.get("query") or "").strip()
                if not q:
                    continue
                text = self.tools._web_search(q)
                sid = new_id("S")
                new_evidence.append({"source_id": sid, "type": "web_search", "query": q, "text": text})
                self.store.append_event(state.job_id, {"event": "tool_call", "tool": "web_search", "query": q[:80]})

            for item in research_plan.get("web_fetch", []):
                url = (item.get("url") or "").strip()
                if not url:
                    continue
                if not _is_safe_url(url):
                    self.store.append_event(state.job_id, {"event": "ssrf_blocked", "url": url[:120]})
                    continue
                result = self.tools.web_fetch_structured(url)
                if result.status in ("captcha", "paywall", "blocked"):
                    intervention = InterventionRequest(
                        kind=result.status,  # type: ignore[arg-type]
                        url=url,
                        message=(
                            f"{result.status.upper()} detected at {url}. "
                            "Upload manual content or skip this source."
                        ),
                        created_at=now_iso(),
                    )
                    self.store.append_event(state.job_id, {"event": "intervention_required", "kind": result.status, "url": url[:120]})
                    break
                if result.text:
                    sid = new_id("S")
                    new_evidence.append({
                        "source_id": sid,
                        "type": "web_fetch",
                        "url": url,
                        "title": result.title or url,
                        "text": result.text[:1200],
                    })
                    sources.append({"source_id": sid, "url": url, "title": result.title or url})
                    self.store.append_event(state.job_id, {"event": "tool_call", "tool": "web_fetch", "url": url[:80]})

            for item in research_plan.get("local_doc_queries", []):
                q = (item.get("query") or "").strip()
                if not q:
                    continue
                text = self.tools._search_documents(q)
                sid = new_id("S")
                new_evidence.append({"source_id": sid, "type": "search_documents", "query": q, "text": text})
                self.store.append_event(state.job_id, {"event": "tool_call", "tool": "search_documents", "query": q[:80]})

            # Pause on intervention (CAPTCHA / paywall)
            if intervention:
                state.status = "blocked"
                state.blocked = intervention
                self.store.write_artifact(state.job_id, "report.md", report_md)
                self.store.write_artifact(state.job_id, "sources.json", json.dumps(sources, indent=2))
                self.store.save(state)
                return state

            # ── 4. Writer ─────────────────────────────────────────────
            state.phase = "writing"
            evidence_pack = _build_evidence_pack(new_evidence)
            try:
                report_md = writer.run(
                    objective=state.spec.objective,
                    deliverable_type=state.spec.deliverable_type,
                    outline=state.outline,
                    report_md=report_md,
                    evidence_pack=evidence_pack,
                )
                self.store.append_event(state.job_id, {"event": "writer_done", "cycle": state.cycle, "chars": len(report_md)})
            except Exception as exc:
                logger.warning("Writer cycle %d: %s", state.cycle, exc)
                self.store.append_event(state.job_id, {"event": "writer_error", "cycle": state.cycle, "error": str(exc)})

            # ── 5. Critic ─────────────────────────────────────────────
            state.phase = "critiquing"
            try:
                critic_result = critic.run(
                    objective=state.spec.objective,
                    cycle=state.cycle,
                    report_md=report_md,
                    sources_compact=_compact_sources(sources),
                )
                scores = critic_result.get("scores", {})
                overall = float(scores.get("overall", 0.5))
                state.metrics = JobMetrics(
                    grounding=float(scores.get("grounding", 0.5)),
                    coverage=float(scores.get("coverage", 0.5)),
                    clarity=float(scores.get("clarity", 0.5)),
                    timeliness=float(scores.get("timeliness", 0.5)),
                    overall=overall,
                    cycle_history=state.metrics.cycle_history + [{"cycle": state.cycle, "overall": overall}],
                )
                last_critic = critic_result
                self.store.append_event(state.job_id, {
                    "event": "critic_done", "cycle": state.cycle,
                    "overall": overall, "gaps": len(critic_result.get("gaps", [])),
                })

                # Early-stop: critic recommends stopping
                if critic_result.get("stop_recommendation", {}).get("stop"):
                    reason = critic_result["stop_recommendation"].get("reason", "")
                    self.store.append_event(state.job_id, {"event": "early_stop", "reason": reason})
                    break

                # Early-stop: quality plateau (< 0.02 improvement for 3 cycles)
                if abs(overall - prev_overall) < 0.02:
                    plateau_cnt += 1
                else:
                    plateau_cnt = 0
                if plateau_cnt >= 3:
                    self.store.append_event(state.job_id, {"event": "plateau_stop", "cycles": plateau_cnt})
                    break
                prev_overall = overall

            except Exception as exc:
                logger.warning("Critic cycle %d: %s", state.cycle, exc)
                self.store.append_event(state.job_id, {"event": "critic_error", "cycle": state.cycle, "error": str(exc)})

            # ── 6. Refiner ────────────────────────────────────────────
            state.phase = "refining"
            try:
                refine = refiner.run(
                    objective=state.spec.objective,
                    critic_feedback=last_critic,
                    current_questions=state.questions_backlog,
                )
                skip_ids = set(refine.get("skip_questions", []))
                new_qs = [
                    q for q in refine.get("priority_questions", state.questions_backlog)
                    if q.get("id") not in skip_ids
                ]
                if new_qs:
                    state.questions_backlog = new_qs
            except Exception as exc:
                logger.warning("Refiner cycle %d: %s", state.cycle, exc)

            # ── 7. Checkpoint ─────────────────────────────────────────
            self.store.write_artifact(state.job_id, "report.md", report_md)
            self.store.write_artifact(state.job_id, "sources.json", json.dumps(sources, indent=2))
            self.store.write_artifact(state.job_id, "metrics.json", state.metrics.model_dump_json(indent=2))
            state.last_checkpoint_at = now_iso()
            if time.monotonic() - last_ckpt_ts >= checkpoint_ivl:
                self.store.write_checkpoint(state.job_id, state.cycle, report_md)
                self.store.append_event(state.job_id, {"event": "checkpoint", "cycle": state.cycle})
                last_ckpt_ts = time.monotonic()
            self.store.save(state)

        # ── Finalise ──────────────────────────────────────────────────
        state.phase = "finalizing"
        self.store.append_event(state.job_id, {"event": "finalizing", "cycle": state.cycle})
        self.store.write_artifact(state.job_id, "report.md", report_md)
        self.store.write_artifact(state.job_id, "sources.json", json.dumps(sources, indent=2))
        self.store.write_artifact(state.job_id, "metrics.json", state.metrics.model_dump_json(indent=2))

        remaining = _time_remaining_s(state.spec.deadline_at)
        state.status = "expired" if remaining <= 0 else "completed"
        state.finished_at = now_iso()
        self.store.save(state)
        self.store.append_event(state.job_id, {"event": "job_done", "status": state.status, "cycles": state.cycle})
        return state

    def resume(self, state: ResearchJobState) -> ResearchJobState:
        """Resume a paused or blocked job."""
        if state.status not in ("paused", "blocked"):
            raise ValueError(f"Cannot resume job in status {state.status!r}")
        state.status = "running"
        state.blocked = None
        self.store.save(state)
        self.store.append_event(state.job_id, {"event": "job_resumed"})
        return self.run_until_done(state)

    def _init_report(self, state: ResearchJobState) -> str:
        return f"# {state.spec.objective}\n\n*Research in progress...*\n"
