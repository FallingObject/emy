from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .jobs_types import ResearchJobState
from .utils import append_jsonl, ensure_dir, now_iso


class JobStore:
    """File-backed job persistence layer.

    Layout::

        jobs_dir/
          <job_id>/
            job.json       — authoritative job state (pydantic dump)
            runs.jsonl     — append-only event log
            report.md      — evolving report draft
            sources.json   — canonical source list
            outline.json   — outline + coverage targets
            metrics.json   — critic scores per cycle
            notes.md       — running claims / notes
            checkpoints/   — periodic report snapshots
            attachments/   — user-supplied manual evidence
    """

    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = jobs_dir
        ensure_dir(jobs_dir)
        self._lock = threading.Lock()

    # ── directory helpers ─────────────────────────────────────────────

    def job_dir(self, job_id: str) -> Path:
        d = self.jobs_dir / job_id
        ensure_dir(d)
        return d

    def attachments_dir(self, job_id: str) -> Path:
        d = self.jobs_dir / job_id / "attachments"
        ensure_dir(d)
        return d

    def checkpoints_dir(self, job_id: str) -> Path:
        d = self.jobs_dir / job_id / "checkpoints"
        ensure_dir(d)
        return d

    # ── state CRUD ────────────────────────────────────────────────────

    def load(self, job_id: str) -> ResearchJobState:
        path = self.jobs_dir / job_id / "job.json"
        if not path.exists():
            raise FileNotFoundError(f"Job {job_id!r} not found")
        return ResearchJobState.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, state: ResearchJobState) -> None:
        with self._lock:
            jdir = self.job_dir(state.job_id)
            (jdir / "job.json").write_text(
                state.model_dump_json(indent=2), encoding="utf-8"
            )

    def list_jobs(self, status: str | None = None) -> list[ResearchJobState]:
        jobs: list[ResearchJobState] = []
        if not self.jobs_dir.exists():
            return jobs
        for d in sorted(self.jobs_dir.iterdir()):
            if not d.is_dir():
                continue
            try:
                state = self.load(d.name)
                if status is None or state.status == status:
                    jobs.append(state)
            except Exception:
                pass
        return jobs

    # ── event log ─────────────────────────────────────────────────────

    def append_event(self, job_id: str, event: dict[str, Any]) -> None:
        event.setdefault("ts", now_iso())
        append_jsonl(self.jobs_dir / job_id / "runs.jsonl", event)

    def tail_events(self, job_id: str, n: int = 50) -> list[dict[str, Any]]:
        path = self.jobs_dir / job_id / "runs.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        result: list[dict[str, Any]] = []
        for line in lines[-n:]:
            try:
                result.append(json.loads(line))
            except Exception:
                pass
        return result

    # ── artifact helpers ──────────────────────────────────────────────

    def read_artifact(self, job_id: str, name: str) -> str:
        path = self.jobs_dir / job_id / name
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")

    def write_artifact(self, job_id: str, name: str, content: str) -> None:
        (self.job_dir(job_id) / name).write_text(content, encoding="utf-8")

    def write_checkpoint(self, job_id: str, cycle: int, report_md: str) -> None:
        cp = self.checkpoints_dir(job_id) / f"report_{cycle:04d}.md"
        cp.write_text(report_md, encoding="utf-8")

    def list_artifacts(self, job_id: str) -> list[str]:
        jdir = self.jobs_dir / job_id
        if not jdir.exists():
            return []
        return [
            p.name
            for p in sorted(jdir.iterdir())
            if p.is_file() and p.name != "job.json"
        ]

    def save_attachment(self, job_id: str, name: str, content: str) -> str:
        """Save user-supplied manual evidence; returns the saved file path."""
        dest = self.attachments_dir(job_id) / name
        dest.write_text(content, encoding="utf-8")
        return str(dest)
