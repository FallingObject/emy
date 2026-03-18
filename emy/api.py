from __future__ import annotations

import io
import tempfile
import threading
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .jobs_runner import ResearchJobRunner
from .jobs_store import JobStore
from .jobs_types import (
    InterventionResolveRequest,
    JobCreateRequest,
    JobSummary,
    ResearchJobSpec,
    ResearchJobState,
)
from .jobs_worker import JobsWorker
from .orchestrator import Emy
from .utils import new_id, now_iso


# ── request / response models ────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    mode: str | None = None


class TrainExampleRequest(BaseModel):
    text: str
    label: str
    source: str = "api"


class FeedbackRequest(BaseModel):
    lesson: str
    trigger: str = "api"
    score: float = 0.5


class FactRequest(BaseModel):
    category: str
    content: str
    confidence: float = 0.9


class EntityRequest(BaseModel):
    name: str
    entity_type: str
    description: str


class ModeRequest(BaseModel):
    mode: str


class IngestRequest(BaseModel):
    corpus_dir: str


class BulkTrainRequest(BaseModel):
    examples: list[TrainExampleRequest] = Field(default_factory=list)
    facts: list[FactRequest] = Field(default_factory=list)
    reflections: list[FeedbackRequest] = Field(default_factory=list)


# ── API factory ──────────────────────────────────────────────────────

def create_api(emy: Emy) -> FastAPI:
    api = FastAPI(
        title="Emy v3 API",
        version="3.0.0",
        description=(
            "Training and inference API for Emy — a memory-first agentic RAG core. "
            "Connect a stronger LLM to these endpoints to train Emy."
        ),
    )

    # ── Job store + worker (lazy-init, one per API instance) ──────────
    _job_store: JobStore | None = None
    _worker: JobsWorker | None = None
    _worker_lock = threading.Lock()

    def _get_store() -> JobStore:
        nonlocal _job_store
        if _job_store is None:
            _job_store = JobStore(emy.config.jobs_dir)
        return _job_store

    def _ensure_worker() -> None:
        nonlocal _worker
        with _worker_lock:
            if _worker is None:
                _worker = JobsWorker(emy, _get_store(), emy.config.job_poll_interval_s)
                _worker.start_in_thread()

    # Start worker when the API is created
    _ensure_worker()

    # ── chat ──────────────────────────────────────────────────────

    @api.post("/api/chat")
    def chat(req: ChatRequest):
        resp = emy.respond(req.message, session_id=req.session_id, mode=req.mode)
        return resp.model_dump()

    # ── training ──────────────────────────────────────────────────

    @api.post("/api/train/example")
    def add_example(req: TrainExampleRequest):
        eid = emy.add_training_example(req.text, req.label, req.source)
        return {"status": "ok", "id": eid, "label": req.label}

    @api.post("/api/train/feedback")
    def add_feedback(req: FeedbackRequest):
        eid = emy.add_reflection(req.lesson, trigger=req.trigger, score=req.score)
        return {"status": "ok", "id": eid}

    @api.post("/api/train/bulk")
    def bulk_train(req: BulkTrainRequest):
        ids: list[str] = []
        for ex in req.examples:
            ids.append(emy.add_training_example(ex.text, ex.label, ex.source))
        for fact in req.facts:
            ids.append(emy.add_fact(fact.category, fact.content, fact.confidence))
        for refl in req.reflections:
            ids.append(emy.add_reflection(refl.lesson, trigger=refl.trigger, score=refl.score))
        return {"status": "ok", "count": len(ids), "ids": ids}

    # ── memory management ─────────────────────────────────────────

    @api.post("/api/memory/facts")
    def add_fact(req: FactRequest):
        eid = emy.add_fact(req.category, req.content, req.confidence)
        return {"status": "ok", "id": eid}

    @api.post("/api/memory/entities")
    def add_entity(req: EntityRequest):
        eid = emy.add_entity(req.name, req.entity_type, req.description)
        return {"status": "ok", "id": eid}

    @api.get("/api/memory/vault/{filename}")
    def get_vault(filename: str):
        if not filename.endswith(".md"):
            filename += ".md"
        entries = emy.vault.read_entries(filename)
        return {
            cat: [{"id": e.id, "fields": e.fields} for e in es]
            for cat, es in entries.items()
        }

    @api.get("/api/memory/export")
    def export_memory():
        """Download all vault memory as a zip archive."""
        buf = io.BytesIO()
        vault_dir = emy.config.vault_dir
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if vault_dir.exists():
                for fp in sorted(vault_dir.rglob("*")):
                    if fp.is_file():
                        zf.write(fp, fp.relative_to(vault_dir))
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=emy_memory.zip"},
        )

    @api.post("/api/memory/import")
    async def import_memory(file: UploadFile = File(...)):
        """Upload a memory vault zip to restore/merge memory."""
        data = await file.read()
        if not zipfile.is_zipfile(io.BytesIO(data)):
            raise HTTPException(400, "Uploaded file is not a valid zip archive")
        vault_dir = emy.config.vault_dir
        vault_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            zf.extractall(vault_dir)
        return {"status": "ok", "message": "Memory vault imported successfully"}

    # ── file upload & ingest ──────────────────────────────────────

    @api.post("/api/upload")
    async def upload_files(files: list[UploadFile] = File(...)):
        """Upload files (including .zip) for ingestion."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for f in files:
                data = await f.read()
                fname = f.filename or "unknown"
                if fname.lower().endswith(".zip"):
                    try:
                        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                            zf.extractall(tmp_path)
                    except zipfile.BadZipFile:
                        raise HTTPException(400, f"Bad zip file: {fname}")
                else:
                    dest = tmp_path / fname
                    dest.write_bytes(data)
            count = emy.ingest(tmp)
        return {"status": "ok", "chunks": count}

    # ── operations ────────────────────────────────────────────────

    @api.post("/api/ingest")
    def ingest(req: IngestRequest):
        try:
            count = emy.ingest(req.corpus_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"status": "ok", "chunks": count}

    @api.post("/api/mode")
    def set_mode(req: ModeRequest):
        if req.mode not in ("train", "deploy", "locked"):
            raise HTTPException(400, "Mode must be train, deploy, or locked")
        emy.set_mode(req.mode)
        return {"status": "ok", "mode": req.mode}

    @api.get("/api/status")
    def status():
        return emy.status()

    # ── research jobs ─────────────────────────────────────────────

    @api.post("/api/jobs")
    def create_job(req: JobCreateRequest):
        """Create a new research job and queue it for the background worker."""
        if req.deadline_at:
            deadline_at = req.deadline_at
        else:
            deadline = datetime.now(timezone.utc) + timedelta(seconds=req.time_budget_s)
            deadline_at = deadline.isoformat()

        job_id = new_id("job")
        spec = ResearchJobSpec(
            objective=req.objective,
            deliverable_type=req.deliverable_type,
            deadline_at=deadline_at,
            time_budget_s=req.time_budget_s,
            models=req.models,
            budgets=req.budgets,
        )
        state = ResearchJobState(
            job_id=job_id,
            spec=spec,
            created_at=now_iso(),
        )
        store = _get_store()
        store.save(state)
        store.append_event(job_id, {"event": "job_created", "objective": req.objective[:120]})
        return state.model_dump()

    @api.get("/api/jobs")
    def list_jobs(status: str | None = None):
        jobs = _get_store().list_jobs(status=status)
        return [
            JobSummary(
                job_id=j.job_id,
                status=j.status,
                phase=j.phase,
                objective=j.spec.objective,
                created_at=j.created_at,
                deadline_at=j.spec.deadline_at,
                finished_at=j.finished_at,
                cycle=j.cycle,
                overall_score=j.metrics.overall,
            ).model_dump()
            for j in jobs
        ]

    @api.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return _get_store().load(job_id).model_dump()
        except FileNotFoundError:
            raise HTTPException(404, f"Job {job_id!r} not found")

    @api.post("/api/jobs/{job_id}/pause")
    def pause_job(job_id: str):
        store = _get_store()
        try:
            state = store.load(job_id)
        except FileNotFoundError:
            raise HTTPException(404, f"Job {job_id!r} not found")
        if state.status != "running":
            raise HTTPException(400, f"Job is {state.status!r}, cannot pause")
        state.status = "paused"
        store.save(state)
        store.append_event(job_id, {"event": "job_paused"})
        return {"status": "ok", "job_status": state.status}

    @api.post("/api/jobs/{job_id}/resume")
    def resume_job(job_id: str):
        store = _get_store()
        try:
            state = store.load(job_id)
        except FileNotFoundError:
            raise HTTPException(404, f"Job {job_id!r} not found")
        if state.status not in ("paused", "blocked"):
            raise HTTPException(400, f"Job is {state.status!r}, cannot resume")
        state.status = "queued"  # re-queue so worker picks it up
        state.blocked = None
        store.save(state)
        store.append_event(job_id, {"event": "job_requeued"})
        return {"status": "ok", "job_status": state.status}

    @api.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        store = _get_store()
        try:
            state = store.load(job_id)
        except FileNotFoundError:
            raise HTTPException(404, f"Job {job_id!r} not found")
        if state.status in ("completed", "failed", "canceled", "expired"):
            raise HTTPException(400, f"Job is already terminal: {state.status!r}")
        state.status = "canceled"
        state.finished_at = now_iso()
        store.save(state)
        store.append_event(job_id, {"event": "job_canceled"})
        return {"status": "ok", "job_status": state.status}

    @api.get("/api/jobs/{job_id}/events")
    def get_events(job_id: str, tail: int = 50):
        if not (_get_store().jobs_dir / job_id).exists():
            raise HTTPException(404, f"Job {job_id!r} not found")
        return _get_store().tail_events(job_id, n=tail)

    @api.get("/api/jobs/{job_id}/artifact/{name}")
    def get_artifact(job_id: str, name: str):
        # Prevent path traversal
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(400, "Invalid artifact name")
        content = _get_store().read_artifact(job_id, name)
        if not content:
            raise HTTPException(404, f"Artifact {name!r} not found for job {job_id!r}")
        media = "text/markdown" if name.endswith(".md") else "application/json"
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @api.get("/api/jobs/{job_id}/interventions")
    def get_intervention(job_id: str):
        try:
            state = _get_store().load(job_id)
        except FileNotFoundError:
            raise HTTPException(404, f"Job {job_id!r} not found")
        if state.blocked is None:
            return {"blocked": False}
        return {"blocked": True, "intervention": state.blocked.model_dump()}

    @api.post("/api/jobs/{job_id}/interventions/submit")
    async def submit_intervention(
        job_id: str,
        req: InterventionResolveRequest,
        file: UploadFile | None = File(default=None),
    ):
        store = _get_store()
        try:
            state = store.load(job_id)
        except FileNotFoundError:
            raise HTTPException(404, f"Job {job_id!r} not found")
        if state.status != "blocked":
            raise HTTPException(400, f"Job is {state.status!r}, not blocked")

        # Save uploaded file as manual attachment evidence
        if file is not None:
            data = await file.read()
            fname = file.filename or "manual_evidence.txt"
            store.save_attachment(job_id, fname, data.decode(errors="ignore"))
            store.append_event(job_id, {"event": "attachment_uploaded", "file": fname})
        elif req.manual_content:
            fname = req.manual_source_name or "manual_evidence.txt"
            store.save_attachment(job_id, fname, req.manual_content)
            store.append_event(job_id, {"event": "attachment_uploaded", "file": fname})

        if req.action == "skip":
            store.append_event(job_id, {"event": "intervention_skipped"})
        else:
            store.append_event(job_id, {"event": "intervention_resolved"})

        state.status = "queued"
        state.blocked = None
        store.save(state)
        return {"status": "ok", "job_status": state.status}

    return api
