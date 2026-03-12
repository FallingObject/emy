from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .orchestrator import Emy


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
        title="Emy v2 API",
        version="2.0.0",
        description=(
            "Training and inference API for Emy — a memory-first agentic RAG core. "
            "Connect a stronger LLM to these endpoints to train Emy."
        ),
    )

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

    return api
