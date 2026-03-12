# Emy v2

Memory-first agentic RAG built on Ollama, web search, reflection scoring, and a human-auditable Markdown vault.

Emy treats the LLM as a reasoning engine, not a state container. All persistent state lives in Markdown files on disk, compiled into FAISS + SQLite indexes for speed. The agent runs a bounded ReAct loop with tool calling, scores its own responses, and stores reflections that improve future routing.

## What Emy does

- **Bounded agent loop** — Ollama tool calling with a max-step limit and automatic fallback if the model doesn't support tools
- **Hybrid retrieval** — BM25 + dense (FAISS) with Reciprocal Rank Fusion merging
- **Markdown memory vault** — facts, intents, reflections, and entities stored as autosorted Markdown (git-diffable, human-auditable)
- **Web search + web fetch** — DuckDuckGo search and page fetching as agent tools
- **Reflection scoring** — LLM-as-judge evaluation (faithfulness, relevance, completeness) stored as scored lessons
- **Three modes** — `train` (active learning + scoring), `deploy` (scoring + reflections), `locked` (read-only)
- **Training API** — FastAPI endpoints so a stronger LLM can teach Emy via HTTP
- **Gradio UI + CLI** — both served from the same process

## Architecture

```text
User --> Gradio UI / CLI / API
            |
       Orchestrator (bounded ReAct loop)
            |
    +-------+-------+-------+
    |       |       |       |
  Memory  Docs    Web     Web
  Search  Search  Search  Fetch
    |       |       |       |
    +---+---+---+---+-------+
        |
    Evidence Pack --> LLM (Ollama, tool calling)
        |
    Answer + Attributions
        |
    Reflection Scorer (LLM-as-judge)
        |
    Markdown Vault + FAISS Index + Episode Log
```

## Project structure

```text
emy/
  __init__.py          # exports Emy, EmyConfig
  config.py            # pydantic-settings with EMY_* env vars
  llm.py               # Ollama via OpenAI SDK (chat + embeddings)
  vault.py             # Markdown vault with deterministic autosort
  embeddings.py        # FAISS + SQLite (WAL) vector index
  retriever.py         # hybrid BM25 + dense retrieval with RRF
  tools.py             # search_memory, search_documents, web_search, web_fetch
  scoring.py           # LLM-as-judge reflection scoring
  orchestrator.py      # bounded ReAct agent loop
  api.py               # FastAPI training + inference endpoints
  ui.py                # Gradio UI
  cli.py               # CLI: chat, serve, ingest, status
  types.py             # pydantic models
  utils.py             # shared utilities
tests/
  test_core.py         # unit tests (no Ollama required)
notebooks/
  01_quickstart.ipynb   # Colab-ready quickstart
data/
  sample_docs/          # example corpus for ingestion
runtime/                # generated at runtime
  memory_vault/         # Markdown vault (source of truth)
    facts.md
    intents.md
    reflections.md
    entities.md
    index.md
  indexes/              # FAISS + SQLite (rebuildable)
  logs/                 # JSONL episode log
```

## Install

### Prerequisites

Ollama must be running with the required models pulled:

```bash
# Install Ollama: https://ollama.com
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### Install Emy

```bash
git clone https://github.com/FallingObject/emy.git
cd emy
pip install -e .

# With PDF/DOCX support:
pip install -e ".[docs]"

# With dev tools (pytest, jupyter):
pip install -e ".[dev]"
```

## Quick start

### CLI chat

```bash
emy chat --mode train
```

Commands inside the chat: `/quit`, `/mode <train|deploy|locked>`, `/fact category=value`, `/reflect <lesson>`, `/ingest <dir>`, `/status`

### Gradio UI + API server

```bash
emy serve --mode train --port 7860
```

- Gradio UI: `http://localhost:7860`
- API docs: `http://localhost:7860/docs`
- Training API: `http://localhost:7860/api/train/bulk`

### Ingest documents

```bash
emy ingest ./data/sample_docs
```

### Python API

```python
from emy import Emy, EmyConfig

emy = Emy(EmyConfig(mode="train"))
emy.ingest("./data/sample_docs")

resp = emy.respond("What do the docs say about memory types?")
print(resp.answer)
print(resp.tools_used)  # e.g. ['search_documents']
print(resp.score)       # faithfulness, relevance, completeness
```

## Modes

| Mode | Writes | Scores | Use case |
|------|--------|--------|----------|
| `train` | facts, intents, reflections, episodes | yes | Active learning, teaching Emy |
| `deploy` | reflections, episodes | yes | Production with self-improvement |
| `locked` | none | no | Stable deployment, no disk writes |

## Training API

The training API lets a stronger LLM teach Emy over HTTP. Start the server, then send training data:

```python
import httpx

API = "http://localhost:7860"

# Bulk training — examples, facts, and reflections in one call
httpx.post(f"{API}/api/train/bulk", json={
    "examples": [
        {"text": "compare these two approaches", "label": "research"},
        {"text": "good morning", "label": "smalltalk"},
    ],
    "facts": [
        {"category": "domain", "content": "This project focuses on agentic RAG"},
    ],
    "reflections": [
        {"lesson": "Use web search when docs lack recent data"},
    ],
})

# Chat with Emy and evaluate responses
resp = httpx.post(f"{API}/api/chat", json={"message": "What is RAG?"})
answer = resp.json()

# Send feedback based on evaluation
httpx.post(f"{API}/api/train/feedback", json={
    "lesson": "Include a concrete example when explaining concepts",
    "trigger": "conceptual question",
    "score": 0.6,
})
```

### All API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/chat` | Chat with Emy |
| POST | `/api/train/example` | Add a labeled intent example |
| POST | `/api/train/feedback` | Add a reflection/lesson |
| POST | `/api/train/bulk` | Bulk add examples + facts + reflections |
| POST | `/api/memory/facts` | Add a fact |
| POST | `/api/memory/entities` | Add an entity |
| GET | `/api/memory/vault/{file}` | Read vault file entries |
| POST | `/api/ingest` | Ingest a document corpus |
| POST | `/api/mode` | Switch mode (train/deploy/locked) |
| GET | `/api/status` | System status |

## Configuration

All settings can be overridden with `EMY_` environment variables or passed to `EmyConfig`:

| Setting | Default | Env var |
|---------|---------|---------|
| `ollama_base_url` | `http://localhost:11434` | `EMY_OLLAMA_BASE_URL` |
| `llm_model` | `qwen2.5:7b` | `EMY_LLM_MODEL` |
| `embedding_model` | `nomic-embed-text` | `EMY_EMBEDDING_MODEL` |
| `embedding_dim` | `768` | `EMY_EMBEDDING_DIM` |
| `mode` | `deploy` | `EMY_MODE` |
| `max_tool_calls` | `5` | `EMY_MAX_TOOL_CALLS` |
| `max_web_results` | `3` | `EMY_MAX_WEB_RESULTS` |
| `max_fetch_chars` | `4000` | `EMY_MAX_FETCH_CHARS` |

## Colab

The notebook at `notebooks/01_quickstart.ipynb` auto-installs Ollama on Colab and walks through the full train-to-deploy flow. Use `qwen2.5:3b` if the free tier runs out of GPU memory.

## Known issues and mitigations

### Tool calling model compatibility

Not all Ollama models support tool calling reliably. `qwen2.5:7b` works well. If a model fails tool calling, Emy automatically falls back to plain chat without tools — it still answers, just without tool-augmented evidence.

**Affected**: models like `gemma2`, older `llama2` variants
**Mitigation**: built-in fallback to `chat_plain()`, logged in trace

### Embedding model switching breaks indexes

FAISS indexes are tied to the embedding dimension. Switching from `nomic-embed-text` (768-dim) to `mxbai-embed-large` (1024-dim) will crash on search. You must rebuild indexes after changing embedding models.

**Mitigation**: `emy ingest <dir>` rebuilds the document index. Memory index entries need to be re-embedded manually (not yet automated — see roadmap).

### DuckDuckGo rate limiting

`duckduckgo-search` has no API key and can be rate-limited, especially on shared Colab IPs. When this happens, `web_search` returns an error message and the agent continues with other tools.

**Mitigation**: exception handling returns a graceful error string to the LLM, which can proceed without web results.

### Reflection scoring doubles latency

Every non-greeting response triggers a second LLM call for scoring. On slow hardware this roughly doubles response time.

**Mitigation**: scoring only runs when tools were used (greetings skip it). For latency-sensitive deployments, use `locked` mode which disables scoring entirely.

### Web fetch prompt injection risk

Fetched web pages could contain adversarial text that attempts to override system instructions. Emy strips HTML tags (script, style, nav, etc.) but does not sanitize against text-level prompt injection.

**Mitigation**: HTML sanitization via BeautifulSoup, content truncation to `max_fetch_chars`. For high-security use, disable `web_fetch` by removing it from the tool definitions.

### No API authentication

The training API has no auth. Anyone on the network can send training data or switch modes.

**Mitigation**: bind to `127.0.0.1` for local-only access (`emy serve --host 127.0.0.1`). For production, put a reverse proxy with auth in front.

### FAISS IndexFlatIP scalability

`IndexFlatIP` does brute-force search. Fine for <100k vectors, but slows down significantly beyond that.

**Mitigation**: sufficient for most local RAG use cases. If you need millions of vectors, swap to `IndexIVFFlat` or `IndexHNSWFlat` in `embeddings.py`.

### Memory reflection bloat

Reflections accumulate without pruning. Over many sessions the vault and memory index grow unbounded.

**Mitigation**: not yet implemented — see roadmap. Workaround: periodically review `reflections.md` and delete low-value entries manually (the vault is just Markdown).

## Roadmap

- [ ] **Embedding migration command** — `emy migrate-embeddings` to re-embed all vault entries when switching models
- [ ] **Reflection pruning** — age out low-score reflections or merge similar ones
- [ ] **Cross-encoder reranker** — optional BGE reranker for top-k refinement after hybrid retrieval
- [ ] **Async scoring** — run reflection scoring in background to halve response latency
- [ ] **API authentication** — token-based auth for the training endpoints
- [ ] **Multi-user sessions** — per-user memory namespaces
- [ ] **Streaming responses** — SSE streaming from the Gradio UI and API
- [ ] **Intent router upgrade** — use memory index embeddings for nearest-centroid routing instead of heuristics
- [ ] **RAGAS evaluation harness** — automated benchmarking of retrieval + generation quality
- [ ] **Configurable tool set** — enable/disable tools per deployment (e.g., no web access in air-gapped environments)

## Running tests

```bash
pip install -e ".[dev]"
pytest -v
```

Tests cover the vault, utils, and config — no Ollama required.

## License

MIT
