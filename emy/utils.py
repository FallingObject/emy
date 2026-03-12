from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sliding_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


def is_greeting(text: str) -> bool:
    t = text.lower().strip()
    patterns = [
        r"^hi[!.\s]*$",
        r"^hello[!.\s]*$",
        r"^hey[!.\s]*$",
        r"^hey there[!.\s]*$",
        r"^good (morning|afternoon|evening)[!.\s]*$",
        r"^howdy[!.\s]*$",
        r"^yo[!.\s]*$",
        r"^sup[!.\s]*$",
        r"^what'?s up[!.\s?]*$",
    ]
    return any(re.match(p, t) for p in patterns)


def read_file_text(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".rst", ".log"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                return ""
        if suffix == ".docx":
            try:
                from docx import Document
                doc = Document(str(path))
                return "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                return ""
    except Exception:
        return ""
    return ""
