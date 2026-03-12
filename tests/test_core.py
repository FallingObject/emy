"""Unit tests for Emy v2 — vault and utils (no Ollama required)."""

from pathlib import Path

from emy.config import EmyConfig
from emy.types import VaultEntry
from emy.utils import is_greeting, new_id, sliding_chunks
from emy.vault import MarkdownVault


# ── utils ─────────────────────────────────────────────────────────────

def test_is_greeting():
    assert is_greeting("hi") is True
    assert is_greeting("Hello!") is True
    assert is_greeting("hey there") is True
    assert is_greeting("good morning") is True
    assert is_greeting("compare these methods") is False
    assert is_greeting("what is machine learning?") is False


def test_new_id():
    a = new_id("test")
    b = new_id("test")
    assert a.startswith("test_")
    assert a != b


def test_sliding_chunks():
    text = "word " * 200  # ~1000 chars
    chunks = sliding_chunks(text.strip(), chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_sliding_chunks_short():
    chunks = sliding_chunks("short text", chunk_size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == "short text"


# ── vault ─────────────────────────────────────────────────────────────

def test_vault_roundtrip(tmp_path: Path):
    vault = MarkdownVault(tmp_path / "vault")

    vault.write_entry("facts.md", "preferences", "fact_001", {
        "content": "concise answers",
        "confidence": "0.9",
    })
    vault.write_entry("facts.md", "constraints", "fact_002", {
        "content": "cite sources",
        "confidence": "0.85",
    })
    vault.write_entry("facts.md", "preferences", "fact_003", {
        "content": "use examples",
        "confidence": "0.8",
    })

    entries = vault.read_entries("facts.md")
    assert "preferences" in entries
    assert "constraints" in entries
    assert len(entries["preferences"]) == 2
    assert len(entries["constraints"]) == 1

    # Check autosort: entries sorted by id
    pref_ids = [e.id for e in entries["preferences"]]
    assert pref_ids == sorted(pref_ids)


def test_vault_update(tmp_path: Path):
    vault = MarkdownVault(tmp_path / "vault")
    vault.write_entry("facts.md", "test", "f1", {"content": "old"})
    vault.write_entry("facts.md", "test", "f1", {"content": "new"})
    entries = vault.all_entries("facts.md")
    assert len(entries) == 1
    assert entries[0].fields["content"] == "new"


def test_vault_delete(tmp_path: Path):
    vault = MarkdownVault(tmp_path / "vault")
    vault.write_entry("facts.md", "test", "f1", {"content": "one"})
    vault.write_entry("facts.md", "test", "f2", {"content": "two"})
    assert len(vault.all_entries("facts.md")) == 2
    vault.delete_entry("facts.md", "f1")
    assert len(vault.all_entries("facts.md")) == 1


def test_vault_file_creation(tmp_path: Path):
    vault = MarkdownVault(tmp_path / "vault")
    assert (tmp_path / "vault" / "facts.md").exists()
    assert (tmp_path / "vault" / "intents.md").exists()
    assert (tmp_path / "vault" / "reflections.md").exists()
    assert (tmp_path / "vault" / "entities.md").exists()
    assert (tmp_path / "vault" / "index.md").exists()


# ── config ────────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = EmyConfig()
    assert cfg.llm_model == "qwen2.5:7b"
    assert cfg.embedding_model == "nomic-embed-text"
    assert cfg.embedding_dim == 768
    assert cfg.mode == "deploy"
    assert cfg.vault_dir == cfg.workdir / "memory_vault"
