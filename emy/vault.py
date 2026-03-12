from __future__ import annotations

import threading
from pathlib import Path

from .types import VaultEntry

VAULT_FILES = ("facts.md", "intents.md", "reflections.md", "entities.md")


class MarkdownVault:
    """Human-auditable Markdown memory vault with deterministic autosort.

    Each file follows the pattern:
        # <title>
        ## <category>
        - id: <entry_id>
          key: value
          ...
    Categories are sorted alphabetically and entries within categories
    are sorted by id, making the files stable, diffable, and merge-friendly.
    """

    def __init__(self, vault_dir: Path):
        self.vault_dir = vault_dir
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_files()

    def _ensure_files(self) -> None:
        for name in VAULT_FILES:
            path = self.vault_dir / name
            if not path.exists():
                title = name.replace(".md", "")
                path.write_text(f"# {title}\n", encoding="utf-8")
        idx = self.vault_dir / "index.md"
        if not idx.exists():
            lines = ["# Emy Memory Vault Index\n"]
            for name in VAULT_FILES:
                lines.append(f"- [{name}](./{name})")
            idx.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── read ──────────────────────────────────────────────────────────

    def read_entries(self, filename: str) -> dict[str, list[VaultEntry]]:
        path = self.vault_dir / filename
        if not path.exists():
            return {}
        return self._parse(path.read_text(encoding="utf-8"))

    def all_entries(self, filename: str) -> list[VaultEntry]:
        entries: list[VaultEntry] = []
        for cat_entries in self.read_entries(filename).values():
            entries.extend(cat_entries)
        return entries

    # ── write ─────────────────────────────────────────────────────────

    def write_entry(
        self,
        filename: str,
        category: str,
        entry_id: str,
        fields: dict[str, str],
    ) -> None:
        with self._lock:
            categories = self.read_entries(filename)
            cat_list = categories.setdefault(category, [])
            # update existing or append
            for e in cat_list:
                if e.id == entry_id:
                    e.fields = fields
                    break
            else:
                cat_list.append(
                    VaultEntry(id=entry_id, category=category, fields=fields)
                )
            self._render(filename, categories)

    def delete_entry(self, filename: str, entry_id: str) -> bool:
        with self._lock:
            categories = self.read_entries(filename)
            found = False
            for cat_list in categories.values():
                for i, e in enumerate(cat_list):
                    if e.id == entry_id:
                        cat_list.pop(i)
                        found = True
                        break
                if found:
                    break
            if found:
                self._render(filename, categories)
            return found

    # ── parse / render ────────────────────────────────────────────────

    @staticmethod
    def _parse(content: str) -> dict[str, list[VaultEntry]]:
        categories: dict[str, list[VaultEntry]] = {}
        current_cat: str | None = None
        current_id: str | None = None
        current_fields: dict[str, str] = {}

        def _flush() -> None:
            nonlocal current_id, current_fields
            if current_id and current_cat is not None:
                categories.setdefault(current_cat, []).append(
                    VaultEntry(id=current_id, category=current_cat, fields=dict(current_fields))
                )
            current_id = None
            current_fields = {}

        for raw_line in content.split("\n"):
            line = raw_line.rstrip()
            if line.startswith("## "):
                _flush()
                current_cat = line[3:].strip()
            elif line.startswith("- id: ") and current_cat is not None:
                _flush()
                current_id = line[6:].strip()
            elif line.startswith("  ") and ":" in line and current_id is not None:
                key, _, val = line.strip().partition(":")
                current_fields[key.strip()] = val.strip()

        _flush()
        return categories

    def _render(self, filename: str, categories: dict[str, list[VaultEntry]]) -> None:
        title = filename.replace(".md", "")
        lines = [f"# {title}\n"]
        for cat_name in sorted(categories):
            lines.append(f"## {cat_name}")
            for entry in sorted(categories[cat_name], key=lambda e: e.id):
                lines.append(f"- id: {entry.id}")
                for key in sorted(entry.fields):
                    lines.append(f"  {key}: {entry.fields[key]}")
            lines.append("")
        path = self.vault_dir / filename
        path.write_text("\n".join(lines), encoding="utf-8")
