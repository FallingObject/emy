from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class EmyConfig(BaseSettings):
    model_config = {"env_prefix": "EMY_"}

    # Ollama connection
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # Paths
    workdir: Path = Path("./runtime")

    # Agent behaviour
    mode: str = "deploy"  # train | deploy | locked
    max_tool_calls: int = 5
    max_retrieval_hits: int = 5
    max_memory_hits: int = 5
    max_web_results: int = 3
    max_fetch_chars: int = 4000
    chunk_size: int = 500
    chunk_overlap: int = 100
    uncertainty_margin: float = 0.15
    session_history_turns: int = 8

    # Server
    host: str = "0.0.0.0"
    port: int = 7860

    @property
    def vault_dir(self) -> Path:
        return self.workdir / "memory_vault"

    @property
    def logs_dir(self) -> Path:
        return self.workdir / "logs"

    @property
    def indexes_dir(self) -> Path:
        return self.workdir / "indexes"
