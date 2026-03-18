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

    # Vision (placeholder — future release will use a vision-capable model)
    vision_model: str = "llava:7b"
    vision_fallback_enabled: bool = False

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

    # Job runtime
    job_poll_interval_s: float = 2.0
    job_checkpoint_interval_s: int = 300
    job_max_cycles: int = 60
    job_finalization_grace_s: int = 300
    job_planner_model: str = ""
    job_writer_model: str = "gemma3:12b"
    job_critic_model: str = "deepseek-r1:8b"

    @property
    def vault_dir(self) -> Path:
        return self.workdir / "memory_vault"

    @property
    def logs_dir(self) -> Path:
        return self.workdir / "logs"

    @property
    def indexes_dir(self) -> Path:
        return self.workdir / "indexes"

    @property
    def jobs_dir(self) -> Path:
        return self.workdir / "jobs"
