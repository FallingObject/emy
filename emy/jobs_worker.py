"""Background worker that polls the job store and runs queued jobs."""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from .jobs_runner import ResearchJobRunner
from .jobs_store import JobStore

if TYPE_CHECKING:
    from .orchestrator import Emy

logger = logging.getLogger(__name__)


class JobsWorker:
    """Scans the job store for queued jobs and runs them one at a time.

    Usage::

        worker = JobsWorker(emy, store)
        thread = worker.start_in_thread()   # daemon thread, auto-stops with process
        # or:
        worker.run_forever()                # blocking, for standalone `emy worker` CLI
    """

    def __init__(
        self,
        emy: "Emy",
        store: JobStore,
        poll_interval_s: float = 2.0,
    ) -> None:
        self.emy = emy
        self.store = store
        self.poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._current_job_id: str | None = None

    # ── lifecycle ─────────────────────────────────────────────────────

    def start_in_thread(self) -> threading.Thread:
        t = threading.Thread(
            target=self.run_forever,
            daemon=True,
            name="emy-jobs-worker",
        )
        t.start()
        logger.info("Jobs worker started in background thread")
        return t

    def run_forever(self) -> None:
        logger.info("Jobs worker polling every %.1fs", self.poll_interval_s)
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as exc:
                logger.error("Worker poll error: %s", exc)
            self._stop.wait(timeout=self.poll_interval_s)
        logger.info("Jobs worker stopped")

    def stop(self) -> None:
        self._stop.set()

    @property
    def current_job_id(self) -> str | None:
        return self._current_job_id

    # ── internals ─────────────────────────────────────────────────────

    def _poll_once(self) -> None:
        for state in self.store.list_jobs(status="queued"):
            self._current_job_id = state.job_id
            try:
                runner = ResearchJobRunner(
                    llm=self.emy.llm,
                    tools=self.emy.tools,
                    store=self.store,
                    config=self.emy.config,
                )
                runner.run_until_done(state)
            except Exception as exc:
                logger.error("Job %s failed: %s", state.job_id, exc)
                try:
                    state = self.store.load(state.job_id)
                    state.status = "failed"
                    self.store.save(state)
                    self.store.append_event(state.job_id, {"event": "job_failed", "error": str(exc)})
                except Exception:
                    pass
            finally:
                self._current_job_id = None
            # process one job per poll tick
            break
