from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import EmyConfig
from .orchestrator import Emy


def cmd_chat(args):
    config = EmyConfig(workdir=Path(args.workdir), mode=args.mode)
    if args.model:
        config.llm_model = args.model
    emy = Emy(config)

    if not emy.llm.health_check():
        print("WARNING: Ollama is not reachable at", config.ollama_base_url)
        print("Make sure Ollama is running: ollama serve")
        return

    print(f"Emy v3 | mode={args.mode} | model={config.llm_model}")
    print("Commands: /quit  /mode <m>  /fact <cat>=<val>  /reflect <text>  /ingest <dir>")
    print("-" * 60)

    session = "cli"
    while True:
        try:
            msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg.startswith("/mode "):
            m = msg.split(" ", 1)[1].strip()
            emy.set_mode(m)
            print(f"Mode set to: {emy.mode}")
            continue
        if msg.startswith("/fact "):
            kv = msg.split(" ", 1)[1]
            if "=" not in kv:
                print("Use: /fact category=content")
                continue
            cat, val = kv.split("=", 1)
            emy.add_fact(cat.strip(), val.strip())
            print("Fact saved.")
            continue
        if msg.startswith("/reflect "):
            text = msg.split(" ", 1)[1].strip()
            emy.add_reflection(text)
            print("Reflection saved.")
            continue
        if msg.startswith("/ingest "):
            path = msg.split(" ", 1)[1].strip()
            try:
                count = emy.ingest(path)
                print(f"Ingested {count} chunks.")
            except Exception as exc:
                print(f"Error: {exc}")
            continue
        if msg == "/status":
            print(json.dumps(emy.status(), indent=2))
            continue

        resp = emy.respond(msg, session_id=session)
        print(f"\nEmy: {resp.answer}")
        if resp.tools_used:
            print(f"\n  Tools: {', '.join(resp.tools_used)}")
        if resp.score:
            print(f"  Score: {resp.score.overall:.2f} — {resp.score.lesson}")


def cmd_serve(args):
    from .gradio_app import build_demo, build_config, get_emy, CSS

    import gradio as gr

    config = build_config(
        base_url=f"http://localhost:11434",
        workdir=args.workdir,
        learning_enabled=(args.mode == "train"),
        vision_enabled=False,
    )
    if args.model:
        config.llm_model = args.model

    demo = build_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        theme=gr.themes.Soft(),
        css=CSS,
    )


def cmd_ingest(args):
    config = EmyConfig(workdir=Path(args.workdir))
    if args.model:
        config.llm_model = args.model
    emy = Emy(config)
    count = emy.ingest(args.corpus)
    print(f"Ingested {count} chunks from {args.corpus}")


def cmd_status(args):
    config = EmyConfig(workdir=Path(args.workdir))
    emy = Emy(config)
    print(json.dumps(emy.status(), indent=2))


# ── research commands ─────────────────────────────────────────────────

def _parse_time_budget(s: str) -> int:
    """Parse '2h', '30m', '3600s', or plain integer seconds."""
    s = s.strip().lower()
    if s.endswith("h"):
        return int(float(s[:-1]) * 3600)
    if s.endswith("m"):
        return int(float(s[:-1]) * 60)
    if s.endswith("s"):
        return int(float(s[:-1]))
    return int(s)


def _make_emy(args) -> Emy:
    config = EmyConfig(workdir=Path(args.workdir))
    if hasattr(args, "model") and args.model:
        config.llm_model = args.model
    return Emy(config)


def cmd_research_submit(args):
    from .jobs_store import JobStore
    from .jobs_types import ResearchJobSpec, ResearchJobState
    from .utils import new_id, now_iso

    budget_s = _parse_time_budget(args.time_budget)
    deadline = datetime.now(timezone.utc) + timedelta(seconds=budget_s)

    models: dict[str, str] = {}
    for role in ("planner", "researcher", "writer", "critic"):
        val = getattr(args, f"{role}_model", None)
        if val:
            models[role] = val

    spec = ResearchJobSpec(
        objective=args.objective,
        deliverable_type=args.deliverable,
        deadline_at=deadline.isoformat(),
        time_budget_s=budget_s,
        models=models,
    )
    job_id = new_id("job")
    state = ResearchJobState(job_id=job_id, spec=spec, created_at=now_iso())

    config = EmyConfig(workdir=Path(args.workdir))
    store = JobStore(config.jobs_dir)
    store.save(state)
    store.append_event(job_id, {"event": "job_created", "objective": args.objective[:120]})
    print(f"Job created: {job_id}")
    print(f"Deadline:    {deadline.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Output:      {store.job_dir(job_id)}")


def cmd_research_run(args):
    """Create and run a job in the foreground (blocking)."""
    from .jobs_runner import ResearchJobRunner
    from .jobs_store import JobStore
    from .jobs_types import ResearchJobSpec, ResearchJobState
    from .utils import new_id, now_iso

    budget_s = _parse_time_budget(args.time_budget)
    deadline = datetime.now(timezone.utc) + timedelta(seconds=budget_s)

    models: dict[str, str] = {}
    for role in ("planner", "researcher", "writer", "critic"):
        val = getattr(args, f"{role}_model", None)
        if val:
            models[role] = val

    spec = ResearchJobSpec(
        objective=args.objective,
        deliverable_type=args.deliverable,
        deadline_at=deadline.isoformat(),
        time_budget_s=budget_s,
        models=models,
    )
    job_id = new_id("job")
    state = ResearchJobState(job_id=job_id, spec=spec, created_at=now_iso())

    emy = _make_emy(args)
    config = emy.config
    store = JobStore(config.jobs_dir)
    store.save(state)
    store.append_event(job_id, {"event": "job_created", "objective": args.objective[:120]})

    print(f"Running job {job_id} (deadline {deadline.strftime('%H:%M:%S UTC')}) ...")
    runner = ResearchJobRunner(llm=emy.llm, tools=emy.tools, store=store, config=config)
    final = runner.run_until_done(state)
    print(f"\nDone — status={final.status}  cycles={final.cycle}  score={final.metrics.overall:.2f}")
    print(f"Report: {store.job_dir(job_id) / 'report.md'}")


def cmd_jobs_list(args):
    from .jobs_store import JobStore

    config = EmyConfig(workdir=Path(args.workdir))
    store = JobStore(config.jobs_dir)
    jobs = store.list_jobs(status=args.status or None)
    if not jobs:
        print("No jobs found.")
        return
    fmt = "{:<18} {:<10} {:<12} {:<6} {:<8} {}"
    print(fmt.format("JOB ID", "STATUS", "PHASE", "CYCLE", "SCORE", "OBJECTIVE"))
    print("-" * 80)
    for j in jobs:
        obj = j.spec.objective[:40]
        print(fmt.format(j.job_id, j.status, j.phase, j.cycle, f"{j.metrics.overall:.2f}", obj))


def cmd_jobs_status(args):
    from .jobs_store import JobStore

    config = EmyConfig(workdir=Path(args.workdir))
    store = JobStore(config.jobs_dir)
    try:
        state = store.load(args.job_id)
    except FileNotFoundError:
        print(f"Job {args.job_id!r} not found.", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(state.model_dump(), indent=2))


def cmd_jobs_tail(args):
    from .jobs_store import JobStore

    config = EmyConfig(workdir=Path(args.workdir))
    store = JobStore(config.jobs_dir)
    events = store.tail_events(args.job_id, n=args.n)
    for ev in events:
        ts = ev.get("ts", "")[:19]
        event = ev.get("event", "?")
        rest = {k: v for k, v in ev.items() if k not in ("ts", "event")}
        suffix = "  " + json.dumps(rest) if rest else ""
        print(f"{ts}  {event}{suffix}")


def cmd_jobs_cancel(args):
    from .jobs_store import JobStore
    from .utils import now_iso

    config = EmyConfig(workdir=Path(args.workdir))
    store = JobStore(config.jobs_dir)
    try:
        state = store.load(args.job_id)
    except FileNotFoundError:
        print(f"Job {args.job_id!r} not found.", file=sys.stderr)
        sys.exit(1)
    if state.status in ("completed", "failed", "canceled", "expired"):
        print(f"Job already in terminal state: {state.status}")
        return
    state.status = "canceled"
    state.finished_at = now_iso()
    store.save(state)
    store.append_event(args.job_id, {"event": "job_canceled", "source": "cli"})
    print(f"Job {args.job_id} canceled.")


def cmd_jobs_resume(args):
    from .jobs_store import JobStore

    config = EmyConfig(workdir=Path(args.workdir))
    store = JobStore(config.jobs_dir)
    try:
        state = store.load(args.job_id)
    except FileNotFoundError:
        print(f"Job {args.job_id!r} not found.", file=sys.stderr)
        sys.exit(1)
    if state.status not in ("paused", "blocked"):
        print(f"Job is {state.status!r}, cannot resume.")
        return
    state.status = "queued"
    state.blocked = None
    store.save(state)
    store.append_event(args.job_id, {"event": "job_requeued", "source": "cli"})
    print(f"Job {args.job_id} re-queued.")


def cmd_worker(args):
    """Run the background job worker (blocking, Ctrl-C to stop)."""
    from .jobs_store import JobStore
    from .jobs_worker import JobsWorker

    emy = _make_emy(args)
    store = JobStore(emy.config.jobs_dir)
    worker = JobsWorker(emy, store, poll_interval_s=emy.config.job_poll_interval_s)
    print(f"Emy worker started — watching {emy.config.jobs_dir}")
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        print("\nWorker stopped.")


# ── shared argparse helpers ───────────────────────────────────────────

def _add_research_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--objective", required=True, help="Research objective / question")
    p.add_argument("--time-budget", default="1h", help="Budget: 2h, 30m, 3600s (default: 1h)")
    p.add_argument("--deliverable", default="markdown_report",
                   choices=["markdown_report", "brief", "memo"])
    p.add_argument("--workdir", default="./runtime")
    p.add_argument("--model", default=None)
    p.add_argument("--planner-model", default=None)
    p.add_argument("--researcher-model", default=None)
    p.add_argument("--writer-model", default=None)
    p.add_argument("--critic-model", default=None)


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="emy",
        description="Emy v3 — Memory-first agentic RAG core",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── chat ──────────────────────────────────────────────────────
    p = sub.add_parser("chat", help="Interactive CLI chat")
    p.add_argument("--workdir", default="./runtime")
    p.add_argument("--mode", default="deploy", choices=["train", "deploy", "locked"])
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_chat)

    # ── serve ─────────────────────────────────────────────────────
    p = sub.add_parser("serve", help="Start Gradio UI + API server")
    p.add_argument("--workdir", default="./runtime")
    p.add_argument("--mode", default="deploy", choices=["train", "deploy", "locked"])
    p.add_argument("--model", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.set_defaults(func=cmd_serve)

    # ── ingest ────────────────────────────────────────────────────
    p = sub.add_parser("ingest", help="Ingest a document corpus")
    p.add_argument("corpus", help="Path to corpus directory")
    p.add_argument("--workdir", default="./runtime")
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_ingest)

    # ── status ────────────────────────────────────────────────────
    p = sub.add_parser("status", help="Show system status")
    p.add_argument("--workdir", default="./runtime")
    p.set_defaults(func=cmd_status)

    # ── research ──────────────────────────────────────────────────
    research = sub.add_parser("research", help="Autonomous research jobs")
    rsub = research.add_subparsers(dest="research_cmd", required=True)

    p = rsub.add_parser("submit", help="Submit a job (queued for background worker)")
    _add_research_args(p)
    p.set_defaults(func=cmd_research_submit)

    p = rsub.add_parser("run", help="Submit and run a job in the foreground (blocking)")
    _add_research_args(p)
    p.set_defaults(func=cmd_research_run)

    # ── jobs ──────────────────────────────────────────────────────
    jobs_cmd = sub.add_parser("jobs", help="Manage research jobs")
    jsub = jobs_cmd.add_subparsers(dest="jobs_cmd", required=True)

    p = jsub.add_parser("list", help="List all jobs")
    p.add_argument("--workdir", default="./runtime")
    p.add_argument("--status", default=None, help="Filter by status")
    p.set_defaults(func=cmd_jobs_list)

    p = jsub.add_parser("status", help="Show full job state as JSON")
    p.add_argument("job_id")
    p.add_argument("--workdir", default="./runtime")
    p.set_defaults(func=cmd_jobs_status)

    p = jsub.add_parser("tail", help="Tail the job event log")
    p.add_argument("job_id")
    p.add_argument("--workdir", default="./runtime")
    p.add_argument("-n", type=int, default=50, help="Number of events")
    p.set_defaults(func=cmd_jobs_tail)

    p = jsub.add_parser("cancel", help="Cancel a job")
    p.add_argument("job_id")
    p.add_argument("--workdir", default="./runtime")
    p.set_defaults(func=cmd_jobs_cancel)

    p = jsub.add_parser("resume", help="Resume a paused or blocked job")
    p.add_argument("job_id")
    p.add_argument("--workdir", default="./runtime")
    p.set_defaults(func=cmd_jobs_resume)

    # ── worker ────────────────────────────────────────────────────
    p = sub.add_parser("worker", help="Run background job worker (blocking)")
    p.add_argument("--workdir", default="./runtime")
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_worker)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
