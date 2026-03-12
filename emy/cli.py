from __future__ import annotations

import argparse
import json
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

    print(f"Emy v2 | mode={args.mode} | model={config.llm_model}")
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
    from .ui import main as ui_main

    # Forward args to ui main via sys.argv manipulation
    import sys

    sys.argv = [
        "emy",
        "--workdir", args.workdir,
        "--mode", args.mode,
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.model:
        sys.argv.extend(["--model", args.model])
    ui_main()


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


def main():
    parser = argparse.ArgumentParser(
        prog="emy",
        description="Emy v2 — Memory-first agentic RAG core",
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
