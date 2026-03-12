from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from .api import create_api
from .config import EmyConfig
from .orchestrator import Emy


class EmyUI:
    def __init__(self, emy: Emy):
        self.emy = emy

    def chat(
        self,
        message: str,
        history: list[dict],
        session_id: str,
        mode: str,
    ):
        if not message.strip():
            return history or [], "", "", ""
        self.emy.set_mode(mode)
        resp = self.emy.respond(message, session_id=session_id)

        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": resp.answer})

        trace_text = "\n".join(f"- {t}" for t in resp.trace)

        score_text = ""
        if resp.score:
            score_text = (
                f"Faithfulness: {resp.score.faithfulness:.2f}\n"
                f"Relevance:    {resp.score.relevance:.2f}\n"
                f"Completeness: {resp.score.completeness:.2f}\n"
                f"Overall:      {resp.score.overall:.2f}\n"
                f"Lesson:       {resp.score.lesson}"
            )

        sources_text = ""
        if resp.sources:
            sources_text = "\n".join(
                f"- [{s.source}] score={s.score:.3f}: {s.text[:120]}"
                for s in resp.sources
            )

        return history, trace_text, score_text, sources_text

    def on_ingest(self, corpus_dir: str):
        try:
            count = self.emy.ingest(corpus_dir)
            return f"Ingested {count} chunks from {corpus_dir}"
        except Exception as exc:
            return f"Error: {exc}"

    def on_add_fact(self, category: str, content: str):
        if not category.strip() or not content.strip():
            return "Category and content are required."
        self.emy.add_fact(category.strip(), content.strip())
        return f"Fact saved in [{category}]."

    def on_add_reflection(self, lesson: str):
        if not lesson.strip():
            return "Lesson text is required."
        self.emy.add_reflection(lesson.strip())
        return "Reflection saved."

    def on_add_example(self, text: str, label: str):
        if not text.strip() or not label.strip():
            return "Text and label are required."
        self.emy.add_training_example(text.strip(), label.strip())
        return f"Training example saved with label [{label}]."

    def on_status(self):
        s = self.emy.status()
        lines = [
            f"Mode:            {s['mode']}",
            f"Ollama healthy:  {s['ollama_healthy']}",
            f"LLM model:       {s['llm_model']}",
            f"Embed model:     {s['embedding_model']}",
            f"Memory entries:  {s['memory_entries']}",
            f"Doc entries:     {s['doc_entries']}",
        ]
        vf = s.get("vault_files", {})
        for k, v in vf.items():
            lines.append(f"Vault {k:12s}: {v}")
        return "\n".join(lines)


def build_ui(emy: Emy) -> gr.Blocks:
    app = EmyUI(emy)

    with gr.Blocks(title="Emy v2", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Emy v2 — Memory-First Agentic RAG")

        with gr.Row():
            session_id = gr.Textbox(label="Session ID", value="default", scale=1)
            mode = gr.Radio(
                ["train", "deploy", "locked"],
                value=emy.mode,
                label="Mode",
                scale=2,
            )
            status_btn = gr.Button("Status", scale=1)

        status_box = gr.Textbox(label="System status", lines=6, interactive=False)
        status_btn.click(app.on_status, outputs=[status_box])

        # ── Chat ──────────────────────────────────────────────────
        chatbot = gr.Chatbot(type="messages", label="Chat", height=400)
        msg = gr.Textbox(label="Message", placeholder="Ask Emy anything...")
        send = gr.Button("Send", variant="primary")

        with gr.Row():
            trace_box = gr.Textbox(label="Trace", lines=6, interactive=False)
            score_box = gr.Textbox(label="Score", lines=6, interactive=False)
        sources_box = gr.Textbox(label="Retrieved sources", lines=4, interactive=False)

        send.click(
            app.chat,
            inputs=[msg, chatbot, session_id, mode],
            outputs=[chatbot, trace_box, score_box, sources_box],
        )
        msg.submit(
            app.chat,
            inputs=[msg, chatbot, session_id, mode],
            outputs=[chatbot, trace_box, score_box, sources_box],
        )

        # ── Training ─────────────────────────────────────────────
        with gr.Accordion("Training & Memory", open=False):
            with gr.Tab("Ingest Documents"):
                corpus_dir = gr.Textbox(
                    label="Corpus directory", value="./data/sample_docs"
                )
                ingest_btn = gr.Button("Ingest")
                ingest_status = gr.Textbox(label="Status", interactive=False)
                ingest_btn.click(app.on_ingest, inputs=[corpus_dir], outputs=[ingest_status])

            with gr.Tab("Add Fact"):
                fact_cat = gr.Textbox(label="Category", value="preferences")
                fact_content = gr.Textbox(label="Content")
                fact_btn = gr.Button("Save Fact")
                fact_status = gr.Textbox(label="Status", interactive=False)
                fact_btn.click(
                    app.on_add_fact,
                    inputs=[fact_cat, fact_content],
                    outputs=[fact_status],
                )

            with gr.Tab("Add Reflection"):
                refl_text = gr.Textbox(
                    label="Lesson",
                    placeholder="e.g. Skip retrieval for greeting-only messages",
                )
                refl_btn = gr.Button("Save Reflection")
                refl_status = gr.Textbox(label="Status", interactive=False)
                refl_btn.click(app.on_add_reflection, inputs=[refl_text], outputs=[refl_status])

            with gr.Tab("Add Training Example"):
                ex_text = gr.Textbox(label="Text", placeholder="e.g. compare these two methods")
                ex_label = gr.Textbox(label="Intent label", placeholder="e.g. research")
                ex_btn = gr.Button("Save Example")
                ex_status = gr.Textbox(label="Status", interactive=False)
                ex_btn.click(
                    app.on_add_example,
                    inputs=[ex_text, ex_label],
                    outputs=[ex_status],
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Emy v2 server (UI + API)")
    parser.add_argument("--workdir", default="./runtime")
    parser.add_argument("--model", default=None, help="Ollama LLM model")
    parser.add_argument("--embed-model", default=None, help="Ollama embedding model")
    parser.add_argument("--mode", default="deploy", choices=["train", "deploy", "locked"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    config = EmyConfig(workdir=Path(args.workdir), mode=args.mode)
    if args.model:
        config.llm_model = args.model
    if args.embed_model:
        config.embedding_model = args.embed_model

    emy = Emy(config)

    # Mount both Gradio UI and FastAPI training endpoints
    api = create_api(emy)
    demo = build_ui(emy)
    app = gr.mount_gradio_app(api, demo, path="/")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
