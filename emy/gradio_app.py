from __future__ import annotations

import io
import json
import secrets
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

try:
    from .config import EmyConfig
    from .orchestrator import Emy
except ImportError:  # pragma: no cover
    from emy.config import EmyConfig
    from emy.orchestrator import Emy

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_WORKDIR = "./runtime"
MAX_RECENT_ATTACHMENTS = 8
SUPPORTED_EXTENSIONS = {
    ".txt": "Plain text", ".md": "Markdown", ".py": "Python", ".json": "JSON",
    ".yaml": "YAML", ".yml": "YAML", ".csv": "CSV", ".rst": "RST", ".log": "Log",
    ".pdf": "PDF", ".docx": "Word", ".pptx": "PowerPoint", ".zip": "ZIP",
}
UPLOAD_TYPES = list(SUPPORTED_EXTENSIONS)
EMYS: dict[tuple[str, str], Emy] = {}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def truncate(text: str, limit: int = 34) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def make_session_id(prefix: str = "gradio") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def derive_session_title(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            return truncate(str(msg["content"]))
    return "New chat"


def app_mode(learning_enabled: bool) -> str:
    return "train" if learning_enabled else "locked"


def sessions_path(workdir: str | Path) -> Path:
    return Path(workdir) / "logs" / "gradio_sessions.json"


def file_size_label(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def file_type_label(name: str) -> str:
    return SUPPORTED_EXTENSIONS.get(Path(name).suffix.lower(), "File")


def attachment_md(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    lines = ["**Attachments**"]
    for item in items:
        lines.append(f"- `{item['name']}` · {item['type']} · {item['size_label']}")
    return "\n".join(lines)


def details_md(resp: Any) -> str:
    parts: list[str] = []
    if getattr(resp, "tools_used", None):
        parts.append("**Tools**: " + ", ".join(resp.tools_used))
    if getattr(resp, "score", None):
        s = resp.score
        parts.append(
            f"**Score**: overall={s.overall:.2f}, faithfulness={s.faithfulness:.2f}, "
            f"relevance={s.relevance:.2f}, completeness={s.completeness:.2f}"
        )
        if getattr(s, "lesson", None):
            parts.append(f"**Lesson**: {s.lesson}")
    if getattr(resp, "sources", None):
        src_lines = ["**Sources**"]
        for src in resp.sources[:6]:
            txt = (src.text or "").strip().replace("\n", " ")[:180]
            src_lines.append(f"- `{src.source}` ({src.score:.3f}) — {txt}")
        parts.append("\n".join(src_lines))
    if getattr(resp, "trace", None):
        parts.append("**Trace**\n```text\n" + "\n".join(resp.trace[:18]) + "\n```")
    return "\n\n".join(parts) if parts else "No extra details."


def build_attachment_hint(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    names = ", ".join(i["name"] for i in items)
    return (
        "\n\nAttached files for this message: " + names +
        "\nIf the user refers to the attachment or uploaded file, use these first."
    )


def build_config(base_url: str, workdir: str, learning_enabled: bool, vision_enabled: bool) -> EmyConfig:
    return EmyConfig(
        ollama_base_url=base_url.strip() or DEFAULT_OLLAMA_BASE_URL,
        workdir=Path(workdir.strip() or DEFAULT_WORKDIR),
        mode=app_mode(learning_enabled),
        vision_fallback_enabled=vision_enabled,
    )


def get_emy(base_url: str, workdir: str, learning_enabled: bool, vision_enabled: bool) -> Emy:
    key = ((base_url or DEFAULT_OLLAMA_BASE_URL).strip(), (workdir or DEFAULT_WORKDIR).strip())
    if key not in EMYS:
        EMYS[key] = Emy(build_config(*key, learning_enabled, vision_enabled))
    emy = EMYS[key]
    emy.set_mode(app_mode(learning_enabled))
    emy.config.vision_fallback_enabled = vision_enabled
    return emy


def load_sessions(workdir: str) -> dict[str, dict[str, Any]]:
    path = sessions_path(workdir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    sessions = payload.get("sessions", {})
    cleaned: dict[str, dict[str, Any]] = {}
    for sid, sess in sessions.items():
        if not isinstance(sess, dict):
            continue
        msgs = sess.get("messages") if isinstance(sess.get("messages"), list) else []
        atts = sess.get("recent_attachments") if isinstance(sess.get("recent_attachments"), list) else []
        cleaned[sid] = {
            "title": sess.get("title") or derive_session_title(msgs),
            "messages": [m for m in msgs if isinstance(m, dict)],
            "recent_attachments": [a for a in atts if isinstance(a, dict)][-MAX_RECENT_ATTACHMENTS:],
            "created_at": sess.get("created_at") or utc_now_iso(),
            "updated_at": sess.get("updated_at") or utc_now_iso(),
        }
    return cleaned


def save_sessions(workdir: str, sessions: dict[str, dict[str, Any]]) -> None:
    path = sessions_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"saved_at": utc_now_iso(), "sessions": sessions}, indent=2), encoding="utf-8")


def fresh_state(workdir: str) -> dict[str, Any]:
    sessions = load_sessions(workdir)
    if not sessions:
        sid = make_session_id()
        now = utc_now_iso()
        sessions = {
            sid: {
                "title": "New chat", "messages": [], "recent_attachments": [],
                "created_at": now, "updated_at": now,
            }
        }
        save_sessions(workdir, sessions)
    active = sorted(sessions.items(), key=lambda x: x[1].get("updated_at", ""), reverse=True)[0][0]
    return {"sessions": sessions, "active_session_id": active, "pending_files": []}


def active_session(state: dict[str, Any]) -> dict[str, Any]:
    return state["sessions"][state["active_session_id"]]


def session_choices(state: dict[str, Any]) -> list[tuple[str, str]]:
    rows = []
    ordered = sorted(state["sessions"].items(), key=lambda x: x[1].get("updated_at", ""), reverse=True)
    for sid, sess in ordered:
        label = f"{sess.get('title') or 'New chat'} · {len(sess.get('messages', []))} msgs"
        if sess.get("recent_attachments"):
            label += f" · {len(sess['recent_attachments'])} files"
        rows.append((label, sid))
    return rows


def render_chat(state: dict[str, Any]) -> list[dict[str, Any]]:
    return active_session(state).get("messages", [])


def render_recent(state: dict[str, Any]) -> str:
    items = active_session(state).get("recent_attachments", [])
    return attachment_md(list(reversed(items))) if items else "*No recent attachments.*"


def render_pending(state: dict[str, Any]) -> str:
    items = state.get("pending_files", [])
    return attachment_md(items) if items else "*No pending files.*"


def render_status(state: dict[str, Any], workdir: str, learning_enabled: bool, note: str = "") -> str:
    mode = app_mode(learning_enabled)
    base = f"**Mode:** `{mode}`  \n**Workdir:** `{workdir}`  \n**Chats:** {len(state['sessions'])}"
    return base + (f"\n\n{note}" if note else "")


def sync_session_to_emy(emy: Emy, state: dict[str, Any]) -> None:
    sid = state["active_session_id"]
    emy.sessions[sid] = []
    session = state["sessions"][sid]
    for msg in session.get("messages", []):
        if msg.get("role") not in {"user", "assistant"}:
            continue
        row = {"role": msg["role"], "content": msg.get("content", "")}
        if msg.get("metadata"):
            row["metadata"] = msg["metadata"]
        emy.sessions[sid].append(row)
    if session.get("recent_attachments"):
        emy.session_attachments[sid] = session["recent_attachments"][-MAX_RECENT_ATTACHMENTS:]


def stage_files(filepaths: list[str] | None, state: dict[str, Any]) -> tuple[dict[str, Any], str]:
    for fp in filepaths or []:
        p = Path(fp)
        if not p.exists():
            continue
        data = p.read_bytes()
        state["pending_files"].append(
            {
                "name": p.name,
                "data": data,
                "type": file_type_label(p.name),
                "size_bytes": len(data),
                "size_label": file_size_label(len(data)),
                "uploaded_at": utc_now_iso(),
            }
        )
    state["pending_files"] = state["pending_files"][-MAX_RECENT_ATTACHMENTS:]
    return state, render_pending(state)


def clear_pending(state: dict[str, Any]) -> tuple[dict[str, Any], str]:
    state["pending_files"] = []
    return state, render_pending(state)


def ingest_blob_items(emy: Emy, items: list[dict[str, Any]]) -> int:
    if not items:
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for item in items:
            name, data = item["name"], item["data"]
            if name.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                        zf.extractall(root)
                except zipfile.BadZipFile:
                    continue
            else:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        return emy.ingest(str(root))


def ingest_files(paths: list[str] | None, folder: str, base_url: str, workdir: str, learning_enabled: bool, vision_enabled: bool, state: dict[str, Any]):
    emy = get_emy(base_url, workdir, learning_enabled, vision_enabled)
    count = 0
    blobs = []
    for fp in paths or []:
        p = Path(fp)
        if p.exists():
            data = p.read_bytes()
            blobs.append({"name": p.name, "data": data})
    if blobs:
        count += ingest_blob_items(emy, [{**b, "type": file_type_label(b['name']), "size_label": ""} for b in blobs])
    if folder and Path(folder).exists():
        count += emy.ingest(folder)
    return render_status(state, workdir, learning_enabled, f"✅ Ingested {count} chunks.")


def remember_recent(session: dict[str, Any], items: list[dict[str, Any]]) -> None:
    session.setdefault("recent_attachments", []).extend(items)
    session["recent_attachments"] = session["recent_attachments"][-MAX_RECENT_ATTACHMENTS:]


def select_session(session_id: str, state: dict[str, Any], workdir: str, learning_enabled: bool):
    if session_id in state["sessions"]:
        state["active_session_id"] = session_id
    return state, render_chat(state), render_recent(state), render_pending(state), render_status(state, workdir, learning_enabled)


def new_chat(state: dict[str, Any], workdir: str, learning_enabled: bool):
    sid = make_session_id()
    now = utc_now_iso()
    state["sessions"][sid] = {
        "title": "New chat", "messages": [], "recent_attachments": [],
        "created_at": now, "updated_at": now,
    }
    state["active_session_id"] = sid
    save_sessions(workdir, state["sessions"])
    return (
        state,
        gr.update(choices=session_choices(state), value=sid),
        render_chat(state),
        render_recent(state),
        render_pending(state),
        render_status(state, workdir, learning_enabled, "✅ New chat created."),
    )


def delete_chat(state: dict[str, Any], workdir: str, learning_enabled: bool):
    if state["active_session_id"] in state["sessions"]:
        state["sessions"].pop(state["active_session_id"], None)
    if not state["sessions"]:
        state = fresh_state(workdir)
    else:
        state["active_session_id"] = next(iter(session_choices(state)))[1]
    save_sessions(workdir, state["sessions"])
    return (
        state,
        gr.update(choices=session_choices(state), value=state["active_session_id"]),
        render_chat(state),
        render_recent(state),
        render_pending(state),
        render_status(state, workdir, learning_enabled, "🗑️ Chat deleted."),
    )


def apply_runtime(base_url: str, workdir: str, learning_enabled: bool, vision_enabled: bool):
    state = fresh_state(workdir)
    get_emy(base_url, workdir, learning_enabled, vision_enabled)
    return (
        state,
        gr.update(choices=session_choices(state), value=state["active_session_id"]),
        render_chat(state),
        render_recent(state),
        render_pending(state),
        render_status(state, workdir, learning_enabled, "🔌 Runtime applied."),
    )


def stream_chunks(text: str, batch_words: int = 8):
    words = (text or "").split()
    if not words:
        yield ""
        return
    built: list[str] = []
    for idx, word in enumerate(words, 1):
        built.append(word)
        if idx % batch_words == 0 or idx == len(words):
            yield " ".join(built)


def send_message(message: str, state: dict[str, Any], base_url: str, workdir: str, learning_enabled: bool, vision_enabled: bool, stream_enabled: bool):
    text = (message or "").strip()
    if not text and not state.get("pending_files"):
        yield (
            state,
            gr.update(choices=session_choices(state), value=state["active_session_id"]),
            render_chat(state),
            render_recent(state),
            render_pending(state),
            render_status(state, workdir, learning_enabled),
            "",
        )
        return

    emy = get_emy(base_url, workdir, learning_enabled, vision_enabled)
    session = active_session(state)
    session.setdefault("messages", [])
    turn_attachments = list(state.get("pending_files", []))
    state["pending_files"] = []

    user_content = text or "Attached files"
    if turn_attachments:
        remember_recent(session, turn_attachments)
        user_content += "\n\n" + attachment_md(turn_attachments)

    session["messages"].append({"role": "user", "content": user_content})
    session["messages"].append({"role": "assistant", "content": "…"})
    session["title"] = derive_session_title(session["messages"])
    session["updated_at"] = utc_now_iso()
    save_sessions(workdir, state["sessions"])

    yield (
        state,
        gr.update(choices=session_choices(state), value=state["active_session_id"]),
        render_chat(state),
        render_recent(state),
        render_pending(state),
        render_status(state, workdir, learning_enabled, "⏳ Thinking..."),
        "",
    )

    sync_session_to_emy(emy, state)
    ingested_count = ingest_blob_items(emy, turn_attachments) if turn_attachments else 0
    query = text + build_attachment_hint(turn_attachments)
    resp = emy.respond(
        query or "Analyze attached files.",
        session_id=state["active_session_id"],
        mode=app_mode(learning_enabled),
        attachments=turn_attachments,
    )

    session["messages"][-1]["content"] = ""
    if stream_enabled:
        for partial in stream_chunks(resp.answer):
            session["messages"][-1]["content"] = partial
            yield (
                state,
                gr.update(choices=session_choices(state), value=state["active_session_id"]),
                render_chat(state),
                render_recent(state),
                render_pending(state),
                render_status(state, workdir, learning_enabled, "✨ Streaming reply..."),
                "",
            )
            time.sleep(0.02)
    else:
        session["messages"][-1]["content"] = resp.answer
        yield (
            state,
            gr.update(choices=session_choices(state), value=state["active_session_id"]),
            render_chat(state),
            render_recent(state),
            render_pending(state),
            render_status(state, workdir, learning_enabled, "✅ Reply generated."),
            "",
        )

    if resp.trace or resp.tools_used or resp.sources or resp.score:
        session["messages"].append(
            {
                "role": "assistant",
                "content": details_md(resp),
                "metadata": {"title": "Details"},
            }
        )
    session["updated_at"] = utc_now_iso()
    save_sessions(workdir, state["sessions"])
    note = f"📎 Ingested {ingested_count} chunks from attachments." if ingested_count else "✅ Reply ready."
    if stream_enabled:
        note += " UI streaming is on."
    yield (
        state,
        gr.update(choices=session_choices(state), value=state["active_session_id"]),
        render_chat(state),
        render_recent(state),
        render_pending(state),
        render_status(state, workdir, learning_enabled, note),
        "",
    )

def add_fact(category: str, content: str, base_url: str, workdir: str, learning_enabled: bool, vision_enabled: bool, state: dict[str, Any]):
    if not category.strip() or not content.strip():
        return render_status(state, workdir, learning_enabled, "⚠️ Category and content are required.")
    get_emy(base_url, workdir, learning_enabled, vision_enabled).add_fact(category.strip(), content.strip())
    return render_status(state, workdir, learning_enabled, f"✅ Saved fact in `{category.strip()}`.")


def add_reflection(lesson: str, base_url: str, workdir: str, learning_enabled: bool, vision_enabled: bool, state: dict[str, Any]):
    if not lesson.strip():
        return render_status(state, workdir, learning_enabled, "⚠️ Lesson is required.")
    get_emy(base_url, workdir, learning_enabled, vision_enabled).add_reflection(lesson.strip())
    return render_status(state, workdir, learning_enabled, "✅ Reflection saved.")


def add_example(text: str, label: str, base_url: str, workdir: str, learning_enabled: bool, vision_enabled: bool, state: dict[str, Any]):
    if not text.strip() or not label.strip():
        return render_status(state, workdir, learning_enabled, "⚠️ Text and label are required.")
    get_emy(base_url, workdir, learning_enabled, vision_enabled).add_training_example(text.strip(), label.strip())
    return render_status(state, workdir, learning_enabled, f"✅ Saved example as `{label.strip()}`.")


def add_entity(name: str, kind: str, desc: str, base_url: str, workdir: str, learning_enabled: bool, vision_enabled: bool, state: dict[str, Any]):
    if not all(x.strip() for x in [name, kind, desc]):
        return render_status(state, workdir, learning_enabled, "⚠️ Name, type, and description are required.")
    get_emy(base_url, workdir, learning_enabled, vision_enabled).add_entity(name.strip(), kind.strip(), desc.strip())
    return render_status(state, workdir, learning_enabled, f"✅ Saved entity `{name.strip()}`.")


def system_status(base_url: str, workdir: str, learning_enabled: bool, vision_enabled: bool):
    emy = get_emy(base_url, workdir, learning_enabled, vision_enabled)
    return json.dumps(emy.status(), indent=2, ensure_ascii=False)


CSS = """
.gradio-container {max-width: 1500px !important; background: linear-gradient(180deg,#0b0f17,#101522) !important;}
.block, .panel {border: 1px solid rgba(212,180,93,.14); border-radius: 16px; background: rgba(20,24,36,.88);}
#hero {padding: 18px 20px; border: 1px solid rgba(212,180,93,.18); border-radius: 18px; margin-bottom: 12px;
background: radial-gradient(circle at top right, rgba(212,180,93,.08), transparent 30%), linear-gradient(180deg,#121826,#0f1421);}
#hero h1 {margin: 0; color: #e5c56b;} #hero p {margin: 6px 0 0; color: #b8c0cf;}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        app_state = gr.State(fresh_state(DEFAULT_WORKDIR))

        gr.Markdown(
            "<div id='hero'><h1>Emy</h1><p>Gradio UI for the current orchestrator. Cleaner workflow, staged attachments, session history, and runtime tools.</p></div>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes="panel"):
                    gr.Markdown("### Runtime")
                    base_url = gr.Textbox(label="Ollama base URL", value=DEFAULT_OLLAMA_BASE_URL)
                    workdir = gr.Textbox(label="Runtime directory", value=DEFAULT_WORKDIR)
                    learning = gr.Checkbox(label="Realtime learning", value=True)
                    vision = gr.Checkbox(label="Vision fallback", value=False)
                    stream = gr.Checkbox(label="Stream replies", value=True)
                    apply_btn = gr.Button("Apply runtime", variant="primary")
                    status_md = gr.Markdown()

                with gr.Group(elem_classes="panel"):
                    gr.Markdown("### Chats")
                    session_picker = gr.Radio(label="History", choices=[], value=None)
                    with gr.Row():
                        new_btn = gr.Button("New chat")
                        del_btn = gr.Button("Delete chat")
                    recent_md = gr.Markdown("*No recent attachments.*")

                with gr.Accordion("Knowledge Studio", open=False):
                    ingest_files_box = gr.File(label="Upload files to ingest", file_count="multiple", file_types=UPLOAD_TYPES, type="filepath")
                    ingest_folder = gr.Textbox(label="Or ingest a local folder", value="./data/sample_docs")
                    ingest_btn = gr.Button("Ingest")
                    fact_cat = gr.Textbox(label="Fact category", value="preferences")
                    fact_content = gr.Textbox(label="Fact content", lines=2)
                    fact_btn = gr.Button("Save fact")
                    refl_text = gr.Textbox(label="Reflection", lines=2)
                    refl_btn = gr.Button("Save reflection")
                    ex_text = gr.Textbox(label="Training text")
                    ex_label = gr.Textbox(label="Intent label")
                    ex_btn = gr.Button("Save training example")
                    ent_name = gr.Textbox(label="Entity name")
                    ent_type = gr.Textbox(label="Entity type")
                    ent_desc = gr.Textbox(label="Entity description", lines=2)
                    ent_btn = gr.Button("Save entity")
                    status_btn = gr.Button("System status")
                    status_json = gr.Code(label="Status", language="json")

            with gr.Column(scale=3):
                with gr.Group(elem_classes="panel"):
                    chat = gr.Chatbot(label="Conversation", height=620)
                    pending_md = gr.Markdown("*No pending files.*")
                    staged_files = gr.File(label="Attach files for next message", file_count="multiple", file_types=UPLOAD_TYPES, type="filepath")
                    with gr.Row():
                        msg = gr.Textbox(label="Message", placeholder="Message Emy…", scale=8)
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                        clear_btn = gr.Button("Clear files", scale=1)

        demo.load(
            apply_runtime,
            inputs=[base_url, workdir, learning, vision],
            outputs=[app_state, session_picker, chat, recent_md, pending_md, status_md],
        )
        apply_btn.click(apply_runtime, [base_url, workdir, learning, vision], [app_state, session_picker, chat, recent_md, pending_md, status_md])
        session_picker.change(select_session, [session_picker, app_state, workdir, learning], [app_state, chat, recent_md, pending_md, status_md])
        new_btn.click(new_chat, [app_state, workdir, learning], [app_state, session_picker, chat, recent_md, pending_md, status_md])
        del_btn.click(delete_chat, [app_state, workdir, learning], [app_state, session_picker, chat, recent_md, pending_md, status_md])
        staged_files.upload(stage_files, [staged_files, app_state], [app_state, pending_md])
        clear_btn.click(clear_pending, [app_state], [app_state, pending_md])
        send_btn.click(send_message, [msg, app_state, base_url, workdir, learning, vision, stream], [app_state, session_picker, chat, recent_md, pending_md, status_md, msg])
        msg.submit(send_message, [msg, app_state, base_url, workdir, learning, vision, stream], [app_state, session_picker, chat, recent_md, pending_md, status_md, msg])
        ingest_btn.click(ingest_files, [ingest_files_box, ingest_folder, base_url, workdir, learning, vision, app_state], [status_md])
        fact_btn.click(add_fact, [fact_cat, fact_content, base_url, workdir, learning, vision, app_state], [status_md])
        refl_btn.click(add_reflection, [refl_text, base_url, workdir, learning, vision, app_state], [status_md])
        ex_btn.click(add_example, [ex_text, ex_label, base_url, workdir, learning, vision, app_state], [status_md])
        ent_btn.click(add_entity, [ent_name, ent_type, ent_desc, base_url, workdir, learning, vision, app_state], [status_md])
        status_btn.click(system_status, [base_url, workdir, learning, vision], [status_json])
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
