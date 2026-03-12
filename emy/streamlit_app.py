"""Emy v2 — Modern Streamlit Chat UI.

Launch with:
    streamlit run emy/streamlit_app.py
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

try:
    from .config import EmyConfig
    from .orchestrator import Emy
except ImportError:
    from emy.config import EmyConfig
    from emy.orchestrator import Emy

# ── Supported file extensions ─────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".txt": "Plain text",
    ".md": "Markdown",
    ".py": "Python",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".csv": "CSV",
    ".rst": "reStructuredText",
    ".log": "Log file",
    ".pdf": "PDF",
    ".docx": "Word document",
    ".pptx": "PowerPoint",
    ".zip": "ZIP archive (auto-extracted)",
}

UPLOAD_EXTENSIONS = [e.lstrip(".") for e in SUPPORTED_EXTENSIONS if e != ".zip"]
UPLOAD_EXTENSIONS.append("zip")


# ── Helpers ───────────────────────────────────────────────────────────

def get_emy() -> Emy:
    """Return a cached Emy instance across reruns."""
    if "emy" not in st.session_state:
        config = EmyConfig(workdir=Path("./runtime"), mode="deploy")
        st.session_state.emy = Emy(config)
    return st.session_state.emy


def ingest_uploaded_files(emy: Emy, uploaded_files: list) -> int:
    """Write uploaded files to a temp dir (extracting zips) and ingest."""
    if not uploaded_files:
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for uf in uploaded_files:
            data = uf.getvalue()
            if uf.name.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                        zf.extractall(tmp_path)
                except zipfile.BadZipFile:
                    st.warning(f"Skipped bad zip: {uf.name}")
            else:
                (tmp_path / uf.name).write_bytes(data)
        return emy.ingest(tmp)


def export_memory(emy: Emy) -> bytes:
    """Package vault memory into a zip."""
    buf = io.BytesIO()
    vault_dir = emy.config.vault_dir
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if vault_dir.exists():
            for fp in sorted(vault_dir.rglob("*")):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(vault_dir))
    buf.seek(0)
    return buf.getvalue()


def export_all_data(emy: Emy) -> bytes:
    """Package full runtime data into a zip."""
    buf = io.BytesIO()
    workdir = emy.config.workdir
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if workdir.exists():
            for fp in sorted(workdir.rglob("*")):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(workdir))
    buf.seek(0)
    return buf.getvalue()


def import_memory(emy: Emy, data: bytes) -> bool:
    """Import a memory vault zip, overwriting existing vault files."""
    if not zipfile.is_zipfile(io.BytesIO(data)):
        return False
    vault_dir = emy.config.vault_dir
    vault_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        zf.extractall(vault_dir)
    return True


def render_meta_hamburger(meta: dict, idx: int = 0) -> None:
    """Render trace/tools/scores/sources in a compact expandable section."""
    has_content = (
        meta.get("trace")
        or meta.get("tools_used")
        or meta.get("score")
        or meta.get("sources")
    )
    if not has_content:
        return

    # Build summary chips
    parts = []
    if meta.get("tools_used"):
        parts.append(f"{len(meta['tools_used'])} tools")
    if meta.get("score"):
        parts.append(f"score {meta['score']['overall']:.2f}")
    if meta.get("sources"):
        parts.append(f"{len(meta['sources'])} sources")

    summary = " | ".join(parts) if parts else "Details"

    with st.expander(f"Details  —  {summary}", expanded=False):
        # Scores
        if meta.get("score"):
            sc = meta["score"]
            st.markdown(
                f'<span class="score-pill">Faithfulness {sc["faithfulness"]:.2f}</span>'
                f'<span class="score-pill">Relevance {sc["relevance"]:.2f}</span>'
                f'<span class="score-pill">Completeness {sc["completeness"]:.2f}</span>'
                f'<span class="score-pill">Overall {sc["overall"]:.2f}</span>',
                unsafe_allow_html=True,
            )
            if sc.get("lesson"):
                st.info(sc["lesson"])

        # Tools
        if meta.get("tools_used"):
            st.markdown(f"**Tools:** {', '.join(meta['tools_used'])}")

        # Sources
        if meta.get("sources"):
            st.markdown("**Retrieved Sources**")
            for src in meta["sources"]:
                st.markdown(
                    f'<div class="source-card">'
                    f'<code>{src["source"]}</code> '
                    f'<span class="score-pill">{src["score"]:.3f}</span><br>'
                    f'{src["text"][:150]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Trace (nested expander)
        if meta.get("trace"):
            with st.expander("Trace log", expanded=False):
                st.code("\n".join(meta["trace"]), language=None)


# ── Page config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Emy",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS — Modern chat UI ─────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Main container ── */
.block-container {
    max-width: 820px;
    padding-top: 1.5rem;
    padding-bottom: 0;
}

/* ── Header ── */
.emy-header {
    text-align: center;
    padding: 0.5rem 0 0.8rem;
    border-bottom: 1px solid rgba(128,128,128,0.15);
    margin-bottom: 0.5rem;
}
.emy-header h1 {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.emy-header p {
    font-size: 0.78rem;
    opacity: 0.5;
    margin: 0.15rem 0 0;
}

/* ── Chat bubbles ── */
.stChatMessage { border-radius: 16px !important; }
div[data-testid="stChatMessage"] {
    border-radius: 16px;
    margin-bottom: 0.5rem;
}

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 50%, #16213e 100%);
}
div[data-testid="stSidebar"] .stMarkdown,
div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] .stRadio label {
    color: #e0e0e0 !important;
}

/* ── Metrics row ── */
.score-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
    background: rgba(102, 126, 234, 0.1);
    color: #667eea;
}
.source-card {
    padding: 8px 12px;
    border-radius: 10px;
    background: rgba(128, 128, 128, 0.06);
    border: 1px solid rgba(128, 128, 128, 0.1);
    margin-bottom: 6px;
    font-size: 0.8rem;
}
.source-card code {
    font-size: 0.72rem;
    background: rgba(102, 126, 234, 0.1);
    padding: 1px 6px;
    border-radius: 4px;
}

/* ── File drop zone ── */
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(102, 126, 234, 0.3) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(102, 126, 234, 0.6) !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 10px;
    font-weight: 600;
}

/* ── Tab styling ── */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.85rem;
}

/* ── Mode badge ── */
.mode-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.mode-deploy { background: rgba(46,204,113,0.15); color: #2ecc71; }
.mode-train { background: rgba(241,196,15,0.15); color: #f1c40f; }
</style>
""", unsafe_allow_html=True)

# ── Init ──────────────────────────────────────────────────────────────

emy = get_emy()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar — Settings & Memory ──────────────────────────────────────

with st.sidebar:
    st.markdown("### Settings")

    mode_choice = st.radio(
        "Mode",
        options=["Deploy", "Train", "Both"],
        index=0,
        help="**Deploy** — Chat only.\n**Train** — Add knowledge.\n**Both** — Full access.",
        horizontal=True,
    )

    emy_mode = "deploy" if mode_choice == "Deploy" else "train"
    emy.set_mode(emy_mode)

    st.divider()

    # ── Memory Management ──
    st.markdown("### Memory")

    mem_action = st.radio(
        "Action",
        ["View", "Download", "Import", "Export All"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mem_action == "View":
        vault_file = st.selectbox(
            "Vault file",
            ["facts", "intents", "reflections", "entities"],
        )
        entries = emy.vault.read_entries(f"{vault_file}.md")
        total = sum(len(v) for v in entries.values())
        st.caption(f"{total} entries")
        for cat, cat_entries in entries.items():
            with st.expander(f"{cat} ({len(cat_entries)})"):
                for e in cat_entries:
                    fields_str = " | ".join(f"**{k}**: {v[:80]}" for k, v in e.fields.items())
                    st.markdown(f"- `{e.id}` — {fields_str}")

    elif mem_action == "Download":
        st.caption("Download memory vault as a zip archive.")
        if st.button("Download Memory", use_container_width=True, type="primary"):
            with st.spinner("Packaging..."):
                mem_zip = export_memory(emy)
            st.download_button(
                label="Save emy_memory.zip",
                data=mem_zip,
                file_name="emy_memory.zip",
                mime="application/zip",
                use_container_width=True,
            )

    elif mem_action == "Import":
        st.caption("Upload a memory vault zip to restore or merge memory.")
        mem_file = st.file_uploader("Memory zip", type=["zip"], key="mem_import")
        if mem_file and st.button("Import Memory", type="primary", use_container_width=True):
            success = import_memory(emy, mem_file.getvalue())
            if success:
                st.success("Memory imported successfully.")
            else:
                st.error("Invalid zip file.")

    elif mem_action == "Export All":
        st.caption("Download full runtime (vault + indexes + logs).")
        if st.button("Export Everything", use_container_width=True, type="primary"):
            with st.spinner("Packaging..."):
                all_zip = export_all_data(emy)
            st.download_button(
                label="Save emy_data.zip",
                data=all_zip,
                file_name="emy_data.zip",
                mime="application/zip",
                use_container_width=True,
            )

    st.divider()

    # ── Status ──
    if st.button("System Status", use_container_width=True):
        s = emy.status()
        st.json(s)


# ── Header ────────────────────────────────────────────────────────────

mode_class = "mode-deploy" if mode_choice == "Deploy" else "mode-train"
st.markdown(f"""
<div class="emy-header">
    <h1>Emy</h1>
    <p>Memory-First Agentic RAG &nbsp;
    <span class="mode-badge {mode_class}">{mode_choice}</span></p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────

if mode_choice == "Deploy":
    tab_names = ["Chat"]
elif mode_choice == "Train":
    tab_names = ["Training"]
else:
    tab_names = ["Chat", "Training"]

tabs = st.tabs(tab_names)

# ── Chat Tab ──────────────────────────────────────────────────────────

if "Chat" in tab_names:
    chat_tab = tabs[tab_names.index("Chat")]
    with chat_tab:
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else None):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("meta"):
                    render_meta_hamburger(msg["meta"], msg.get("_idx", 0))

        # File upload area (drag & drop)
        with st.expander("Attach files", expanded=False):
            chat_files = st.file_uploader(
                "Drag & drop files here (including .zip)",
                type=UPLOAD_EXTENSIONS,
                accept_multiple_files=True,
                key="chat_upload",
                help="Files will be ingested into Emy's document index.",
                label_visibility="collapsed",
            )
            if chat_files:
                st.caption(f"{len(chat_files)} file(s) ready")
                if st.button("Ingest attached files", type="primary"):
                    with st.spinner("Ingesting..."):
                        count = ingest_uploaded_files(emy, chat_files)
                    st.success(f"Ingested **{count}** chunks.")

        # Chat input
        if prompt := st.chat_input("Message Emy..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🧠"):
                with st.spinner(""):
                    resp = emy.respond(prompt, session_id="streamlit")

                st.markdown(resp.answer)

                meta = {
                    "trace": resp.trace,
                    "tools_used": resp.tools_used,
                    "score": resp.score.model_dump() if resp.score else None,
                    "sources": [
                        {"source": s.source, "score": s.score, "text": s.text[:200]}
                        for s in (resp.sources or [])
                    ],
                }

                msg_idx = len(st.session_state.messages)
                render_meta_hamburger(meta, msg_idx)

            st.session_state.messages.append({
                "role": "assistant",
                "content": resp.answer,
                "meta": meta,
                "_idx": msg_idx,
            })


# ── Training Tab ──────────────────────────────────────────────────────

if "Training" in tab_names:
    train_tab = tabs[tab_names.index("Training")]
    with train_tab:
        train_section = st.radio(
            "Section",
            ["Upload & Ingest", "Add Fact", "Add Reflection", "Add Training Example", "Add Entity"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if train_section == "Upload & Ingest":
            st.markdown("#### Upload Documents")
            st.caption(
                "Drag & drop files for Emy to ingest. "
                "ZIP archives are automatically extracted."
            )

            uploaded_files = st.file_uploader(
                "Drop files here",
                type=UPLOAD_EXTENSIONS,
                accept_multiple_files=True,
                help="Supports: " + ", ".join(SUPPORTED_EXTENSIONS.keys()),
                label_visibility="collapsed",
            )

            if uploaded_files:
                st.caption(f"{len(uploaded_files)} file(s) selected")

            col1, col2 = st.columns([1, 3])
            with col1:
                ingest_btn = st.button(
                    "Ingest Files", type="primary", disabled=not uploaded_files
                )

            if ingest_btn and uploaded_files:
                with st.spinner("Ingesting files..."):
                    count = ingest_uploaded_files(emy, uploaded_files)
                st.success(f"Ingested **{count}** chunks from {len(uploaded_files)} file(s).")

            st.divider()

            st.markdown("#### Ingest from Directory")
            corpus_dir = st.text_input(
                "Corpus directory path",
                value="./data/sample_docs",
                help="Path to a local directory containing documents.",
            )
            if st.button("Ingest Directory"):
                if corpus_dir.strip():
                    with st.spinner("Ingesting..."):
                        try:
                            count = emy.ingest(corpus_dir.strip())
                            st.success(f"Ingested **{count}** chunks.")
                        except Exception as exc:
                            st.error(f"Error: {exc}")

        elif train_section == "Add Fact":
            st.markdown("#### Add a Fact")
            st.caption("Facts Emy should remember across conversations.")
            fact_cat = st.text_input(
                "Category", value="preferences",
                placeholder="e.g. preferences, constraints, domain",
            )
            fact_content = st.text_area(
                "Content", placeholder="e.g. User prefers concise, technical answers",
            )
            if st.button("Save Fact", type="primary"):
                if fact_cat.strip() and fact_content.strip():
                    emy.add_fact(fact_cat.strip(), fact_content.strip())
                    st.success(f"Fact saved in **[{fact_cat}]**.")
                else:
                    st.warning("Both category and content are required.")

        elif train_section == "Add Reflection":
            st.markdown("#### Add a Reflection")
            st.caption("Teach Emy a lesson for future responses.")
            lesson = st.text_area(
                "Lesson",
                placeholder="e.g. Skip retrieval for greeting-only messages to reduce latency",
            )
            if st.button("Save Reflection", type="primary"):
                if lesson.strip():
                    emy.add_reflection(lesson.strip())
                    st.success("Reflection saved.")
                else:
                    st.warning("Lesson text is required.")

        elif train_section == "Add Training Example":
            st.markdown("#### Add a Training Example")
            st.caption("Labeled examples for intent routing.")
            ex_text = st.text_input("Text", placeholder="e.g. compare these two methods")
            ex_label = st.text_input(
                "Intent label", placeholder="e.g. research, document_query, smalltalk",
            )
            if st.button("Save Example", type="primary"):
                if ex_text.strip() and ex_label.strip():
                    emy.add_training_example(ex_text.strip(), ex_label.strip())
                    st.success(f"Example saved with label **[{ex_label}]**.")
                else:
                    st.warning("Both text and label are required.")

        elif train_section == "Add Entity":
            st.markdown("#### Add an Entity")
            st.caption("Store named entities Emy should know about.")
            ent_name = st.text_input("Name", placeholder="e.g. FastAPI")
            ent_type = st.text_input("Type", placeholder="e.g. framework, person, company")
            ent_desc = st.text_area("Description", placeholder="e.g. Modern Python web framework")
            if st.button("Save Entity", type="primary"):
                if ent_name.strip() and ent_type.strip() and ent_desc.strip():
                    emy.add_entity(ent_name.strip(), ent_type.strip(), ent_desc.strip())
                    st.success(f"Entity **{ent_name}** saved.")
                else:
                    st.warning("All fields are required.")


# ── Entry point ───────────────────────────────────────────────────────

def main():
    """CLI entry point — just launches streamlit."""
    import sys
    import subprocess

    app_path = Path(__file__).resolve()
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
    )
