"""Emy v2 — Streamlit Flagship UI.

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
    ".pdf": "PDF (requires pypdf)",
    ".docx": "Word document (requires python-docx)",
}

UPLOAD_EXTENSIONS = list(SUPPORTED_EXTENSIONS.keys())


# ── Helpers ───────────────────────────────────────────────────────────

def get_emy() -> Emy:
    """Return a cached Emy instance across reruns."""
    if "emy" not in st.session_state:
        config = EmyConfig(
            workdir=Path("./runtime"),
            mode="deploy",
        )
        st.session_state.emy = Emy(config)
    return st.session_state.emy


def ingest_uploaded_files(emy: Emy, uploaded_files: list) -> int:
    """Write uploaded files to a temp dir and ingest them."""
    if not uploaded_files:
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        for uf in uploaded_files:
            dest = Path(tmp) / uf.name
            dest.write_bytes(uf.getvalue())
        return emy.ingest(tmp)


def export_all_data(emy: Emy) -> bytes:
    """Package runtime data (vault + indexes + logs) into a zip."""
    buf = io.BytesIO()
    workdir = emy.config.workdir
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if workdir.exists():
            for fp in sorted(workdir.rglob("*")):
                if fp.is_file():
                    arcname = fp.relative_to(workdir)
                    zf.write(fp, arcname)
    buf.seek(0)
    return buf.getvalue()


# ── Page config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Emy v2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .block-container { max-width: 960px; }
    .stChatMessage { border-radius: 12px; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    div[data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }
    div[data-testid="stSidebar"] label { color: #e0e0e0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Emy v2")
    st.caption("Memory-First Agentic RAG")

    st.divider()

    mode_choice = st.radio(
        "Mode",
        options=["Deployment", "Training", "Both"],
        index=0,
        help=(
            "**Deployment** — Chat with Emy using existing knowledge (no new training).\n\n"
            "**Training** — Add facts, reflections, examples, and ingest documents.\n\n"
            "**Both** — Full access to chat and training tools together."
        ),
    )

    # Map UI mode to Emy internal mode
    if mode_choice == "Deployment":
        emy_mode = "deploy"
    elif mode_choice == "Training":
        emy_mode = "train"
    else:  # Both
        emy_mode = "train"

    emy = get_emy()
    emy.set_mode(emy_mode)

    st.divider()

    # ── Status ──
    if st.button("System Status", use_container_width=True):
        s = emy.status()
        st.json(s)

    st.divider()

    # ── Data Export ──
    st.subheader("Export Data")
    st.caption("Download all runtime data (vault, indexes, logs) as a zip.")
    if st.button("Package & Download", use_container_width=True):
        with st.spinner("Packaging..."):
            zip_bytes = export_all_data(emy)
        st.download_button(
            label="Download emy_data.zip",
            data=zip_bytes,
            file_name="emy_data.zip",
            mime="application/zip",
            use_container_width=True,
        )

# ── Main area ─────────────────────────────────────────────────────────

st.header("Emy v2")
st.caption(f"Mode: **{mode_choice}** | Engine: `{emy.config.llm_model}`")

# ── Tabs based on mode ────────────────────────────────────────────────

if mode_choice == "Deployment":
    tabs = st.tabs(["Chat"])
    tab_chat = tabs[0]
    tab_train = None
elif mode_choice == "Training":
    tabs = st.tabs(["Training"])
    tab_chat = None
    tab_train = tabs[0]
else:  # Both
    tabs = st.tabs(["Chat", "Training"])
    tab_chat = tabs[0]
    tab_train = tabs[1]

# ── Chat Tab ──────────────────────────────────────────────────────────

if tab_chat is not None:
    with tab_chat:
        # Init chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("meta"):
                    meta = msg["meta"]
                    with st.expander("Details"):
                        if meta.get("trace"):
                            st.markdown("**Trace**")
                            st.code("\n".join(meta["trace"]), language=None)
                        if meta.get("tools_used"):
                            st.markdown(f"**Tools:** {', '.join(meta['tools_used'])}")
                        if meta.get("score"):
                            sc = meta["score"]
                            cols = st.columns(4)
                            cols[0].metric("Faithfulness", f"{sc['faithfulness']:.2f}")
                            cols[1].metric("Relevance", f"{sc['relevance']:.2f}")
                            cols[2].metric("Completeness", f"{sc['completeness']:.2f}")
                            cols[3].metric("Overall", f"{sc['overall']:.2f}")
                            if sc.get("lesson"):
                                st.info(f"Lesson: {sc['lesson']}")
                        if meta.get("sources"):
                            st.markdown("**Sources**")
                            for src in meta["sources"]:
                                st.markdown(f"- `{src['source']}` (score {src['score']:.3f}): {src['text'][:150]}")

        # Chat input
        if prompt := st.chat_input("Ask Emy anything..."):
            # Show user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
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

                if resp.tools_used or resp.score:
                    with st.expander("Details"):
                        if resp.trace:
                            st.markdown("**Trace**")
                            st.code("\n".join(resp.trace), language=None)
                        if resp.tools_used:
                            st.markdown(f"**Tools:** {', '.join(resp.tools_used)}")
                        if resp.score:
                            cols = st.columns(4)
                            cols[0].metric("Faithfulness", f"{resp.score.faithfulness:.2f}")
                            cols[1].metric("Relevance", f"{resp.score.relevance:.2f}")
                            cols[2].metric("Completeness", f"{resp.score.completeness:.2f}")
                            cols[3].metric("Overall", f"{resp.score.overall:.2f}")
                            if resp.score.lesson:
                                st.info(f"Lesson: {resp.score.lesson}")
                        if resp.sources:
                            st.markdown("**Sources**")
                            for s in resp.sources:
                                st.markdown(f"- `{s.source}` (score {s.score:.3f}): {s.text[:150]}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": resp.answer,
                "meta": meta,
            })

# ── Training Tab ──────────────────────────────────────────────────────

if tab_train is not None:
    with tab_train:
        st.subheader("Training Tools")

        train_section = st.radio(
            "Section",
            ["Upload & Ingest", "Add Fact", "Add Reflection", "Add Training Example"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if train_section == "Upload & Ingest":
            st.markdown("### Upload Documents")
            st.markdown(
                "Upload files for Emy to ingest into its document index. "
                "**Supported formats:**"
            )

            # Show supported extensions in a clean grid
            ext_cols = st.columns(3)
            for i, (ext, desc) in enumerate(SUPPORTED_EXTENSIONS.items()):
                ext_cols[i % 3].markdown(f"- `{ext}` — {desc}")

            uploaded_files = st.file_uploader(
                "Choose files",
                type=[e.lstrip(".") for e in UPLOAD_EXTENSIONS],
                accept_multiple_files=True,
                help="Select one or more files to ingest.",
            )

            if uploaded_files:
                st.caption(f"{len(uploaded_files)} file(s) selected")

            col1, col2 = st.columns([1, 3])
            with col1:
                ingest_btn = st.button("Ingest Files", type="primary", disabled=not uploaded_files)

            if ingest_btn and uploaded_files:
                with st.spinner("Ingesting files..."):
                    count = ingest_uploaded_files(emy, uploaded_files)
                st.success(f"Ingested **{count}** chunks from {len(uploaded_files)} file(s).")

            st.divider()

            st.markdown("### Ingest from Directory")
            corpus_dir = st.text_input(
                "Corpus directory path",
                value="./data/sample_docs",
                help="Path to a local directory containing documents to ingest.",
            )
            if st.button("Ingest Directory"):
                if corpus_dir.strip():
                    with st.spinner("Ingesting..."):
                        try:
                            count = emy.ingest(corpus_dir.strip())
                            st.success(f"Ingested **{count}** chunks from `{corpus_dir}`.")
                        except Exception as exc:
                            st.error(f"Error: {exc}")

        elif train_section == "Add Fact":
            st.markdown("### Add a Fact")
            st.caption("Store facts that Emy should remember across conversations.")
            fact_cat = st.text_input("Category", value="preferences", placeholder="e.g. preferences, constraints, domain")
            fact_content = st.text_area("Content", placeholder="e.g. User prefers concise, technical answers")
            if st.button("Save Fact", type="primary"):
                if fact_cat.strip() and fact_content.strip():
                    emy.add_fact(fact_cat.strip(), fact_content.strip())
                    st.success(f"Fact saved in **[{fact_cat}]**.")
                else:
                    st.warning("Both category and content are required.")

        elif train_section == "Add Reflection":
            st.markdown("### Add a Reflection")
            st.caption("Teach Emy a lesson it should apply to future responses.")
            lesson = st.text_area("Lesson", placeholder="e.g. Skip retrieval for greeting-only messages to reduce latency")
            if st.button("Save Reflection", type="primary"):
                if lesson.strip():
                    emy.add_reflection(lesson.strip())
                    st.success("Reflection saved.")
                else:
                    st.warning("Lesson text is required.")

        elif train_section == "Add Training Example":
            st.markdown("### Add a Training Example")
            st.caption("Provide labeled examples for intent routing.")
            ex_text = st.text_input("Text", placeholder="e.g. compare these two methods")
            ex_label = st.text_input("Intent label", placeholder="e.g. research, document_query, smalltalk")
            if st.button("Save Example", type="primary"):
                if ex_text.strip() and ex_label.strip():
                    emy.add_training_example(ex_text.strip(), ex_label.strip())
                    st.success(f"Training example saved with label **[{ex_label}]**.")
                else:
                    st.warning("Both text and label are required.")


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
