"""
╔══════════════════════════════════════════════════════╗
║         NEXUS — Multi-Utility AI Chat Frontend       ║
║         Streamlit UI for LangGraph RAG Backend       ║
╚══════════════════════════════════════════════════════╝

HOW FILE ROUTING WORKS:
  PDF / DOCX / TXT  → RAG indexed (vector search) + read_full_document tool
  CSV / XLSX        → Absolute path passed to run_python_data_analysis (pandas)
  PNG / JPG / JPEG  → Absolute path passed to analyze_image (vision LLM)
  ALL FILES         → Path injected into every agent call via active_files config

Run:  streamlit run app.py
Req:  langgraph_rag_backend.py in the same directory
"""

import base64
import io
import os
import uuid
from datetime import datetime

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# ── Backend ────────────────────────────────────────────────────────────────────
from backend import chatbot, ingest_files

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NEXUS — AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Manrope:wght@300;400;500;600;700&display=swap');

:root {
    --bg:          #0d0d0f;
    --surface:     #141417;
    --s2:          #1c1c21;
    --s3:          #242429;
    --border:      #2a2a31;
    --bhi:         #38383f;
    --accent:      #f0a500;
    --adim:        #c4860a;
    --aglow:       rgba(240,165,0,0.13);
    --abg:         rgba(240,165,0,0.07);
    --t1:          #f0ede8;
    --t2:          #a09d98;
    --t3:          #6a6760;
    --green:       #4ade80;
    --gbg:         rgba(74,222,128,0.09);
    --blue:        #60a5fa;
    --bbg:         rgba(96,165,250,0.09);
    --purple:      #c084fc;
    --pbg:         rgba(192,132,252,0.09);
    --r:           10px;
    --rl:          16px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--t1) !important;
    font-family: 'Manrope', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none!important; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--t1) !important; }

.main .block-container {
    padding: 0 2rem 7rem 2rem !important;
    max-width: 860px !important;
    margin: 0 auto !important;
}

/* ── Header ── */
.nx-hdr {
    display:flex;align-items:center;gap:14px;
    padding:26px 0 18px 0;border-bottom:1px solid var(--border);margin-bottom:22px;
}
.nx-logo {
    width:40px;height:40px;background:var(--accent);border-radius:10px;
    display:flex;align-items:center;justify-content:center;font-size:19px;
    box-shadow:0 0 22px var(--aglow);flex-shrink:0;
}
.nx-title { font-family:'DM Serif Display',serif;font-size:26px;color:var(--t1);letter-spacing:-.4px;line-height:1; }
.nx-sub   { font-size:11px;color:var(--t3);letter-spacing:.09em;text-transform:uppercase;font-weight:600;margin-top:3px; }
.nx-badge { margin-left:auto;background:var(--s2);border:1px solid var(--border);
            border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;font-size:11px;color:var(--t3); }

/* ── Welcome ── */
.welcome {
    background:var(--surface);border:1px solid var(--border);border-radius:var(--rl);
    padding:30px 28px;margin-bottom:22px;text-align:center;
}
.welcome h2 { font-family:'DM Serif Display',serif;font-size:21px;color:var(--t1);margin-bottom:8px; }
.welcome p  { color:var(--t2);font-size:13.5px;line-height:1.65;max-width:480px;margin:0 auto 18px; }
.caps { display:flex;flex-wrap:wrap;gap:7px;justify-content:center; }
.chip { background:var(--s2);border:1px solid var(--bhi);border-radius:8px;
        padding:6px 12px;font-size:12px;color:var(--t2);display:flex;align-items:center;gap:6px; }
.chip b { color:var(--accent);font-size:13px; }

/* ── File panel ── */
.fp {
    background:var(--surface);border:1px solid var(--border);
    border-radius:var(--rl);padding:18px 20px;margin-bottom:18px;
}
.fp-hd {
    font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
    color:var(--t3);margin-bottom:12px;display:flex;align-items:center;gap:8px;
}
.fp-hd .n { background:var(--accent);color:#0d0d0f;border-radius:10px;
             padding:1px 7px;font-size:10px;font-weight:800; }

/* Route tags */
.rt-row { display:flex;gap:7px;flex-wrap:wrap;margin-bottom:14px; }
.rt {
    border-radius:7px;padding:6px 12px;font-size:11.5px;
    display:flex;align-items:center;gap:6px;font-weight:600;
}
.rt-rag  { background:var(--bbg);border:1px solid rgba(96,165,250,.22);color:var(--blue); }
.rt-data { background:var(--gbg);border:1px solid rgba(74,222,128,.22);color:var(--green); }
.rt-img  { background:var(--abg);border:1px solid rgba(240,165,0,.22);color:var(--accent); }

/* File cards */
.fc-grid { display:flex;flex-wrap:wrap;gap:9px;margin-top:14px; }
.fc {
    display:flex;align-items:flex-start;gap:9px;
    background:var(--s2);border:1px solid var(--bhi);border-radius:10px;
    padding:10px 13px;min-width:160px;max-width:210px;
    position:relative;
}
.fc.fc-img { flex-direction:column;align-items:stretch;max-width:180px; }
.fc-icon { font-size:19px;flex-shrink:0;margin-top:1px; }
.fc-name { font-size:12px;color:var(--t1);
           white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:160px; }
.fc-meta { font-size:10px;color:var(--t3);margin-top:2px; }
.fc-tag  {
    font-size:9.5px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
    padding:2px 7px;border-radius:4px;margin-top:4px;display:inline-block;
}
.tg-rag  { background:var(--bbg);color:var(--blue); }
.tg-data { background:var(--gbg);color:var(--green); }
.tg-img  { background:var(--abg);color:var(--accent); }
.tg-doc  { background:var(--pbg);color:var(--purple); }

.fc-thumb {
    width:100%;height:72px;object-fit:cover;
    border-radius:6px;border:1px solid var(--border);display:block;margin-bottom:7px;
}

/* Suggestions */
.sug {
    font-size:11.5px;color:var(--t3);margin-top:12px;line-height:1.9;
}
.sug b { color:var(--t2); }

/* ── Chat bubbles ── */
.crow { display:flex;gap:12px;margin-bottom:18px;animation:fsi .22s ease-out; }
.crow.uc { flex-direction:row-reverse; }
@keyframes fsi { from{opacity:0;transform:translateY(7px)} to{opacity:1;transform:translateY(0)} }
.av {
    width:32px;height:32px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    font-size:14px;flex-shrink:0;margin-top:3px;
}
.av.ai  { background:var(--accent);box-shadow:0 0 12px var(--aglow); }
.av.usr { background:var(--s3);border:1px solid var(--bhi); }
.bub {
    max-width:78%;padding:12px 16px;border-radius:var(--rl);
    font-size:14px;line-height:1.68;word-break:break-word;
}
.bub.ai  { background:var(--s2);border:1px solid var(--border);border-top-left-radius:4px;color:var(--t1); }
.bub.usr { background:var(--accent);border-top-right-radius:4px;color:#0d0d0f;font-weight:500; }

/* ── Tool pills ── */
.trow { display:flex;flex-wrap:wrap;gap:5px;margin-bottom:6px;padding-left:44px; }
.tp {
    display:inline-flex;align-items:center;gap:5px;
    background:var(--s3);border:1px solid var(--bhi);
    border-radius:20px;padding:3px 10px;
    font-size:11px;color:var(--t2);font-family:'DM Mono',monospace;
}
.td { width:5px;height:5px;border-radius:50%;background:var(--accent);animation:pulse 1.2s infinite; }
@keyframes pulse { 0%,100%{opacity:1}50%{opacity:.25} }

/* ── Sidebar ── */
.sbh {
    font-size:10px;font-weight:700;letter-spacing:.11em;text-transform:uppercase;
    color:var(--t3);margin:18px 0 7px 0;padding-bottom:5px;border-bottom:1px solid var(--border);
}
.sbd {
    display:flex;align-items:center;gap:8px;
    background:var(--s2);border:1px solid var(--border);border-left:3px solid var(--accent);
    border-radius:6px;padding:7px 10px;font-size:11.5px;color:var(--t2);margin-bottom:5px;
}
.sbd-n { white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:160px; }
.sbd-m { font-size:9.5px;color:var(--t3);margin-top:1px; }

/* ── Streamlit overrides ── */
[data-testid="stChatInput"] textarea {
    background:var(--s2)!important;border:1px solid var(--bhi)!important;
    border-radius:var(--r)!important;color:var(--t1)!important;
    font-family:'Manrope',sans-serif!important;font-size:14px!important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color:var(--accent)!important;box-shadow:0 0 0 3px var(--aglow)!important;
}
.stButton>button {
    background:var(--s2)!important;color:var(--t2)!important;
    border:1px solid var(--bhi)!important;border-radius:8px!important;
    font-family:'Manrope',sans-serif!important;font-size:13px!important;font-weight:500!important;
    transition:all .15s!important;
}
.stButton>button:hover {
    background:var(--s3)!important;border-color:var(--accent)!important;color:var(--accent)!important;
}
[data-testid="stFileUploader"] {
    background:var(--s2)!important;border:1.5px dashed var(--bhi)!important;border-radius:var(--r)!important;
}
[data-testid="stFileUploader"] * { color:var(--t2)!important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color:var(--t3)!important; }
[data-testid="stSelectbox"]>div>div {
    background:var(--s2)!important;border-color:var(--bhi)!important;
    color:var(--t1)!important;border-radius:8px!important;
}
[data-testid="stAlert"] { background:var(--s2)!important;border-radius:var(--r)!important; }
[data-testid="stSpinner"]>div { border-top-color:var(--accent)!important; }
[data-testid="stExpander"] {
    background:var(--s2)!important;border:1px solid var(--border)!important;border-radius:var(--r)!important;
}
hr { border-color:var(--border)!important; }
::-webkit-scrollbar { width:4px;height:4px; }
::-webkit-scrollbar-thumb { background:var(--bhi);border-radius:10px; }
code {
    background:var(--s3)!important;color:var(--accent)!important;
    font-family:'DM Mono',monospace!important;border-radius:4px!important;
    padding:1px 5px!important;font-size:12.5px!important;
}
pre code { display:block!important;padding:12px!important;overflow-x:auto!important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
TOOL_ICONS = {
    "search_web":                    "🌐",
    "calculator":                    "🧮",
    "rag_retrieve":                  "🔍",
    "run_python_data_analysis":      "🐍",
    "read_full_document":            "📖",
    "analyze_image":                 "🖼️",
    "translate_to_bengali":          "🔤",
    "write_job_application_message": "✍️",
}

# Per-extension routing metadata shown to user
EXT_INFO = {
    ".pdf":  {"icon":"📄","tag":"RAG",  "cls":"tg-rag",  "hint":"vector search + full read"},
    ".docx": {"icon":"📝","tag":"RAG",  "cls":"tg-rag",  "hint":"vector search + full read"},
    ".doc":  {"icon":"📝","tag":"RAG",  "cls":"tg-rag",  "hint":"vector search + full read"},
    ".txt":  {"icon":"📃","tag":"RAG",  "cls":"tg-rag",  "hint":"vector search + full read"},
    ".csv":  {"icon":"📊","tag":"DATA", "cls":"tg-data", "hint":"pandas / Python analysis"},
    ".xlsx": {"icon":"📊","tag":"DATA", "cls":"tg-data", "hint":"pandas / Python analysis"},
    ".xls":  {"icon":"📊","tag":"DATA", "cls":"tg-data", "hint":"pandas / Python analysis"},
    ".png":  {"icon":"🖼️","tag":"IMG",  "cls":"tg-img",  "hint":"vision LLM"},
    ".jpg":  {"icon":"🖼️","tag":"IMG",  "cls":"tg-img",  "hint":"vision LLM"},
    ".jpeg": {"icon":"🖼️","tag":"IMG",  "cls":"tg-img",  "hint":"vision LLM"},
}
SUPPORTED = [e.lstrip(".") for e in EXT_INFO]
UPLOAD_DIR = "/tmp/nexus_uploads"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def new_tid() -> str:
    return str(uuid.uuid4())[:8]


def save_file(uf) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, uf.name)
    with open(path, "wb") as f:
        f.write(uf.getbuffer())
    return path


def ext_info(fname: str) -> dict:
    ext = os.path.splitext(fname)[-1].lower()
    return EXT_INFO.get(ext, {"icon":"📁","tag":"FILE","cls":"tg-doc","hint":"general"})


def make_thumb(raw: bytes) -> str | None:
    """Return base64 PNG thumbnail or None."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw))
        img.thumbnail((240, 120))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def render_msg(role: str, content: str):
    if role == "user":
        st.markdown(f"""
        <div class="crow uc">
            <div class="av usr">👤</div>
            <div class="bub usr">{content}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="crow">
            <div class="av ai">🧠</div>
            <div class="bub ai">{content}</div>
        </div>""", unsafe_allow_html=True)


def render_tools(tools: list):
    if not tools:
        return
    pills = "".join(
        f'<span class="tp"><span class="td"></span>'
        f'{TOOL_ICONS.get(t,"⚙️")} {t.replace("_"," ").title()}</span>'
        for t in tools
    )
    st.markdown(f'<div class="trow">{pills}</div>', unsafe_allow_html=True)


def reset_chat():
    tid = new_tid()
    st.session_state.thread_id   = tid
    st.session_state.messages    = []
    st.session_state.active_files = {}
    st.session_state.all_threads.insert(0, {
        "id": tid,
        "created": datetime.now().strftime("%b %d, %H:%M"),
        "preview": "New conversation",
    })


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "thread_id"     not in st.session_state: st.session_state.thread_id     = new_tid()
if "messages"      not in st.session_state: st.session_state.messages      = []
if "active_files"  not in st.session_state: st.session_state.active_files  = {}
# active_files: { fname: {path,ext,icon,tag,cls,hint,chunks,pages,size_kb,thumb_b64} }
if "all_threads"   not in st.session_state:
    st.session_state.all_threads = [{
        "id": st.session_state.thread_id,
        "created": datetime.now().strftime("%b %d, %H:%M"),
        "preview": "New conversation",
    }]

thread_id = st.session_state.thread_id


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 16px 0;border-bottom:1px solid var(--border);margin-bottom:6px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:30px;height:30px;background:var(--accent);border-radius:8px;
                        display:flex;align-items:center;justify-content:center;font-size:15px;
                        box-shadow:0 0 13px var(--aglow);">🧠</div>
            <div>
                <div style="font-family:'DM Serif Display',serif;font-size:19px;line-height:1;">NEXUS</div>
                <div style="font-size:10px;color:var(--t3);letter-spacing:.1em;text-transform:uppercase;">Multi-Utility AI</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("＋  New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    if st.session_state.active_files:
        st.markdown('<div class="sbh">🗂️ Files in this chat</div>', unsafe_allow_html=True)
        for fname, meta in st.session_state.active_files.items():
            chunks = meta.get("chunks","—"); pages = meta.get("pages","—")
            detail = " · ".join(filter(lambda x: x!="—", [
                f"{chunks} chunks" if chunks!="—" else "",
                f"{pages}p" if pages!="—" else "",
            ])) or f"{meta['size_kb']} KB"
            st.markdown(f"""
            <div class="sbd">
                <span style="font-size:15px;">{meta['icon']}</span>
                <div style="overflow:hidden;">
                    <div class="sbd-n">{fname}</div>
                    <div class="sbd-m">{meta['hint']} · {detail}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sbh">🛠️ Tools</div>', unsafe_allow_html=True)
    for icon, label in [
        ("🌐","Web Search"),("🧮","Calculator"),("🔍","RAG Retrieval"),
        ("🐍","Python / pandas"),("📖","Read Document"),("🖼️","Analyze Image"),
        ("🔤","Bengali Translate"),("✍️","Job Application"),
    ]:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:5px 0;
                    border-bottom:1px solid var(--border);font-size:12px;color:var(--t2);">
            <span style="font-size:13px;">{icon}</span>{label}
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sbh">💬 Conversations</div>', unsafe_allow_html=True)
    for t in st.session_state.all_threads:
        tid  = t["id"]
        active = tid == thread_id
        label  = f"{'▶ ' if active else ''}#{tid}  {t['created']}"
        if st.button(label, key=f"t_{tid}", use_container_width=True):
            if tid != thread_id:
                st.session_state.thread_id = tid
                state = chatbot.get_state(config={"configurable":{"thread_id":tid}})
                raw   = state.values.get("messages", [])
                st.session_state.messages = []
                for m in raw:
                    if isinstance(m, HumanMessage) and m.content:
                        st.session_state.messages.append({"role":"user","content":m.content,"tools":[]})
                    elif isinstance(m, AIMessage) and m.content:
                        st.session_state.messages.append({"role":"assistant","content":m.content,"tools":[]})
                st.session_state.active_files = {}
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
n = len(st.session_state.active_files)

st.markdown(f"""
<div class="nx-hdr">
    <div class="nx-logo">🧠</div>
    <div>
        <div class="nx-title">NEXUS</div>
        <div class="nx-sub">Multi-Utility AI Assistant</div>
    </div>
    <div class="nx-badge">#{thread_id}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ▼ FILE PANEL — always rendered above chat
# ══════════════════════════════════════════════════════════════════════════════
expander_label = (
    f"📎  Attach Files — {n} file{'s' if n!=1 else ''} loaded, agent can see them all"
    if n else "📎  Attach Files — click to upload PDFs, images, CSVs…"
)

with st.expander(expander_label, expanded=(n == 0)):

    # ── Routing legend ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="rt-row">
        <div class="rt rt-rag">📄 PDF / DOCX / TXT → RAG vector search + full read</div>
        <div class="rt rt-data">📊 CSV / XLSX → Python pandas analysis</div>
        <div class="rt rt-img">🖼️ PNG / JPG → Vision LLM (describe, OCR, extract)</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Uploader ────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop any files here — multiple files supported simultaneously",
        type=SUPPORTED,
        accept_multiple_files=True,
        label_visibility="visible",
        key=f"fu_{thread_id}",
    )

    # ── Process new files ───────────────────────────────────────────────────
    if uploaded:
        new_files = [f for f in uploaded if f.name not in st.session_state.active_files]
        if new_files:
            prog = st.progress(0, text="Saving…")
            file_list, paths = [], {}

            for i, uf in enumerate(new_files):
                prog.progress((i+1)/(len(new_files)*2), text=f"Saving {uf.name}…")
                path = save_file(uf)
                paths[uf.name] = path
                file_list.append({"bytes": uf.getvalue(), "name": uf.name, "path": path})

            prog.progress(0.6, text="Indexing RAG vector store…")
            summaries = ingest_files(file_list, thread_id)
            prog.progress(1.0, text="Done ✓")
            prog.empty()

            for uf, summary in zip(new_files, summaries):
                info  = ext_info(uf.name)
                raw   = uf.getvalue()
                thumb = make_thumb(raw) if info["tag"] == "IMG" else None
                st.session_state.active_files[uf.name] = {
                    "path":      paths[uf.name],
                    "ext":       os.path.splitext(uf.name)[-1].lower(),
                    "icon":      info["icon"],
                    "tag":       info["tag"],
                    "cls":       info["cls"],
                    "hint":      info["hint"],
                    "chunks":    summary.get("chunks", "—"),
                    "pages":     summary.get("pages", summary.get("sheets","—")),
                    "size_kb":   round(len(raw)/1024, 1),
                    "thumb_b64": thumb,
                }
            st.rerun()

    # ── File cards ──────────────────────────────────────────────────────────
    if st.session_state.active_files:
        st.markdown(
            f"<div style='font-size:12px;color:var(--t2);margin-bottom:8px;'>"
            f"✅ <b>{n} file{'s' if n!=1 else ''}</b> loaded — "
            f"the agent has their full paths and will use the right tool automatically.</div>",
            unsafe_allow_html=True,
        )

        # Render cards in rows of 4
        fnames = list(st.session_state.active_files.keys())
        to_remove = []

        for row_start in range(0, len(fnames), 4):
            row_names = fnames[row_start : row_start + 4]
            cols = st.columns(len(row_names))
            for col, fname in zip(cols, row_names):
                meta = st.session_state.active_files[fname]
                is_img = meta["thumb_b64"] is not None
                with col:
                    if is_img:
                        st.markdown(f"""
                        <div class="fc fc-img">
                            <img class="fc-thumb" src="data:image/png;base64,{meta['thumb_b64']}" alt="{fname}"/>
                            <div class="fc-name" title="{fname}">{fname}</div>
                            <div class="fc-meta">{meta['size_kb']} KB</div>
                            <span class="fc-tag {meta['cls']}">{meta['tag']} · {meta['hint']}</span>
                        </div>""", unsafe_allow_html=True)
                    else:
                        chunks = meta.get("chunks","—"); pages = meta.get("pages","—")
                        parts = []
                        if chunks!="—": parts.append(f"{chunks} chunks")
                        if pages !="—": parts.append(f"{pages}p")
                        detail = " · ".join(parts) or f"{meta['size_kb']} KB"
                        st.markdown(f"""
                        <div class="fc">
                            <span class="fc-icon">{meta['icon']}</span>
                            <div>
                                <div class="fc-name" title="{fname}">{fname}</div>
                                <div class="fc-meta">{detail}</div>
                                <span class="fc-tag {meta['cls']}">{meta['tag']} · {meta['hint']}</span>
                            </div>
                        </div>""", unsafe_allow_html=True)

                    if st.button("✕ Remove", key=f"rm_{fname}", use_container_width=True):
                        to_remove.append(fname)

        if to_remove:
            for f in to_remove:
                st.session_state.active_files.pop(f, None)
            st.rerun()

        # ── Contextual prompt suggestions ───────────────────────────────────
        has_img  = any(m["tag"]=="IMG"  for m in st.session_state.active_files.values())
        has_data = any(m["tag"]=="DATA" for m in st.session_state.active_files.values())
        has_rag  = any(m["tag"]=="RAG"  for m in st.session_state.active_files.values())

        sugs = []
        if has_img:  sugs += ["🖼️ *Describe what's in the image*", "🖼️ *Extract all text from the image*"]
        if has_data: sugs += ["🐍 *What are the column names and how many rows?*", "🐍 *Show me summary statistics*"]
        if has_rag:  sugs += ["📖 *Summarize this document*", "🔍 *What does this say about [topic]?*"]

        if sugs:
            sug_html = "  ·  ".join(sugs)
            st.markdown(f'<div class="sug">💡 Try: {sug_html}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ACTIVE FILE PATHS (injected into every agent turn)
# ══════════════════════════════════════════════════════════════════════════════
active_path_str = (
    "\n".join(f"{fname}  →  {meta['path']}"
              for fname, meta in st.session_state.active_files.items())
    if st.session_state.active_files else "No files uploaded."
)


# ══════════════════════════════════════════════════════════════════════════════
# WELCOME (empty chat)
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <h2>What can I help you with?</h2>
        <p>Upload files above, then ask anything — or chat without files to search the web,
        translate to Bengali, run calculations, or write job applications.</p>
        <div class="caps">
            <div class="chip"><b>🌐</b> Web Search</div>
            <div class="chip"><b>📄</b> PDF / Doc Analysis</div>
            <div class="chip"><b>🐍</b> CSV / Excel Analysis</div>
            <div class="chip"><b>🖼️</b> Image Understanding</div>
            <div class="chip"><b>🔤</b> Bengali Translation</div>
            <div class="chip"><b>🧮</b> Calculator</div>
            <div class="chip"><b>✍️</b> Job Applications</div>
            <div class="chip"><b>🔍</b> Smart RAG Search</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    if msg["role"] == "assistant" and msg.get("tools"):
        render_tools(msg["tools"])
    render_msg(msg["role"], msg["content"])


# ══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT + STREAMING
# ══════════════════════════════════════════════════════════════════════════════
user_input = st.chat_input(
    f"Ask anything… ({n} file{'s' if n!=1 else ''} loaded)"
    if n else "Ask anything, or upload files above first…"
)

if user_input:
    st.session_state.messages.append({"role":"user","content":user_input,"tools":[]})
    render_msg("user", user_input)

    for t in st.session_state.all_threads:
        if t["id"] == thread_id:
            t["preview"] = user_input[:50]
            break

    config = {
        "configurable": {"thread_id": thread_id, "active_files": active_path_str},
        "recursion_limit": 25,
    }

    tools_turn: list[str] = []
    tool_ph  = st.empty()
    resp_ph  = st.empty()
    full     = ""

    with st.spinner(""):
        for chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        ):
            # ── Tool call starting ─────────────────────────────────────────
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                for tc in chunk.tool_calls:
                    tname = tc.get("name", "tool")
                    if tname not in tools_turn:
                        tools_turn.append(tname)
                        pills = "".join(
                            f'<span class="tp"><span class="td"></span>'
                            f'{TOOL_ICONS.get(t,"⚙️")} {t.replace("_"," ").title()}</span>'
                            for t in tools_turn
                        )
                        tool_ph.markdown(
                            f'<div class="trow">{pills}</div>',
                            unsafe_allow_html=True,
                        )

            # ── AI text streaming ──────────────────────────────────────────
            elif hasattr(chunk, "content") and chunk.content:
                if not (hasattr(chunk, "tool_calls") and chunk.tool_calls):
                    full += chunk.content
                    tool_ph.empty()
                    resp_ph.markdown(f"""
                    <div class="crow">
                        <div class="av ai">🧠</div>
                        <div class="bub ai">{full}▌</div>
                    </div>""", unsafe_allow_html=True)

    tool_ph.empty()
    resp_ph.empty()

    if tools_turn:
        render_tools(tools_turn)
    if full:
        render_msg("assistant", full)

    st.session_state.messages.append({"role":"assistant","content":full,"tools":tools_turn})


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:30px 0 8px 0;color:var(--t3);font-size:11px;letter-spacing:.05em;">
    NEXUS · LangGraph + Ollama · 100% Local &amp; Private
</div>
""", unsafe_allow_html=True)