import base64
import io
import json
import logging
import os
import random
import re,sys
import tempfile
import time
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings


# --- FORCE THE AGENT TO USE YOUR OLLAMA SETUP ---
# os.environ["OLLAMA_MODEL"] = "qwen2.5-coder:14b"
# os.environ["OLLAMA_EMBED_MODEL"] = "qwen3-embedding:4b"
# os.environ["OLLAMA_VISION"] = "qwen3.5:9b"

# os.environ["OLLAMA_HOST"] = "http://localhost:11434"

# Langchain & Langgraph imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown

from langgraph.checkpoint.memory import MemorySaver 
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings

from dotenv import load_dotenv

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()

# ---------------------------------------------------------------------------
# 1. LLM & Embeddings Initialization using ollama
# ---------------------------------------------------------------------------
# llm = ChatOllama(
#     model=os.environ["OLLAMA_MODEL"],
#     base_url=os.environ["OLLAMA_HOST"],
#     temperature=0.7,
# )

# # Vision LLM instance for the image tool
# llm_vision = ChatOllama(
#     model=os.environ["OLLAMA_VISION"],
#     base_url=os.environ["OLLAMA_HOST"],
#     temperature=0.0, # 0.0 for maximum OCR accuracy
# )

# llm_accuracy=llm_vision

# embeddings = OllamaEmbeddings(
#     model=os.environ["OLLAMA_EMBED_MODEL"],
#     base_url=os.environ["OLLAMA_HOST"],
# )


nvidia_api_key = os.environ.get("NVIDIA_API_KEY")


# Get your token
hf_token = os.environ.get("HF_TOKEN")

# ---------------------------------------------------------------------------
# 1. Base LLM (Gemma 4 E4B)
# HuggingFaceEndpoint connects to the cloud API, ChatHuggingFace formats it for LangChain
# ---------------------------------------------------------------------------
# llm_endpoint = HuggingFaceEndpoint(
#     model=os.environ.get("HF_MODEL", "google/gemma-4-e4b-it"),
#     # repo_id=os.environ.get("HF_MODEL", "google/gemma-4-e4b-it"),
#     task="text-generation",
#     max_new_tokens=1024,   # How many tokens it can generate per response
#     temperature=0.7,
#     huggingfacehub_api_token=hf_token
# )

# llm = ChatHuggingFace(llm=llm_endpoint)

# ---------------------------------------------------------------------------
# 2. Vision LLM Initialization
# Gemma 4 E4B handles audio/text, but for Images (OCR), you'll want a Vision model
# like Idefics2 or LLaVA hosted on Hugging Face.
# ---------------------------------------------------------------------------
# vision_endpoint = HuggingFaceEndpoint(
#     model=os.environ.get("HF_VISION", "deepseek-ai/DeepSeek-V3-0324"),
#     # repo_id=os.environ.get("HF_VISION", "Qwen/Qwen2.5-VL-7B-Instruct"),
#     task="text-generation",
#     max_new_tokens=512,
#     temperature=0.01, # 0.01 is better than 0.0 for HF API to prevent division-by-zero errors
#     huggingfacehub_api_token=hf_token
# )

# llm_vision = ChatHuggingFace(llm=vision_endpoint)



# -------------------------------------------------------------------
# MAIN LLM
# -------------------------------------------------------------------
llm = ChatNVIDIA(
    model=os.environ.get(
        "NVIDIA_MODEL",
        "qwen/qwen3-coder-480b-a35b-instruct"
    ),
    api_key=nvidia_api_key,
    temperature=0.7,
    top_p=0.8,
    max_tokens=4096,
)

# -------------------------------------------------------------------
# VISION MODEL
# -------------------------------------------------------------------
llm_vision = ChatNVIDIA(
    model=os.environ.get(
        "NVIDIA_VISION_MODEL",
        "meta/llama-3.2-90b-vision-instruct"
    ),
    api_key=nvidia_api_key,
    temperature=0.01,
    max_tokens=2048,
)

llm_accuracy = llm_vision

# -------------------------------------------------------------------
# EMBEDDINGS
# -------------------------------------------------------------------
embeddings = NVIDIAEmbeddings(
    model=os.environ.get(
        "NVIDIA_EMBED_MODEL",
        "nvidia/nv-embedqa-e5-v5"
    ),
    api_key=nvidia_api_key,
)

llm_accuracy = llm_vision

# ---------------------------------------------------------------------------
# 3. Embeddings Initialization
# Replaces OllamaEmbeddings. Connects to HF's feature-extraction API.
# ---------------------------------------------------------------------------
# embeddings = HuggingFaceEndpointEmbeddings(
#     model=os.environ.get("HF_EMBED_MODEL", "BAAI/bge-m3"),
#     task="feature-extraction",
#     huggingfacehub_api_token=hf_token
# )


# ---------------------------------------------------------------------------
# 2. Per-thread document store
# ---------------------------------------------------------------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, List[dict]] = {}

def _get_retriever(thread_id: Optional[str]):
    return _THREAD_RETRIEVERS.get(thread_id)


# Public SearXNG instances - tries each one until one works
SEARXNG_INSTANCES = [
    "https://searx.be",
    "https://search.bus-hit.me",
    "https://searxng.world",
]

def _searxng_search(query: str) -> str:
    """Try multiple public SearXNG instances as fallback."""
    for instance in SEARXNG_INSTANCES:
        try:
            resp = requests.get(
                f"{instance}/search",
                params={"q": query, "format": "json", "engines": "google,bing,duckduckgo"},
                timeout=8,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])[:3]
                if results:
                    return "\n---\n".join(
                        f"Title: {r.get('title','')}\nURL: {r.get('url','')}\nSnippet: {r.get('content','')}"
                        for r in results
                    )
        except Exception as e:
            continue  # Try next instance
    return None



# ─────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────
 
def _read_resume(path: str) -> str:
    """Read resume from PDF / DOCX / TXT and return plain text."""
    if not os.path.exists(path):
        return f"[ERROR] Resume file not found: {path}"
    ext = os.path.splitext(path)[-1].lower()
    try:
        if ext == ".pdf":
            with pdfplumber.open(path) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            return text[:60000]
        else:
            md = MarkItDown()
            return md.convert(path).text_content[:60000]
    except Exception as e:
        return f"[ERROR] Could not read resume: {e}"
 
 
def _build_writer_prompt(resume: str, jd: str, platform: str, tone: str) -> str:
    return f"""You are a career coach helping a job seeker write a SHORT outreach message.
 
PLATFORM : {platform}
TONE     : {tone}
 
━━━━━━━━━━━━ CANDIDATE RESUME ━━━━━━━━━━━━
{resume}
 
━━━━━━━━━━━━ JOB DESCRIPTION ━━━━━━━━━━━━
{jd}
 
━━━━━━━━━━━━ RULES ━━━━━━━━━━━━
Write ONE outreach message (100–180 words). Follow EVERY rule:
 
1. Sound like a REAL human — not AI, not a template.
2. BANNED phrases (instant FAIL if used): "I am passionate", "fast learner",
   "I came across", "I would love to", "please find attached", "synergy",
   "leverage", "I am writing to express", "I am excited to apply",
   "I hope this message finds you well", "dynamic", "results-driven".
3. Open with something SPECIFIC from the JD — a product, a tech stack,
   a team goal, a challenge mentioned. NOT "My name is…" or "I saw your post…".
4. Mention 1–2 CONCRETE things from the resume (numbers, project names,
   specific skills that directly match what the JD needs).
5. ONE clear, low-friction ask at the end, suited to the platform:
   - LinkedIn  → e.g. "Happy to connect and share more."
   - Email     → e.g. "Would a quick 15-minute call this week work for you?"
   - WhatsApp  → e.g. "Let me know if you'd like to chat!"
6. Use contractions naturally: I've, I'd, let's, you're, it's, we've.
7. Max 4 short paragraphs. No bullet points. No subject line.
 
Output ONLY the message body. No preamble, no explanation, no quotes.
"""
 
 
def _build_evaluator_prompt(message: str, resume: str, jd: str) -> str:
    return f"""You are a senior recruiter evaluating a job-seeker's outreach message.
 
━━━━━━━━━━━━ MESSAGE TO EVALUATE ━━━━━━━━━━━━
{message}
 
━━━━━━━━━━━━ RESUME (for context) ━━━━━━━━━━━━
{resume[:2500]}
 
━━━━━━━━━━━━ JOB DESCRIPTION (for context) ━━━━━━━━━━━━
{jd[:2500]}
 
━━━━━━━━━━━━ SCORING CRITERIA ━━━━━━━━━━━━
Score each dimension 0–20 (total out of 100):
 
1. HUMAN_FEEL    – Reads like a real person typed it; zero AI patterns or banned phrases.
2. JD_RELEVANCE  – References something SPECIFIC from the job description (not generic).
3. RESUME_MATCH  – Mentions concrete, relevant achievements from the resume.
4. HOOK_STRENGTH – First sentence is specific and compelling enough not to be ignored.
5. CALL_TO_ACTION– Ask is clear, natural, and low-friction for the platform.
 
DEDUCT points heavily if you detect:
- Any banned phrase (see writer rules)
- Generic opener like "My name is…" or "I saw your job posting…"
- Vague claims with no numbers or specifics
- Bullet points or lists
 
Respond ONLY with valid JSON — no markdown fences, no explanation:
{{
  "scores": {{
    "human_feel": <int 0-20>,
    "jd_relevance": <int 0-20>,
    "resume_match": <int 0-20>,
    "hook_strength": <int 0-20>,
    "call_to_action": <int 0-20>
  }},
  "total": <int 0-100>,
  "weakest_point": "<one specific, actionable sentence on what to fix>",
  "verdict": "PASS" or "FAIL"
}}
 
PASS threshold = total >= 74. Be strict. Generic messages must FAIL.
"""
 
 
def _generate(llm, resume: str, jd: str, platform: str, tone: str) -> str:
    prompt = _build_writer_prompt(resume, jd, platform, tone)
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()
 
 
def _evaluate(llm, message: str, resume: str, jd: str) -> dict:
    prompt = _build_evaluator_prompt(message, resume, jd)
    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    # Strip markdown fences if model wraps output
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    # Fallback — force a retry
    return {
        "scores": {},
        "total": 0,
        "weakest_point": "Evaluator output could not be parsed — retrying.",
        "verdict": "FAIL",
    }
 
 
# ─────────────────────────────────────────────────────────────────
# THE TOOL
# ─────────────────────────────────────────────────────────────────
 
@tool
def write_job_application_message(
    resume_path: str,
    jd_text: str,
    platform: str = "LinkedIn",
    tone: str = "confident",
    max_attempts: int = 1,
) -> str:
    """
    Writes a short, human-sounding job outreach / application message (100–180 words).
 
    Call this whenever the user wants to apply for a job or reach out to a recruiter.
 
    The agent should extract:
    - resume_path : from the uploaded/active files list (the user's resume PDF)
    - jd_text     : from the user's message (they paste the job description as text)
 
    The tool self-evaluates each draft on 5 dimensions and regenerates
    automatically if the score is below 74/100, up to max_attempts times.
 
    Args:
        resume_path  : Absolute path to the candidate's resume (PDF, DOCX, or TXT).
        jd_text      : Full job description text pasted by the user in the chat.
        platform     : Where this message will be sent — LinkedIn | Email | WhatsApp.
                       Default: LinkedIn.
        tone         : Writing tone — confident | humble | enthusiastic.
                       Default: confident.
        max_attempts : How many times to regenerate if quality is low. Default: 1.
 
    Returns:
        The best message generated, with a quality score breakdown.
    """
    # Grab the LLM initialized in the parent script
    try:
        from __main__ import llm as _llm
    except (ImportError, AttributeError):
        try:
            from __main__ import llm as _llm
        except ImportError:
            return "[ERROR] LLM not found. Use this tool inside the agent script."
 
    # ── Read resume from file ──
    print(f"\n📄 Reading resume: {resume_path}")
    resume = _read_resume(resume_path)
    if resume.startswith("[ERROR]"):
        return resume
 
    # ── Validate JD text ──
    if not jd_text or len(jd_text.strip()) < 50:
        return (
            "[ERROR] Job description too short. "
            "Please paste the full JD text into the chat."
        )
 
    jd = jd_text.strip()
    print(f"📋 JD received: {len(jd)} characters")
    print(f"🎯 Platform: {platform}  |  Tone: {tone}")
    print(f"🔄 Will try up to {max_attempts} drafts\n")
    print("─" * 60)
 
    best_message = ""
    best_score   = -1
    best_eval    = {}
    history      = []
 
    for attempt in range(1, max_attempts + 1):
        print(f"\n✍️  Draft #{attempt} — Generating message...")
        message = _generate(_llm, resume, jd, platform, tone)
 
        print(f"🔍 Draft #{attempt} — Self-evaluating...")
        evaluation = _evaluate(_llm, message, resume, jd)
 
        total   = evaluation.get("total", 0)
        verdict = evaluation.get("verdict", "FAIL")
        weak    = evaluation.get("weakest_point", "N/A")
        scores  = evaluation.get("scores", {})
 
        history.append({"attempt": attempt, "score": total, "verdict": verdict})
 
        print(f"   Score   : {total}/100  ({verdict})")
        print(f"   Details : {scores}")
        print(f"   Fix     : {weak}")
 
        if total > best_score:
            best_score   = total
            best_message = message
            best_eval    = evaluation
 
        if verdict == "PASS":
            print(f"\n✅ Passed quality check on attempt #{attempt}!")
            break
        else:
            if attempt < max_attempts:
                print(f"   ↩️  Not good enough — regenerating (fix: {weak})...")
                time.sleep(0.8)
            else:
                print(
                    f"\n⚠️  Reached max attempts. "
                    f"Using best draft so far (score: {best_score}/100)."
                )
 
    # ── Format final output ──
    scores_display = "\n".join(
        f"   • {k.replace('_', ' ').title():<20}: {v}/20"
        for k, v in best_eval.get("scores", {}).items()
    )
    history_str = "  →  ".join(
        f"#{a['attempt']}: {a['score']}" for a in history
    )
 
    return f"""
╔══════════════════════════════════════════════════════════════════╗
║               YOUR JOB APPLICATION MESSAGE                      ║
╠══════════════════════════════════════════════════════════════════╣
 
{best_message}
 
╠══════════════════════════════════════════════════════════════════╣
║  QUALITY REPORT                                                  ║
╠══════════════════════════════════════════════════════════════════╣
  Score     : {best_score}/100  ({best_eval.get('verdict', 'N/A')})
  Platform  : {platform}  |  Tone: {tone}
  Attempts  : {len(history)}  |  History: {history_str}
 
  Breakdown :
{scores_display}
╚══════════════════════════════════════════════════════════════════╝
""".strip()

# ---------------------------------------------------------------------------
# 1. NEW DATA ANALYSIS TOOL (This is how Gemini/ChatGPT do it)
# ---------------------------------------------------------------------------
@tool
def run_python_data_analysis(python_code: str) -> str:
    """
    Executes Python code to analyze tabular data (CSVs/Excel). 
    pandas is available as pd. 
    ALWAYS use print() to output your final answer!
    Example code to generate:
    import pandas as pd
    df = pd.read_csv('/kaggle/input/data.csv')
    print(df[df['inventory'] == 0]['title'].tolist())
    """
    # Capture standard output so the LLM can read the results of its print() statements
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    try:
        # Execute the AI's generated code safely
        exec(python_code, {"pd": pd})
        output = redirected_output.getvalue()
        return output if output else "Code executed successfully but printed nothing. Please use print() to output results."
    except Exception as e:
        return f"Python Error: {e}\nPlease fix the code and try again."
    finally:
        sys.stdout = old_stdout
        
@tool
def read_full_document(file_path: str) -> str:
    """
    Reads the ENTIRE exact text of a PDF, Word Doc, or Text file. 
    Use this when the user asks you to 'read', 'summarize', or extract exact details from a whole file, 
    rather than just searching it.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    try:
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                # Truncate to ~25,000 words to prevent context window crash
                return text[:100000] + "\n...[TRUNCATED]" if len(text) > 100000 else text
        else:
            md = MarkItDown()
            text = md.convert(file_path).text_content
            return text[:100000] + "\n...[TRUNCATED]" if len(text) > 100000 else text
    except Exception as e:
        return f"Failed to read document: {str(e)}"


@tool
def analyze_image(image_path: str, specific_question: str = "Extract all text and describe the image accurately.") -> str:
    """
    Use this tool to read, parse, and extract 100% accurate data from Images (JPG, PNG).
    Provide the path to the image, and a specific question of what to extract.
    """
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"
    
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        ext = os.path.splitext(image_path)[-1].lower().replace(".", "")
        ext = "jpeg" if ext == "jpg" else ext
        
        msg = HumanMessage(content=[
            {"type": "text", "text": specific_question},
            {"type": "image_url", "image_url": f"data:image/{ext};base64,{image_data}"},
        ])
        
        # Invoke the Vision LLM specifically
        response = llm_vision.invoke([msg])
        return response.content
    except Exception as e:
        return f"Failed to analyze image: {str(e)}"

# pip install deep-translator
from deep_translator import GoogleTranslator as DeepGoogleTranslator

@tool  
def translate_to_bengali(text: str) -> str:
    """
    Translates ANY English text (words, sentences, paragraphs, full essays) 
    into perfect Bengali. Use whenever user wants English → Bengali translation.
    """
    if not text.strip():
        return "Error: No text provided."
    
    try:
        translator = DeepGoogleTranslator(source='en', target='bn')
        
        # deep_translator handles up to 5000 chars per call
        MAX_CHUNK = 4999
        
        if len(text) <= MAX_CHUNK:
            return translator.translate(text)
        
        # For long essays: chunk by paragraphs first, then by size
        paragraphs = text.split('\n')
        chunks, current = [], ""
        
        for para in paragraphs:
            if len(current) + len(para) + 1 <= MAX_CHUNK:
                current += para + "\n"
            else:
                if current:
                    chunks.append(current.strip())
                # If single paragraph is too long, split by sentences
                if len(para) > MAX_CHUNK:
                    for i in range(0, len(para), MAX_CHUNK):
                        chunks.append(para[i:i+MAX_CHUNK])
                else:
                    current = para + "\n"
        
        if current:
            chunks.append(current.strip())
        
        translated = [translator.translate(chunk) for chunk in chunks]
        return "\n".join(translated)

    except Exception as e:
        return f"Translation error: {e}\nPlease run: pip install deep-translator"


def ingest_files(file_list: List[dict], thread_id: str) -> List[dict]:
    all_chunks = []
    summaries =[]
    md_converter = MarkItDown()

    for file_info in file_list:
        raw_bytes: bytes = file_info["bytes"]
        filename: str = file_info.get("name", "unknown")
        ext = os.path.splitext(filename)[-1].lower()


        if ext not in[".csv", ".xlsx", ".xls"]:
            try:
                if ext == ".pdf":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(raw_bytes)
                        tmp_path = tmp.name
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    os.unlink(tmp_path)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.split_documents(docs)
                    all_chunks.extend(chunks)
                    summaries.append({"filename": filename, "pages": len(docs), "chunks": len(chunks)})
    
                elif ext in (".xlsx", ".xls", ".csv"):
                    if ext == ".csv":
                        df = pd.read_csv(io.BytesIO(raw_bytes))
                        sheets = {"Sheet1": df}
                    else:
                        sheets = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=None)
    
                    text_parts =[f"File: {filename}"]
                    for sname, df in sheets.items():
                        df = df.dropna(how="all").dropna(axis=1, how="all").fillna("")
                        text_parts.append(f"\n[Sheet: {sname}]\n{df.to_string(index=False)}")
    
                    full_text = "\n".join(text_parts)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    from langchain_core.documents import Document
                    chunks = splitter.create_documents([full_text], metadatas=[{"source": filename}])
                    all_chunks.extend(chunks)
                    summaries.append({"filename": filename, "sheets": len(sheets), "chunks": len(chunks)})
    
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(raw_bytes)
                        tmp_path = tmp.name
                    result = md_converter.convert(tmp_path)
                    os.unlink(tmp_path)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    from langchain_core.documents import Document
                    chunks = splitter.create_documents([result.text_content], metadatas=[{"source": filename}])
                    all_chunks.extend(chunks)
                    summaries.append({"filename": filename, "chunks": len(chunks)})
    
            except Exception as e:
                logger.error(f"Failed to ingest {filename}: {e}")
                summaries.append({"filename": filename, "error": str(e)})

    if all_chunks:
        if thread_id in _THREAD_RETRIEVERS:
            _THREAD_RETRIEVERS[thread_id].vectorstore.add_documents(all_chunks)
        else:
            store = FAISS.from_documents(all_chunks, embeddings)
            _THREAD_RETRIEVERS[thread_id] = store.as_retriever(search_kwargs={"k": 5})

    _THREAD_METADATA.setdefault(thread_id,[]).extend(summaries)
    return summaries

# ---------------------------------------------------------------------------
# 3. Tools
# ---------------------------------------------------------------------------
_ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)

@tool
def search_web(query: str) -> str:
    """Search the web for information. Tries DuckDuckGo first, 
    then falls back to SearXNG if DDG fails or returns no results.
    Returns titles, URLs and snippets."""
    try:
        results = _ddg_wrapper.results(query, max_results=3)
        return "\n---\n".join(f"Title: {r['title']}\nSnippet: {r['snippet']}" for r in results) if results else "No results."
    except Exception as e:
        pass

     # --- Fallback: SearXNG ---
    print("\n   [⚠️ DDG failed, trying SearXNG fallback...]", end="")
    searxng_result = _searxng_search(query)
    if searxng_result:
        return searxng_result

    # --- Both failed ---
    return (
        "ERROR: All search engines failed. "
        "Do NOT retry the search. "
        "Use your training knowledge to answer the question instead."
    )

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform arithmetic. operation must be one of: add, sub, mul, div."""
    ops = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "div": lambda a, b: a / b if b != 0 else None}
    if operation not in ops: return {"error": f"Unknown operation '{operation}'"}
    result = ops[operation](first_num, second_num)
    return {"result": result} if result is not None else {"error": "Division by zero"}

@tool
def rag_retrieve(query: str, thread_id: str) -> dict:
    """Retrieve relevant passages from documents uploaded in this chat thread. ALWAYS pass the current thread_id."""
    retriever = _get_retriever(thread_id)
    if not retriever:
        return {"error": "No documents indexed. Tell the user to upload files."}
    docs = retriever.invoke(query)
    return {"chunks": [d.page_content for d in docs]}

TOOLS =[search_web, calculator, rag_retrieve,run_python_data_analysis,read_full_document,analyze_image,translate_to_bengali,write_job_application_message]
llm_with_tools = llm.bind_tools(TOOLS)
llm_accuracy_with_tools = llm_accuracy.bind_tools(TOOLS)

# ---------------------------------------------------------------------------
# 4. Graph Construction
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

SYSTEM_PROMPT = """You are an advanced AI assistant running inside a Kaggle notebook.
You have access to powerful tools. Follow these rules strictly:

1. TABULAR DATA (CSV/Excel): ALWAYS use `run_python_data_analysis`. Write a pandas script using the exact filepaths provided below. Use print().
2. FINANCIAL DATA / SPECIFIC FACTS FROM DOCS: ALWAYS use rag_retrieve first.
   This includes: income, revenue, expenses, totals, dates, figures, 
   or any specific number from an uploaded document.
   Only use read_full_document if rag_retrieve returns no results.
3. READING PDFS/DOCS: To read, summarize, or extract full context from a document, use `read_full_document` using the exact filepath provided.
4. IMAGES (JPG/PNG): ALWAYS use `analyze_image` to look at the image and answer the user's question accurately.
5. LARGE DOC SEARCH: If a document is hundreds of pages long and you just need a specific fact, use `rag_retrieve`.
6. TRANSLATION (English → Bengali): ALWAYS use `translate_to_bengali`. 
   This works for single words, sentences, paragraphs, and full essays with 100% accuracy.
   NEVER translate manually - always use this tool.
7. NEVER GUESS file contents. Always use a tool to open and read them.
8. JOB APPLICATIONS: Use `write_job_application_message`.
        - resume_path : file path to the user's resume PDF (from active files)
        - jd_text     : the job description the user pasted in the chat
        - platform    : LinkedIn | Email | WhatsApp  (ask if unsure)
        - tone        : confident | humble | enthusiastic (default: confident)

CRITICAL SEARCH RULES:
- Call `search_web` a MAXIMUM of 3 times per response. After 3 searches, STOP and write your answer using what you have.
- If search returns an ERROR message, do NOT search again. Use your training knowledge immediately.
- Never search for the same query twice.

Active File Paths available to you:
{active_files}
"""

def chat_node(state: ChatState, config=None):
    thread_id = config.get("configurable", {}).get("thread_id", "") if config else ""
    active_files = config.get("configurable", {}).get("active_files", "No files provided.")
    system = SystemMessage(content=SYSTEM_PROMPT + f"\n\nCurrent thread_id: `{thread_id}` \nUploaded File Paths (Use these in your pd.read_csv code):\n{active_files}")
    if active_files:
        model = llm_accuracy_with_tools
    else:
        model = llm_with_tools
    response = model.invoke([system] + state["messages"], config=config)
    return {"messages": [response]}

memory_checkpointer = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=memory_checkpointer)

# ---------------------------------------------------------------------------
# 5. Kaggle Interactivity Helpers
# ---------------------------------------------------------------------------
def ingest_local_files(file_paths: List[str], thread_id: str):
    file_list =[]
    for path in file_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                file_list.append({"bytes": f.read(), "name": os.path.basename(path)})
        else:
            print(f"❌ File not found: {path}")
            
    if file_list:
        print(f"📥 Extracting and Indexing {len(file_list)} files via nomic-embed-text...")
        summaries = ingest_files(file_list, thread_id)
        for s in summaries: print(f"✅ Indexed: {s}")

def ask_agent(query: str, thread_id: str = "kaggle_chat_1", files: List[str] = None):
    if files: ingest_local_files(files, thread_id)

    file_paths_str = "\n".join(files) if files else "None"
        
    config = {"configurable": 
              {"thread_id": thread_id,"active_files": file_paths_str},
              "recursion_limit": 25 
             }
    print(f"\n🧑‍💻 User: {query}\n")
    print("🤖 Agent: ", end="")
    
    for msg_chunk, metadata in chatbot.stream({"messages":[HumanMessage(content=query)]}, config=config, stream_mode="messages"):
        if isinstance(msg_chunk, ToolMessage):
            print(f"\n   [✅ Tool finished: {msg_chunk.name}]\n🤖 Agent: ", end="")
        elif hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls:
            for tc in msg_chunk.tool_calls:
                print(f"\n   [🛠️ Using Tool: {tc['name']}]", end="")
        # elif isinstance(msg_chunk, AIMessage) and msg_chunk.content:
        #     print(msg_chunk.content, end="", flush=True)
        # FIX: Check content exists and the message is NOT a tool-call-only chunk
        elif hasattr(msg_chunk, "content") and msg_chunk.content:
            if not (hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls):
                print(msg_chunk.content, end="", flush=True)
            
            
    print("\n\n" + "-"*60)
