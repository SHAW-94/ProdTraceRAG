# app/api.py
from __future__ import annotations

from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import os, time, uuid, glob, json, re
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from rank_bm25 import BM25Okapi
import jieba

from app.llm_gen import llm_generate
from app.security import (
    make_fastapi_app,
    require_public,
    require_admin,
    maybe_require_admin_scope,
    build_config_response,
    sanitize_trace_on_write,
    sanitize_trace_item,
    normalize_and_validate_ingest_folder,
    validate_glob_pattern,
    path_ref,
    public_citation_snippet,
    chunk_payload,
)


app = make_fastapi_app(title="ProdTraceRAG")


# Paths & persistent state

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.join(PROJECT_ROOT, ".chroma"))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "prodracerag")
os.makedirs(CHROMA_DIR, exist_ok=True)

LAST_INGEST_PATH = os.path.join(CHROMA_DIR, "last_ingest.json")
TRACE_LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
TRACE_LOG_PATH = os.path.join(TRACE_LOG_DIR, "ask_traces.jsonl")
os.makedirs(TRACE_LOG_DIR, exist_ok=True)


# Runtime config (tunable)

def _envf(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _envi(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _envb(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")



# LLM circuit breaker + error surfacing policy

LLM_BREAKER: Dict[str, Any] = {
    "open_until": 0.0,
    "consecutive_errors": 0,
    "trips": 0,
    "last_error": "",
}


def _breaker_enabled() -> bool:
    return _envb("LLM_BREAKER_ENABLED", True)


def _breaker_trip_errors() -> int:
    return _envi("LLM_BREAKER_TRIP_ERRORS", 2)


def _breaker_cooldown_s() -> float:
    return _envf("LLM_BREAKER_COOLDOWN_S", 600.0)


def _surface_llm_error_in_stats(debug_flag: bool) -> bool:

    if _envb("SURFACE_LLM_ERROR", False):
        return True
    return bool(debug_flag)


def _breaker_is_open() -> bool:
    try:
        return _breaker_enabled() and time.time() < float(LLM_BREAKER.get("open_until", 0.0))
    except Exception:
        return False


def _breaker_record_success() -> None:
    if not _breaker_enabled():
        return
    LLM_BREAKER["consecutive_errors"] = 0
    LLM_BREAKER["last_error"] = ""


def _breaker_record_error(err: str) -> None:
    if not _breaker_enabled():
        return
    e = (err or "").lower()
    hard = (
        "http_429" in e
        or "rate limit" in e
        or "exceeded your current quota" in e
        or "insufficient_quota" in e
        or "quota" in e
        or "overloaded" in e
    )
    if not hard:
        return

    LLM_BREAKER["consecutive_errors"] = int(LLM_BREAKER.get("consecutive_errors", 0)) + 1
    LLM_BREAKER["last_error"] = (err or "")[:240]
    if int(LLM_BREAKER["consecutive_errors"]) >= _breaker_trip_errors():
        LLM_BREAKER["open_until"] = time.time() + float(_breaker_cooldown_s())
        LLM_BREAKER["trips"] = int(LLM_BREAKER.get("trips", 0)) + 1

RUNTIME_CFG: Dict[str, Any] = {
    "alpha": _envf("ALPHA", 0.65),
    "min_evidence_score": _envf("MIN_EVIDENCE_SCORE", 0.25),

    # clarification band: if top_score in [clarify_min, clarify_max) => ask to clarify
    "clarify_max_score": _envf("CLARIFY_MAX_SCORE", 0.60),

    # strict citation alignment guardrail
    "strict_citation": _envb("CITATION_STRICT", True),

    # retrieval features toggles
    "enable_bm25": _envb("ENABLE_BM25", True),
    "enable_expansion": _envb("ENABLE_EXPANSION", True),
    "enable_diversity": _envb("ENABLE_DIVERSITY", True),

    # expansion + diversity params
    "query_expansions": _envi("QUERY_EXPANSIONS", 1),
    "diversity_lambda": _envf("DIVERSITY_LAMBDA", 0.25),

    # injection chunk downweight
    "injection_penalty": _envf("INJECTION_PENALTY", 0.25),

    # trace
    "enable_tracing": _envb("ENABLE_TRACING", True),

    # LLM failure policy:
    # fail_open=True => if LLM fails (429/timeout/etc), fall back to extractive answer for in-domain
    "llm_fail_open": _envb("LLM_FAIL_OPEN", True),
}

# Derived: clarify_min defaults to 0.6 * clarify_max (empirically stable)
def clarify_min_score() -> float:
    return float(RUNTIME_CFG.get("clarify_max_score", 0.60)) * 0.60



# Chroma

client = chromadb.PersistentClient(path=CHROMA_DIR)
default_ef = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=default_ef,
    metadata={"hnsw:space": "cosine"},
)


# BM25 store

BM25_STORE: Dict[str, Dict[str, Any]] = {}  # chunk_id -> {"text": str, "meta": dict, "inj": bool}
BM25_IDS: List[str] = []
BM25_MODEL: Optional[BM25Okapi] = None

_cjk_re = re.compile(r"[\u4e00-\u9fff]")

def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    if _cjk_re.search(text):
        return [t.strip() for t in jieba.lcut(text) if t.strip()]
    return re.findall(r"[a-z0-9_]+", text)

def rebuild_bm25():
    global BM25_IDS, BM25_MODEL
    BM25_IDS = list(BM25_STORE.keys())
    if not BM25_IDS:
        BM25_MODEL = None
        return
    tokenized = [tokenize(BM25_STORE[cid]["text"]) for cid in BM25_IDS]
    BM25_MODEL = BM25Okapi(tokenized)


# Last ingest persistence

def _load_last_ingest() -> Dict[str, Any]:
    try:
        if os.path.exists(LAST_INGEST_PATH):
            with open(LAST_INGEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}

def _save_last_ingest(d: Dict[str, Any]) -> None:
    try:
        with open(LAST_INGEST_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

LAST_INGEST: Dict[str, Any] = _load_last_ingest()


# Prompt-injection detection

INJ_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"system\s+prompt",
    r"developer\s+message",
    r"print\s+all\s+documents",
    r"dump\s+all\s+documents",
    r"reveal\s+the\s+system",
    r"exfiltrat",
    r"password",
    r"api[_\-\s]?key",
    r"ssh\s+key",
r"你是chatgpt",
    r"系统提示词",
    r"忽略.*(规则|指令|以上)",
    r"无视.*(规则|指令|以上)",
    r"把所有文档.*(输出|打印|发给我)",
]
INJ_RE = re.compile("|".join(f"(?:{p})" for p in INJ_PATTERNS), re.IGNORECASE)

SENSITIVE_EXFIL_PATTERNS = [
    r"root\s+密码",
    r"数据库\s*root\s*密码",
    r"内网资产清单",
    r"公司.*收入",
    r"OKR",
]
SENSITIVE_RE = re.compile("|".join(f"(?:{p})" for p in SENSITIVE_EXFIL_PATTERNS), re.IGNORECASE)


def extract_question_after_injection(q: str) -> str | None:
    """
    If the query contains prompt-injection preamble, try to salvage the real question.
    Examples:
      - "...然后回答：<REAL_Q>"
      - "...Then answer: <REAL_Q>"
    Return None if cannot extract.
    """
    q0 = (q or "").strip()
    if not q0:
        return None

    # Prefer the last explicit "answer:" segment
    markers = [
        "然后回答：", "然后回答:", "再回答：", "再回答:", "回答：", "回答:",
        "then answer:", "then answer：", "answer:", "answer：",
    ]
    qlow = q0.lower()
    best = None
    for m in markers:
        idx = qlow.rfind(m.lower())
        if idx != -1:
            cand = q0[idx + len(m):].strip(" \t\r\n-—:：;；")
            if len(cand) >= 3:
                best = cand
    return best

def is_injection_query(q: str) -> bool:
    return bool(INJ_RE.search(q or ""))

def is_sensitive_exfil(q: str) -> bool:
    return bool(SENSITIVE_RE.search(q or ""))

def is_injection_text(t: str) -> bool:
    return bool(INJ_RE.search(t or ""))

def extract_ascii_entities(q: str) -> List[str]:

    q = q or ""
    toks = re.findall(r"\b[A-Za-z][A-Za-z0-9]{2,}\b", q)
    # Keep CamelCase or mixed-case or ALLCAPS or contains digits
    out = []
    for t in toks:
        if any(c.isdigit() for c in t) or (t != t.lower()) or t.isupper():
            out.append(t)
    # de-dup while preserving order
    seen = set()
    res = []
    for t in out:
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        res.append(t)
    return res

def scope_guard_out_of_domain(q: str, citations: List["Citation"]) -> Optional[str]:

    q = q or ""
    if is_injection_query(q):
        return "prompt_injection_detected"
    if is_sensitive_exfil(q):
        return "sensitive_request_out_of_scope"

    entities = extract_ascii_entities(q)
    if not entities:
        return None

    hay = " ".join(
        [c.doc_id + " " + c.title_path + " " + c.snippet for c in citations[:5]]
    ).lower()

    missing = [e for e in entities if e.lower() not in hay]
    if missing and len(missing) >= 1:
        return f"out_of_scope_entity:{missing[0]}"
    return None



# Schemas

class IngestLocalReq(BaseModel):
    folder_path: str = Field(..., alias="folder", description="Local folder path containing .md/.txt files")
    glob_pattern: str = Field("**/*.md", description="glob pattern, e.g. **/*.md or **/*.txt")
    chunk_chars: int = Field(800, ge=100, le=4000, description="approx chunk size in characters")
    overlap_chars: int = Field(133, ge=0, le=1000, description="overlap between chunks")

    class Config:
        allow_population_by_field_name = True

class SetConfigReq(BaseModel):
    alpha: Optional[float] = None
    clarify_min_score: Optional[float] = None  # accepted for compatibility; derived at runtime
    min_evidence_score: Optional[float] = None
    clarify_max_score: Optional[float] = None
    strict_citation: Optional[bool] = None
    enable_bm25: Optional[bool] = None
    enable_expansion: Optional[bool] = None
    enable_diversity: Optional[bool] = None
    query_expansions: Optional[int] = None
    diversity_lambda: Optional[float] = None
    injection_penalty: Optional[float] = None
    enable_tracing: Optional[bool] = None
    llm_fail_open: Optional[bool] = None

class AskReq(BaseModel):
    q: str = Field(..., min_length=1, max_length=4000)
    topk: int = Field(8, ge=1, le=20)
    min_score: float = Field(0.25, ge=0.0, le=1.0)
    debug: bool = False

class Citation(BaseModel):
    doc_id: str
    title_path: str
    source: str
    source_ref: Optional[str] = None
    updated_at: str
    chunk_id: str
    snippet: str
    score: float

class AskResp(BaseModel):
    trace_id: str
    answer: str
    citations: List[Citation]
    refused: bool
    refusal_reason: Optional[str] = None
    stats: Dict[str, Any] = {}


# Utils

def read_text_file(path: str) -> str:
    max_chars = _envi("INGEST_MAX_CHARS_PER_FILE", 400000)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read(max_chars + 1)
    if len(data) > max_chars:
        data = data[:max_chars]
    return data

def simple_chunk(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    text = (text or "").replace("\r\n", "\n")
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap_chars)
    return chunks

def get_updated_at(path: str) -> str:
    try:
        ts = os.path.getmtime(path)
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return "unknown"

def _write_trace(item: Dict[str, Any]) -> None:
    if not RUNTIME_CFG.get("enable_tracing", True):
        return
    try:
        safe_item = sanitize_trace_on_write(item, project_root=PROJECT_ROOT)
        with open(TRACE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_item, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _read_recent_traces(limit: int = 10) -> List[Dict[str, Any]]:
    if not os.path.exists(TRACE_LOG_PATH):
        return []
    try:
        with open(TRACE_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        items = []
        for ln in lines[-limit:]:
            ln = ln.strip()
            if not ln:
                continue
            try:
                items.append(json.loads(ln))
            except Exception:
                continue
        return items[::-1]
    except Exception:
        return []


# Query expansion & diversity

def expand_queries(q: str, n: int) -> List[str]:
    q = (q or "").strip()
    if n <= 0:
        return [q]
    expansions = [q]

    ql = q.lower()
    hints = []
    if any(k in q for k in ["限流", "rate", "rps", "burst", "429"]):
        hints.append("rate limit rps burst 429 RATE_LIMITED")
    if "502" in q or "风暴" in q or "retry" in ql:
        hints.append("502 error storm retry storm upstream timeout backoff jitter")
    if "延迟" in q or "latency" in ql or "p95" in ql:
        hints.append("p95 latency spike runbook dependency cpu memory saturation")
    if "OOM" in q or "CrashLoop" in q or "oom" in ql:
        hints.append("oomkilled crashloopbackoff memory limit rss allocator")
    if any(k in q for k in ["API", "接口", "错误码", "timeout", "鉴权", "auth"]):
        hints.append("API Spec errors timeout auth")

    for h in hints[:n]:
        expansions.append(f"{q} {h}")
    return expansions[: (1 + max(0, n))]

def diversify(cands: List[Tuple[float, str, Dict[str, Any]]], lam: float, k: int) -> List[Tuple[float, str, Dict[str, Any]]]:

    if k <= 0 or not cands:
        return []
    selected = []
    seen_docs = set()
    for score, cid, it in cands:
        doc_id = (it.get("meta") or {}).get("doc_id", "")
        adj = score
        if doc_id and doc_id in seen_docs:
            adj = score * (1.0 - max(0.0, min(0.9, lam)))
        selected.append((adj, cid, it, score))
    selected.sort(reverse=True, key=lambda x: x[0])

    out = []
    for adj, cid, it, raw in selected:
        doc_id = (it.get("meta") or {}).get("doc_id", "")
        if doc_id:
            seen_docs.add(doc_id)
        out.append((raw, cid, it))
        if len(out) >= k:
            break
    return out



# Retrieval intent helpers

_intent_key_re = re.compile(r"/([A-Za-z][A-Za-z0-9_]{2,})")
_word_key_re = re.compile(r"\b([A-Za-z][A-Za-z0-9_]{2,})\b")

_INTENT_STOP = {
    "http", "https", "api", "spec", "get", "post", "put", "delete", "patch",
    "rps", "qps", "rpm", "mb", "kb", "ms", "s", "sec", "timeout", "auth",
    "rate", "limit", "limits", "errors", "error",
}


def extract_intent_key(q: str) -> Optional[str]:
    q = (q or "").strip()
    if not q:
        return None

    m = _intent_key_re.search(q)
    if m:
        return m.group(1)

    # Word tokens (ASCII-ish) — prefer tokens that look like endpoint names.
    toks = []
    for m2 in _word_key_re.finditer(q):
        t = m2.group(1)
        tl = t.lower()
        if tl in _INTENT_STOP:
            continue
        # Prefer tokens containing uppercase (camelCase/PascalCase) or underscore
        score = 0
        if any(ch.isupper() for ch in t):
            score += 2
        if "_" in t:
            score += 1
        if t[0].isalpha():
            score += 1
        toks.append((score, t))

    if not toks:
        return None
    toks.sort(reverse=True, key=lambda x: (x[0], len(x[1])))
    return toks[0][1] if toks[0][0] > 0 else None


def citation_matches_intent(c: "Citation", key: str) -> bool:
    if not c or not key:
        return False
    kl = key.lower()
    if kl in (c.doc_id or "").lower():
        return True
    if kl in (c.title_path or "").lower():
        return True
    sn = (c.snippet or "").lower()
    if f"/{kl}" in sn:
        return True
    # Conservative: only match whole-ish word in snippet
    if re.search(rf"\b{re.escape(kl)}\b", sn):
        return True
    return False


def intent_doc_bonus(key: Optional[str], meta: Dict[str, Any], doc: str) -> float:

    if not key:
        return 0.0
    kl = key.lower()
    doc_id = str((meta or {}).get("doc_id") or "").lower()
    title = str((meta or {}).get("title_path") or "").lower()
    doc_l = (doc or "").lower()

    if kl and (kl in doc_id or kl in title):
        return 0.35
    if f"/{kl}" in doc_l:
        return 0.25
    if re.search(rf"\b{re.escape(kl)}\b", doc_l):
        return 0.15
    return 0.0


# API endpoints

@app.get("/")
def root(request: Request):
    return {
        "ok": True,
        "message": "ProdTraceRAG is running.",
        "docs": "/docs",
        "auth_required": bool(os.getenv("AUTH_REQUIRED", "").strip().lower() in ("1","true","yes","on")),
    }

@app.get("/config")
def config(request: Request, scope: str = "public"):
    llm_provider = os.getenv("LLM_PROVIDER", "none").lower().strip()
    llm_model = os.getenv("OPENAI_MODEL") or os.getenv("QWEN_MODEL") or os.getenv("DEEPSEEK_MODEL") or ""
    admin = maybe_require_admin_scope(request, scope=scope)
    return build_config_response(
        runtime_cfg=RUNTIME_CFG,
        clarify_min_score=clarify_min_score(),
        chroma_dir=CHROMA_DIR,
        collection=COLLECTION_NAME,
        bm25_ready=BM25_MODEL is not None,
        bm25_chunks=len(BM25_IDS),
        last_ingest=LAST_INGEST,
        llm_provider=llm_provider,
        llm_model=llm_model,
        project_root=PROJECT_ROOT,
        admin=admin,
    )

@app.post("/set_config", dependencies=[Depends(require_admin)])
def set_config(req: SetConfigReq):
    for k, v in req.dict(exclude_none=True).items():
        if k == "clarify_min_score":
            continue
        RUNTIME_CFG[k] = v
    return {"ok": True, **RUNTIME_CFG, "clarify_min_score": clarify_min_score()}

@app.post("/reset_index", dependencies=[Depends(require_admin)])
def reset_index():
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    global collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=default_ef,
        metadata={"hnsw:space": "cosine"},
    )

    BM25_STORE.clear()
    rebuild_bm25()

    LAST_INGEST.clear()

    try:
        if os.path.exists(TRACE_LOG_PATH):
            os.remove(TRACE_LOG_PATH)
    except Exception:
        pass

    return {"ok": True, "message": "collection reset"}

@app.post("/ingest_local", dependencies=[Depends(require_admin)])
def ingest_local(req: IngestLocalReq):
    t0 = time.time()
    folder = normalize_and_validate_ingest_folder(req.folder_path, PROJECT_ROOT)
    glob_pattern = validate_glob_pattern(req.glob_pattern)
    pattern = os.path.join(folder, glob_pattern)
    files = glob.glob(pattern, recursive=True)

    added = 0
    real_files = 0
    inj_chunks = 0

    for fp in files:
        if not os.path.isfile(fp):
            continue
        ext = os.path.splitext(fp)[1].lower()
        if ext not in [".md", ".txt"]:
            continue

        real_files += 1
        text = read_text_file(fp)
        chunks = simple_chunk(text, req.chunk_chars, req.overlap_chars)

        doc_id = os.path.relpath(fp, folder)
        source = path_ref(fp, PROJECT_ROOT, ingest_root=folder)
        updated_at = get_updated_at(fp)
        title_path = path_ref(fp, PROJECT_ROOT, ingest_root=folder)

        ids, docs, metas = [], [], []
        for i, ch in enumerate(chunks):
            chunk_id = f"{doc_id}::chunk{i}"
            is_inj = is_injection_text(ch)
            if is_inj:
                inj_chunks += 1

            ids.append(chunk_id)
            docs.append(ch)
            metas.append({
                "doc_id": doc_id,
                "title_path": title_path,
                "source": source,
                "source_ref": source,
                "updated_at": updated_at,
                "chunk_id": chunk_id,
                "is_injection": bool(is_inj),
            })

        if ids:
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            added += len(ids)

            for ch, meta in zip(docs, metas):
                cid = meta.get("chunk_id")
                if cid:
                    BM25_STORE[cid] = {"text": ch, "meta": meta, "inj": bool(meta.get("is_injection", False))}

    rebuild_bm25()

    LAST_INGEST.clear()
    LAST_INGEST.update({
        "folder": path_ref(folder, PROJECT_ROOT),
        "folder_ref": path_ref(folder, PROJECT_ROOT),
        "glob_pattern": glob_pattern,
        "chunk_chars": req.chunk_chars,
        "overlap_chars": req.overlap_chars,
        "files": real_files,
        "chunks_added": added,
        "inj_chunks": inj_chunks,
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    cost = time.time() - t0
    return {"ok": True, "folder": "<redacted>", "folder_ref": path_ref(folder, PROJECT_ROOT), "files": real_files, "chunks_added": added, "inj_chunks": inj_chunks, "seconds": round(cost, 3)}

@app.get("/traces/recent", dependencies=[Depends(require_admin)])
def traces_recent(request: Request, limit: int = 10, raw: bool = False):
    limit = max(1, min(100, int(limit)))
    items = [sanitize_trace_item(it, project_root=PROJECT_ROOT, admin=True, raw=raw) for it in _read_recent_traces(limit=limit)]
    return {"ok": True, "items": items}


@app.get("/traces/get", dependencies=[Depends(require_admin)])
def traces_get(request: Request, trace_id: str, raw: bool = False):
    if not trace_id:
        return {"ok": False, "error": "missing_trace_id"}
    if not os.path.exists(TRACE_LOG_PATH):
        return {"ok": True, "item": None}
    try:
        with open(TRACE_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        # scan backwards (recent first)
        for ln in reversed(lines[-5000:]):
            ln = ln.strip()
            if not ln:
                continue
            try:
                it = json.loads(ln)
            except Exception:
                continue
            if it.get("trace_id") == trace_id:
                return {"ok": True, "item": sanitize_trace_item(it, project_root=PROJECT_ROOT, admin=True, raw=raw)}
        return {"ok": True, "item": None}
    except Exception as e:
        return {"ok": False, "error": f"read_failed:{type(e).__name__}:{str(e)[:120]}"}

@app.get("/chunk", dependencies=[Depends(require_admin)])
def chunk_get(chunk_id: str):
    if not chunk_id:
        return {"ok": False, "error": "missing_chunk_id"}
    it = BM25_STORE.get(chunk_id)
    if not it:
        return {"ok": False, "error": "chunk_not_found"}
    return chunk_payload(chunk_id, it, project_root=PROJECT_ROOT)


# Generation helpers

def rule_based_answer(q: str, citations: List[Citation]) -> str:
    q = (q or "").strip()
    ql = q.lower()

    if citations:
        top = citations[0].snippet
        lines = [ln.strip() for ln in top.splitlines() if ln.strip()]
        if lines:
            return "\n".join(lines[:4]) + "\n[1]"

    if "限流" in q or "rate" in ql:
        return "请参考接口文档中的 Rate Limiting 段落（包含默认值、burst、超限返回码）。[1]"
    if "502" in q or "风暴" in q:
        return "请参考事故复盘中的 Root Cause/ Fix 段落（包含根因与修复动作）。[1]"
    return "根据证据：先确认影响范围与时间窗口，再按 runbook 检查依赖健康、资源饱和、限流/重试与变更回滚路径。[1]"

def _md_sections(md: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    cur = ""
    for raw in (md or "").splitlines():
        ln = raw.rstrip()
        if not ln.strip():
            continue
        if ln.lstrip().startswith("#"):
            cur = ln.lstrip("#").strip() or "(untitled)"
            out.setdefault(cur, [])
            continue
        out.setdefault(cur, []).append(ln.strip("- ").strip())
    return out


def _pick_lines(secs: Dict[str, List[str]], prefer_titles: List[str], max_lines: int = 3) -> List[str]:
    titles = list(secs.keys())
    # exact/substring match first
    for pref in prefer_titles:
        for t in titles:
            if pref.lower() in (t or "").lower():
                lines = [x for x in secs.get(t, []) if len(x) >= 4]
                return lines[:max_lines]
    # fallback: take from the biggest section
    best = ""
    best_n = 0
    for t, ls in secs.items():
        n = len(ls)
        if n > best_n:
            best_n, best = n, t
    if best:
        lines = [x for x in secs.get(best, []) if len(x) >= 4]
        return lines[:max_lines]
    return []



def build_clarify_answer(q: str, citations: List["Citation"], intent_key: Optional[str] = None) -> str:
    if not citations:
        return "我没有找到可靠证据。你能补充更多上下文吗？"

    top = citations[0]
    secs = _md_sections(top.snippet or "")
    ql = (q or "").lower()

    want_payload = any(k in q for k in ["文件", "大小", "最大", "payload", "limit", "限制", "上传"]) or "mb" in ql
    want_limit = any(k in q for k in ["限流", "rate", "rps", "burst", "429"])

    hint_lines = []
    if want_payload:
        hint_lines = _pick_lines(secs, ["Limits", "限制"], 2)
    elif want_limit:
        hint_lines = _pick_lines(secs, ["Rate Limiting", "限流"], 2)

    hint = ("；".join(hint_lines) + " [1]") if hint_lines else ""

    qs = []
    if intent_key:
        qs.append(f"你确认问题指的是 `{intent_key}` 这个接口/实体吗？")
    if want_payload:
        qs.append("这里的限制是指**单文件**还是**整个请求(payload)** 的总大小？")
    if want_limit:
        qs.append("你关心的是**每租户(tenant)** 还是**全局**的限流？")
    if not qs:
        qs.append("你能补充一下你关注的具体场景/参数吗？")

    return (f"我找到了部分相关证据：{hint}\n" if hint else "我找到了部分相关证据。\n") + "为了给出可靠结论，请确认：\n- " + "\n- ".join(qs[:2])


def evidence_extractive_answer(q: str, citations: List[Citation]) -> Tuple[str, List[int]]:
    if not citations:
        return "证据不足，无法可靠回答。", []

    q = (q or "").strip()
    ql = q.lower()

    # Use up to 2 strongest citations to keep answer focused.
    picked = citations[:2]
    used_idx: List[int] = [0]

    # Build sections per citation
    secs = [_md_sections(c.snippet) for c in picked]

    # Decide what to extract based on intent keywords
    want_rca = any(k in q for k in ["根因", "原因", "Root Cause"]) or "root" in ql
    want_mitig = any(k in q for k in ["止血", "缓解", "Mitigation", "回滚"]) or "mitig" in ql
    want_limit = any(k in q for k in ["限流", "rate", "rps", "burst", "429"])
    want_payload = any(k in q for k in ["文件", "大小", "最大", "payload", "limit", "限制", "上传"]) or "mb" in ql
    want_timeout = any(k in q for k in ["超时", "timeout", "p95", "延迟"])
    want_errors = any(k in q for k in ["错误", "错误码", "Errors", "status"])
    want_auth = any(k in q for k in ["鉴权", "auth", "token", "授权"])

    bullets: List[str] = []

    # API spec style
    if want_payload:
        ln = _pick_lines(secs[0], ["Limits", "限制"], 2)
        if ln:
            bullets.append("- 大小/限制：" + "；".join(ln) + " [1]")

    if want_limit:
        ln = _pick_lines(secs[0], ["Rate Limiting", "限流"], 3)
        if ln:
            bullets.append("- 限流：" + "；".join(ln) + " [1]")
    if want_timeout:
        ln = _pick_lines(secs[0], ["Timeout", "超时"], 2)
        if ln:
            bullets.append("- 超时：" + "；".join(ln) + " [1]")
    if want_auth:
        ln = _pick_lines(secs[0], ["Auth", "鉴权"], 2)
        if ln:
            bullets.append("- 鉴权：" + "；".join(ln) + " [1]")
    if want_errors:
        ln = _pick_lines(secs[0], ["Errors", "错误"], 4)
        if ln:
            bullets.append("- 错误码：" + "；".join(ln) + " [1]")

    # Postmortem / runbook style
    if want_rca:
        ln = _pick_lines(secs[0], ["Root Cause", "根因"], 3)
        if ln:
            bullets.append("- 根因：" + "；".join(ln) + " [1]")
    if want_mitig:
        ln1 = _pick_lines(secs[0], ["Mitigation", "Fix", "缓解", "止血"], 3)
        if ln1:
            bullets.append("- 止血/修复：" + "；".join(ln1) + " [1]")
        if len(secs) > 1:
            ln2 = _pick_lines(secs[1], ["Mitigation", "Fix", "First checks", "检查"], 3)
            if ln2:
                bullets.append("- 运行手册：" + "；".join(ln2) + " [2]")
                if 1 not in used_idx:
                    used_idx.append(1)

    if not bullets:
        # Generic extractive: take 2-3 informative lines from each snippet.
        parts = []
        for i, c in enumerate(picked, start=1):
            lines = [ln.strip("- ").strip() for ln in (c.snippet or "").splitlines() if ln.strip()]
            lines = [ln for ln in lines if len(ln) >= 4][:3]
            if lines:
                parts.append(f"[{i}] " + "；".join(lines))
        if parts:
            return "\n".join(parts), list(range(0, len(picked)))
        return rule_based_answer(q, citations), [0] if citations else []

    used_idx = sorted(set(used_idx))
    return "\n".join(bullets), used_idx


# General (non-RAG) answering

_math_allowed = re.compile(r"^[0-9\.\s\+\-\*\/\(\)%]+$")

def is_general_math_query(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    ql = q.lower()
    if "等于几" in q or "多少" in q and any(op in q for op in ["+", "-", "*", "/"]):
        return True
    if _math_allowed.match(q) is not None:
        return True
    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", q) and ("等于" in q or "是多少" in q or "?" in q):
        return True
    return False

def safe_calc(q: str) -> str:

    import ast
    import operator as op

    allowed_ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("bad constant")
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    m = re.search(r"([0-9\.\s\+\-\*\/\(\)%]+)", q)
    expr = (m.group(1) if m else q).strip()
    tree = ast.parse(expr, mode="eval")
    val = _eval(tree)
    if isinstance(val, float) and abs(val - round(val)) < 1e-12:
        val = int(round(val))
    return str(val)


# Ask endpoint

@app.post("/ask", response_model=AskResp, dependencies=[Depends(require_public)])
def ask(req: AskReq):
    trace_id = str(uuid.uuid4())
    t0 = time.time()

    llm_provider = os.getenv("LLM_PROVIDER", "none").lower().strip()

    # (0) Hard refuse: injection / sensitive exfil (stable for demos & safety)
    user_q = req.q
    q = user_q
    if is_injection_query(user_q):
        tail = extract_question_after_injection(user_q)
        if tail:
            q = tail
        else:
            total_ms = int((time.time() - t0) * 1000)
            resp = AskResp(
                trace_id=trace_id,
                answer="证据不足，无法可靠回答。",
                citations=[],
                refused=True,
                refusal_reason="prompt_injection_detected",
                stats={"mode": "refuse", "total_ms": total_ms, "llm_provider": llm_provider},
            )
            _write_trace({"trace_id": trace_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "refuse", "q": user_q, "expanded_queries": [user_q], "answer": resp.answer, "citations": [], "stats": resp.stats})
            return resp


    if is_sensitive_exfil(req.q):
        total_ms = int((time.time() - t0) * 1000)
        resp = AskResp(
            trace_id=trace_id,
            answer="证据不足，无法可靠回答。",
            citations=[],
            refused=True,
            refusal_reason="out_of_scope_sensitive_request",
            stats={"mode": "refuse", "total_ms": total_ms, "llm_provider": llm_provider},
        )
        _write_trace({"trace_id": trace_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "refuse", "q": req.q, "expanded_queries": [req.q], "answer": resp.answer, "citations": [], "stats": resp.stats})
        return resp

    # (0b) General math query (non-RAG). Deterministic and does not require citations.
    if is_general_math_query(req.q):
        try:
            ans = safe_calc(req.q)
        except Exception:
            ans = "无法解析表达式。"
        total_ms = int((time.time() - t0) * 1000)
        resp = AskResp(
            trace_id=trace_id,
            answer=ans,
            citations=[],
            refused=False,
            refusal_reason=None,
            stats={"mode": "general", "total_ms": total_ms, "llm_provider": llm_provider},
        )
        _write_trace({"trace_id": trace_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "general", "q": req.q, "expanded_queries": [req.q], "answer": resp.answer, "citations": [], "stats": resp.stats})
        return resp

    # (1) query expansion
    expanded = [q]
    if RUNTIME_CFG.get("enable_expansion", True):
        expanded = expand_queries(q, int(RUNTIME_CFG.get("query_expansions", 1)))

    # (2) retrieval: vector + (optional) BM25
    r0 = time.time()
    vec_topk = max(1, min(50, req.topk))
    vec_candidates: Dict[str, Dict[str, Any]] = {}

    for qx in expanded:
        res = collection.query(
            query_texts=[qx],
            n_results=vec_topk,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            cid = meta.get("chunk_id")
            if not cid:
                continue
            v = float(max(0.0, 1.0 - float(dist)))  # cosine distance -> similarity
            if bool(meta.get("is_injection", False)):
                v *= float(RUNTIME_CFG.get("injection_penalty", 0.25))
            if cid not in vec_candidates or v > vec_candidates[cid]["v"]:
                vec_candidates[cid] = {"meta": meta, "doc": doc, "v": v, "b": 0.0}

    bm25_hits: Dict[str, Dict[str, Any]] = {}
    if RUNTIME_CFG.get("enable_bm25", True) and BM25_MODEL is not None and BM25_IDS:
        q_tokens = tokenize(req.q)
        bm25_scores = BM25_MODEL.get_scores(q_tokens)
        max_s = float(np.max(bm25_scores)) if len(bm25_scores) else 0.0
        if max_s > 0:
            top_idx = np.argsort(bm25_scores)[::-1][:req.topk]
            for i in top_idx:
                cid = BM25_IDS[int(i)]
                it = BM25_STORE.get(cid)
                if not it:
                    continue
                b = float(bm25_scores[int(i)] / max_s)  # normalize 0~1
                meta = it["meta"]
                if bool(it.get("inj", False)):
                    b *= float(RUNTIME_CFG.get("injection_penalty", 0.25))
                bm25_hits[cid] = {"meta": meta, "doc": it["text"], "v": 0.0, "b": b}

    merged = dict(vec_candidates)
    for cid, it in bm25_hits.items():
        if cid not in merged:
            merged[cid] = it
        else:
            merged[cid]["b"] = max(merged[cid]["b"], it["b"])

    alpha = float(RUNTIME_CFG.get("alpha", 0.65))
    # Intent-aware bonus: avoid answering createOrder with listOrders evidence.
    intent_key = extract_intent_key(req.q)
    combined: List[Tuple[float, str, Dict[str, Any]]] = []
    for cid, it in merged.items():
        comb = alpha * it["v"] + (1.0 - alpha) * it["b"]
        bonus = intent_doc_bonus(intent_key, it.get("meta") or {}, it.get("doc") or "")
        it["intent_bonus"] = bonus
        comb = float(comb) + float(bonus)
        combined.append((float(comb), cid, it))
    combined.sort(reverse=True, key=lambda x: x[0])

    retrieval_ms = int((time.time() - r0) * 1000)

    if RUNTIME_CFG.get("enable_diversity", True):
        lam = float(RUNTIME_CFG.get("diversity_lambda", 0.25))
        combined = diversify(combined, lam=lam, k=req.topk)

    citations: List[Citation] = []
    for comb, cid, it in combined[:req.topk]:
        meta = it["meta"]
        doc = it["doc"] or ""
        citations.append(Citation(
            doc_id=meta.get("doc_id", "unknown"),
            title_path=meta.get("title_path", "unknown"),
            source=meta.get("source_ref", meta.get("source", "unknown")),
            source_ref=meta.get("source_ref", meta.get("source", "unknown")),
            updated_at=meta.get("updated_at", "unknown"),
            chunk_id=meta.get("chunk_id", cid),
            snippet=public_citation_snippet((doc[:320] + "…") if len(doc) > 320 else doc),
            score=float(comb),
        ))

    min_evidence_score = float(RUNTIME_CFG.get("min_evidence_score", 0.08))
    citations = [c for c in citations if c.score >= min_evidence_score]
    citations = citations[:req.topk]
    top_score = citations[0].score if citations else 0.0

    base_stats: Dict[str, Any] = {
        "retrieval_ms": retrieval_ms,
        "topk": req.topk,
        "top_score": top_score,
        "alpha": alpha,
        "min_evidence_score": min_evidence_score,
        "clarify_min": clarify_min_score(),
        "clarify_max": float(RUNTIME_CFG.get("clarify_max_score", 0.60)),
        "diversity_lambda": float(RUNTIME_CFG.get("diversity_lambda", 0.25)),
        "query_expansions": int(RUNTIME_CFG.get("query_expansions", 1)),
        "bm25_ready": BM25_MODEL is not None,
        "bm25_chunks": len(BM25_IDS),
        "llm_provider": llm_provider,
    }

    base_stats["intent_key"] = intent_key

    if req.debug:
        base_stats["debug"] = {"expanded_queries": expanded}

    if not citations:
        total_ms = int((time.time() - t0) * 1000)
        base_stats.update({"total_ms": total_ms, "mode": "refuse"})
        resp = AskResp(
            trace_id=trace_id,
            answer="证据不足，无法可靠回答。",
            citations=[],
            refused=True,
            refusal_reason="no_relevant_evidence",
            stats=base_stats,
        )
        _write_trace({"trace_id": trace_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "refuse", "q": req.q, "expanded_queries": expanded, "answer": resp.answer, "citations": [c.dict() for c in resp.citations], "stats": resp.stats})
        return resp

    scope_reason = scope_guard_out_of_domain(req.q, citations)
    if scope_reason:
        total_ms = int((time.time() - t0) * 1000)
        base_stats.update({"total_ms": total_ms, "mode": "refuse", "scope_guard": scope_reason})
        resp = AskResp(
            trace_id=trace_id,
            answer="证据不足，无法可靠回答。",
            citations=citations,
            refused=True,
            refusal_reason=scope_reason,
            stats=base_stats,
        )
        _write_trace({"trace_id": trace_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "refuse", "q": req.q, "expanded_queries": expanded, "answer": resp.answer, "citations": [c.dict() for c in resp.citations], "stats": resp.stats})
        return resp

    if top_score < float(req.min_score):
        total_ms = int((time.time() - t0) * 1000)
        base_stats.update({"total_ms": total_ms, "mode": "refuse"})
        resp = AskResp(
            trace_id=trace_id,
            answer="证据不足，无法可靠回答。",
            citations=citations,
            refused=True,
            refusal_reason=f"low_relevance(top_score={top_score:.3f} < min_score={req.min_score:.3f})",
            stats=base_stats,
        )
        _write_trace({"trace_id": trace_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "refuse", "q": req.q, "expanded_queries": expanded, "answer": resp.answer, "citations": [c.dict() for c in resp.citations], "stats": resp.stats})
        return resp

    cmin = float(base_stats["clarify_min"])
    cmax = float(base_stats["clarify_max"])

    # If the question contains an endpoint/entity key, ensure the *top* evidence matches it.
    mismatch = False
    if intent_key and citations:
        if not citation_matches_intent(citations[0], intent_key):
            matches = [c for c in citations if citation_matches_intent(c, intent_key)]
            if matches:
                others = [c for c in citations if c not in matches]
                citations = matches + others
                top_score = citations[0].score
            else:
                mismatch = True

    # Evidence gating:
    # - below cmin: refuse (too weak)
    # - [cmin, cmax): clarify
    # - >= cmax: answer
    if mismatch:
        total_ms = int((time.time() - t0) * 1000)
        base_stats.update({"total_ms": total_ms, "mode": "clarify"})
        resp = AskResp(
            trace_id=trace_id,
            answer=build_clarify_answer(req.q, citations, intent_key=intent_key),
            citations=citations,
            refused=True,
            refusal_reason=f"intent_mismatch(intent_key={intent_key})",
            stats=base_stats,
        )
        _write_trace({"trace_id": trace_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "clarify", "q": req.q, "expanded_queries": expanded, "answer": resp.answer, "citations": [c.dict() for c in resp.citations], "stats": resp.stats})
        return resp

    if top_score < cmin:
        total_ms = int((time.time() - t0) * 1000)
        base_stats.update({"total_ms": total_ms, "mode": "refuse"})
        resp = AskResp(
            trace_id=trace_id,
            answer="证据不足，无法可靠回答。",
            citations=citations,
            refused=True,
            refusal_reason=f"insufficient_evidence(top_score={top_score:.3f} < clarify_min={cmin:.3f})",
            stats=base_stats,
        )
        _write_trace({"trace_id": trace_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "refuse", "q": req.q, "expanded_queries": expanded, "answer": resp.answer, "citations": [c.dict() for c in resp.citations], "stats": resp.stats})
        return resp

    if cmin <= top_score < cmax:
        total_ms = int((time.time() - t0) * 1000)
        base_stats.update({"total_ms": total_ms, "mode": "clarify"})
        resp = AskResp(
            trace_id=trace_id,
            answer=build_clarify_answer(req.q, citations, intent_key=intent_key),
            citations=citations,
            refused=True,
            refusal_reason=f"borderline_evidence({cmin:.2f}≤top_score={top_score:.2f}<{cmax:.2f})",
            stats=base_stats,
        )
        _write_trace({"trace_id": trace_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "clarify", "q": req.q, "expanded_queries": expanded, "answer": resp.answer, "citations": [c.dict() for c in resp.citations], "stats": resp.stats})
        return resp

    evidence = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "snippet": c.snippet} for c in citations[:req.topk]]
    allowed_ids = {e["chunk_id"] for e in evidence}

    answer = ""
    gen_mode = "rule"
    refused = False
    refusal_reason = None

    if llm_provider == "none":
        # CI/no-key path: generate extractive answer and ONLY return citations that are actually used.
        ans, used = evidence_extractive_answer(req.q, citations)
        answer = ans
        citations = [citations[i] for i in used if 0 <= i < len(citations)] or citations[:2]
        gen_mode = "extractive_no_llm"
    else:
        # Circuit breaker: when tripped, skip LLM entirely to keep latency stable.
        if _breaker_is_open():
            base_stats["llm_state"] = "breaker_open"
            ans, used = evidence_extractive_answer(req.q, citations)
            answer = ans
            citations = [citations[i] for i in used if 0 <= i < len(citations)] or citations[:2]
            gen_mode = "fallback_extractive_breaker_open"
        else:
            gen = llm_generate(
                provider=llm_provider,
                question=req.q,
                evidence=evidence,
                timeout=int(os.getenv("LLM_TIMEOUT", os.getenv("LLM_TOTAL_TIMEOUT", "18"))),
            )

            if gen.get("refused", False):
                fail_open = bool(RUNTIME_CFG.get("llm_fail_open", True))
                rr = gen.get("refusal_reason") or "llm_refused"
                _breaker_record_error(rr)

                if fail_open:
                    ans, used = evidence_extractive_answer(req.q, citations)
                    answer = ans
                    citations = [citations[i] for i in used if 0 <= i < len(citations)] or citations[:2]
                    gen_mode = "fallback_extractive_on_llm_error"
                    refused = False
                    refusal_reason = None
                    if _surface_llm_error_in_stats(req.debug):
                        base_stats["llm_error"] = rr
                else:
                    total_ms = int((time.time() - t0) * 1000)
                    base_stats.update({"total_ms": total_ms, "mode": "refuse", "gen_mode": "llm_refuse"})
                    if _surface_llm_error_in_stats(req.debug):
                        base_stats["llm_error"] = rr
                    resp = AskResp(
                        trace_id=trace_id,
                        answer="证据不足，无法可靠回答。",
                        citations=citations,
                        refused=True,
                        refusal_reason=rr,
                        stats=base_stats,
                    )
                    _write_trace({"trace_id": trace_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "refuse", "q": req.q, "expanded_queries": expanded, "answer": resp.answer, "citations": [c.dict() for c in resp.citations], "stats": resp.stats})
                    return resp
            else:
                _breaker_record_success()

                raw_answer = (gen.get("answer") or "").strip()
                cited_ids = gen.get("cited_chunk_ids", []) or []
                invalid = [cid for cid in cited_ids if cid not in allowed_ids]
                strict = bool(RUNTIME_CFG.get("strict_citation", True))

                if invalid and strict:
                    if bool(RUNTIME_CFG.get("llm_fail_open", True)):
                        ans, used = evidence_extractive_answer(req.q, citations)
                        answer = ans
                        citations = [citations[i] for i in used if 0 <= i < len(citations)] or citations[:2]
                        gen_mode = "fallback_extractive_bad_citation"
                        if _surface_llm_error_in_stats(req.debug):
                            base_stats["llm_error"] = f"bad_citation_ids:{invalid[:3]}"
                    else:
                        total_ms = int((time.time() - t0) * 1000)
                        base_stats.update({"total_ms": total_ms, "mode": "refuse", "gen_mode": "llm_bad_citation", "invalid_cited_ids": invalid})
                        resp = AskResp(
                            trace_id=trace_id,
                            answer="证据不足，无法可靠回答。",
                            citations=citations,
                            refused=True,
                            refusal_reason=f"llm_bad_citation_ids:{invalid[:3]}",
                            stats=base_stats,
                        )
                        _write_trace({"trace_id": trace_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "refuse", "q": req.q, "expanded_queries": expanded, "answer": resp.answer, "citations": [c.dict() for c in resp.citations], "stats": resp.stats})
                        return resp
                else:
                    if cited_ids:
                        cited_set = set(cited_ids)
                        citations = [c for c in citations if c.chunk_id in cited_set]
                        citations = citations[:req.topk]

                    if raw_answer:
                        answer = raw_answer
                        gen_mode = "llm"
                    else:
                        ans, used = evidence_extractive_answer(req.q, citations)
                        answer = ans
                        citations = [citations[i] for i in used if 0 <= i < len(citations)] or citations[:2]
                        gen_mode = "fallback_extractive_empty_llm"

    total_ms = int((time.time() - t0) * 1000)
    base_stats.update({"total_ms": total_ms, "mode": "answer", "gen_mode": gen_mode, "strict_citation": bool(RUNTIME_CFG.get("strict_citation", True))})

    resp = AskResp(
        trace_id=trace_id,
        answer=answer,
        citations=citations[:req.topk],
        refused=refused,
        refusal_reason=refusal_reason,
        stats=base_stats,
    )

    _write_trace({"trace_id": trace_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": resp.stats.get("mode"), "q": req.q, "expanded_queries": expanded, "answer": resp.answer, "citations": [c.dict() for c in resp.citations], "stats": resp.stats})
    return resp
