# app/security.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import copy
import ipaddress
import os
import re

from fastapi import FastAPI, HTTPException, Request, status


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v)


def _envb(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _envlist(name: str) -> List[str]:
    raw = _env(name, "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def make_fastapi_app(title: str) -> FastAPI:
    enable_docs = _envb("ENABLE_DOCS", True)
    docs_url = "/docs" if enable_docs else None
    redoc_url = "/redoc" if (_envb("ENABLE_REDOC", False) and enable_docs) else None
    openapi_url = "/openapi.json" if enable_docs else None
    return FastAPI(title=title, docs_url=docs_url, redoc_url=redoc_url, openapi_url=openapi_url)


def _auth_required() -> bool:
    return _envb("AUTH_REQUIRED", False)


def _public_token() -> str:
    return _env("PRODTRACERAG_API_TOKEN", "").strip()


def _admin_token() -> str:
    t = _env("PRODTRACERAG_ADMIN_TOKEN", "").strip()
    return t or _public_token()


def _extract_bearer(request: Request) -> str:
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    auth = auth.strip()
    if not auth:
        return ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _client_ip(request: Request) -> str:
    if _envb("TRUST_PROXY_HEADERS", False):
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip()
    host = getattr(getattr(request, "client", None), "host", None)
    return str(host or "")


def _ip_allowed(ip_s: str, cidr_list: List[str]) -> bool:
    if not cidr_list:
        return True
    if not ip_s:
        return False
    try:
        ip = ipaddress.ip_address(ip_s)
    except Exception:
        return False
    for cidr in cidr_list:
        try:
            if ip in ipaddress.ip_network(cidr, strict=False):
                return True
        except Exception:
            continue
    return False


def _enforce_ip_allowlist(request: Request, role: str) -> None:
    cidrs = _envlist("ADMIN_ALLOWLIST_CIDRS") if role == "admin" else _envlist("PUBLIC_ALLOWLIST_CIDRS")
    if not cidrs:
        return
    ip = _client_ip(request)
    if not _ip_allowed(ip, cidrs):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"ip_not_allowed:{role}")


def require_public(request: Request) -> None:
    _enforce_ip_allowlist(request, "public")
    if not _auth_required():
        return
    token = _extract_bearer(request)
    pub = _public_token()
    adm = _admin_token()
    if not pub and not adm:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="auth_required_but_token_not_configured")
    if token and token in {pub, adm}:
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")


def require_admin(request: Request) -> None:
    _enforce_ip_allowlist(request, "admin")
    if not _auth_required():
        return
    token = _extract_bearer(request)
    adm = _admin_token()
    if not adm:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="auth_required_but_admin_token_not_configured")
    if token == adm:
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin_required")


def maybe_require_admin_scope(request: Request, scope: str) -> bool:
    if (scope or "public").lower().strip() == "admin":
        require_admin(request)
        return True
    if _auth_required():
        # Optional elevation when admin token is provided, but do not fail public reads.
        token = _extract_bearer(request)
        adm = _admin_token()
        return bool(adm and token == adm)
    return False


def _mask_paths(text: str) -> str:
    if not text:
        return text
    # Unix/Windows-like path masks.
    text = re.sub(r'([A-Za-z]:\\[^\s"\'<>]+)', '<path>', text)
    text = re.sub(r'(?<![A-Za-z0-9_])/(?:[^\\s"\'<>]+/)*[^\\s"\'<>]*', '<path>', text)
    return text


def _mask_secrets(text: str) -> str:
    if not text:
        return text
    out = text
    out = re.sub(r'(?i)(bearer\s+)[A-Za-z0-9\-\._~+/=]+', r'\1<redacted>', out)
    out = re.sub(r'(?i)(api[_\- ]?key\s*[=:]\s*)[^\s,;]+', r'\1<redacted>', out)
    out = re.sub(r'(?i)(secret\s*[=:]\s*)[^\s,;]+', r'\1<redacted>', out)
    out = re.sub(r'(sk-[A-Za-z0-9]{10,})', '<redacted>', out)
    return out


def mask_text(text: str, max_len: int = 160) -> str:
    s = str(text or "")
    s = _mask_secrets(_mask_paths(s))
    if len(s) > max_len:
        s = s[:max_len].rstrip() + "…"
    return s


def _safe_rel(path: str) -> str:
    try:
        return os.path.normpath(os.path.abspath(os.path.expanduser(path)))
    except Exception:
        return ""


def path_ref(path: str, project_root: str, ingest_root: Optional[str] = None) -> str:
    raw = str(path or "").strip()
    if raw and (not os.path.isabs(raw)) and (not re.match(r"^[A-Za-z]:\\", raw)):
        return os.path.normpath(raw).replace("\\", "/")
    ap = _safe_rel(raw)
    pr = _safe_rel(project_root)
    ir = _safe_rel(ingest_root) if ingest_root else ""
    try:
        if ir and ap.startswith(ir + os.sep):
            return ap[len(ir) + 1:].replace("\\", "/")
        if ir and ap == ir:
            return "."
        if pr and ap.startswith(pr + os.sep):
            return ap[len(pr) + 1:].replace("\\", "/")
        if pr and ap == pr:
            return "."
    except Exception:
        pass
    base = os.path.basename(ap) or "<path>"
    return f"<redacted>/{base}"


def normalize_and_validate_ingest_folder(folder: str, project_root: str) -> str:
    p = _safe_rel(folder)
    if not p or (not os.path.exists(p)) or (not os.path.isdir(p)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_ingest_folder")
    allowed_roots = [_safe_rel(x) for x in _envlist("INGEST_ALLOWED_ROOTS")]
    require_allowlist = _envb("INGEST_REQUIRE_ALLOWLIST", False)
    if allowed_roots:
        ok = False
        for root in allowed_roots:
            try:
                if p == root or p.startswith(root + os.sep):
                    ok = True
                    break
            except Exception:
                continue
        if not ok:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="ingest_folder_not_in_allowlist")
    elif require_allowlist:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="ingest_allowlist_required")
    # Optional: keep ingestion inside repo when set.
    if _envb("INGEST_RESTRICT_TO_PROJECT_ROOT", False):
        pr = _safe_rel(project_root)
        if not (p == pr or p.startswith(pr + os.sep)):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="ingest_folder_not_in_project_root")
    return p


def validate_glob_pattern(glob_pattern: str) -> str:
    gp = (glob_pattern or "").strip() or "**/*.md"
    allowed = set(x.strip() for x in (_env("INGEST_ALLOWED_GLOBS", "**/*.md,**/*.txt,*.md,*.txt")).split(",") if x.strip())
    if allowed and gp not in allowed:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="glob_pattern_not_allowed")
    return gp


def public_citation_snippet(snippet: str) -> str:
    if not _envb("EXPOSE_CITATION_SNIPPETS", True):
        return "<redacted>"
    max_len = int(_env("PUBLIC_CITATION_SNIPPET_MAX", "320") or "320")
    return mask_text(snippet, max_len=max_len)


def sanitize_last_ingest(last_ingest: Dict[str, Any], project_root: str, admin: bool) -> Dict[str, Any]:
    li = dict(last_ingest or {})
    if not li:
        return li
    folder_raw = str(li.get("folder") or "")
    if folder_raw:
        li["folder_ref"] = path_ref(folder_raw, project_root)
        if not admin and not _envb("PUBLIC_CONFIG_SHOW_PATHS", False):
            li["folder"] = "<redacted>"
    return li


def build_config_response(
    *,
    runtime_cfg: Dict[str, Any],
    clarify_min_score: float,
    chroma_dir: str,
    collection: str,
    bm25_ready: bool,
    bm25_chunks: int,
    last_ingest: Dict[str, Any],
    llm_provider: str,
    llm_model: str,
    project_root: str,
    admin: bool,
) -> Dict[str, Any]:
    out = {
        **runtime_cfg,
        "clarify_min_score": clarify_min_score,
        "chroma_dir": chroma_dir,
        "collection": collection,
        "bm25_ready": bm25_ready,
        "bm25_chunks": bm25_chunks,
        "last_ingest": sanitize_last_ingest(last_ingest, project_root=project_root, admin=admin),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
    }
    if not admin:
        if not _envb("PUBLIC_CONFIG_SHOW_PATHS", False):
            out["chroma_dir"] = "<redacted>"
        if not _envb("PUBLIC_CONFIG_SHOW_MODEL", False):
            out["llm_model"] = "<configured>" if str(llm_model or "").strip() else ""
    return out


def sanitize_trace_item(item: Dict[str, Any], *, project_root: str, admin: bool = False, raw: bool = False) -> Dict[str, Any]:
    if item is None:
        return {}
    it = copy.deepcopy(item)
    if raw and admin and _envb("TRACE_ALLOW_RAW_EXPORT", False):
        return it

    # Path and snippet masking is on by default.
    redact_query = _envb("TRACE_REDACT_QUERY", True)
    redact_answer = _envb("TRACE_REDACT_ANSWER", True)
    redact_snippets = _envb("TRACE_REDACT_SNIPPETS", True)

    if redact_query and "q" in it:
        q = str(it.get("q") or "")
        it["q"] = f"<redacted len={len(q)}>"
        it["q_preview"] = mask_text(q, max_len=int(_env("TRACE_Q_PREVIEW_MAX", "80")))
    if redact_answer and "answer" in it:
        ans = str(it.get("answer") or "")
        it["answer"] = f"<redacted len={len(ans)}>"
        it["answer_preview"] = mask_text(ans, max_len=int(_env("TRACE_ANSWER_PREVIEW_MAX", "120")))

    cites = []
    for c in (it.get("citations") or []):
        c2 = dict(c)
        src = str(c2.get("source_ref") or c2.get("source") or "")
        c2["source_ref"] = path_ref(src, project_root)
        c2["source"] = c2["source_ref"]
        if redact_snippets and "snippet" in c2:
            c2["snippet"] = "<redacted>"
        cites.append(c2)
    if "citations" in it:
        it["citations"] = cites
    return it


def sanitize_trace_on_write(item: Dict[str, Any], *, project_root: str) -> Dict[str, Any]:
    if _envb("TRACE_REDACT_ON_WRITE", True):
        return sanitize_trace_item(item or {}, project_root=project_root, admin=False, raw=False)
    return item or {}


def chunk_payload(chunk_id: str, bm25_store_item: Dict[str, Any], *, project_root: str) -> Dict[str, Any]:
    meta = dict((bm25_store_item or {}).get("meta") or {})
    src = str(meta.get("source_ref") or meta.get("source") or "")
    return {
        "ok": True,
        "chunk_id": chunk_id,
        "doc_id": meta.get("doc_id"),
        "title_path": meta.get("title_path"),
        "source": path_ref(src, project_root),
        "source_ref": path_ref(src, project_root),
        "updated_at": meta.get("updated_at"),
        "is_injection": bool((bm25_store_item or {}).get("inj", False)),
        "text": (bm25_store_item or {}).get("text", ""),
    }
