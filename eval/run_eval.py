# eval/run_eval.py
from __future__ import annotations

import argparse
import json
import os
import time
import statistics
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import requests


def _auth_headers(admin: bool = False) -> Dict[str, str]:
    api = (os.getenv("PRODTRACERAG_API_TOKEN") or "").strip()
    adm = (os.getenv("PRODTRACERAG_ADMIN_TOKEN") or api).strip()
    tok = adm if admin else (api or adm)
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def p95(xs: List[int]) -> int:
    if not xs:
        return 0
    xs2 = sorted(xs)
    k = int(0.95 * (len(xs2) - 1))
    return int(xs2[k])


def safe_get_json(url: str, timeout: int = 5, admin: bool = False) -> Dict[str, Any]:
    try:
        r = requests.get(url, timeout=timeout, headers=_auth_headers(admin=admin))
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def post_json(url: str, payload: Dict[str, Any], timeout: int = 30, admin: bool = False) -> Tuple[Dict[str, Any], Optional[str], int]:
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout, headers=_auth_headers(admin=admin))
        dt = int((time.time() - t0) * 1000)
        if r.status_code >= 400:
            return {}, f"http_{r.status_code}:{r.text[:200]}", dt
        return r.json(), None, dt
    except Exception as e:
        dt = int((time.time() - t0) * 1000)
        return {}, f"exception:{type(e).__name__}:{e}", dt



# New metrics
_NUM_TOKEN_RE = re.compile(r"(\b\d+(\.\d+)?\b|\b\d+\s*(ms|s|rps|%|qps)\b)", re.IGNORECASE)
_CODE_RE = re.compile(r"\b[A-Z_]{3,}\b")
_PATH_RE = re.compile(r"/[a-zA-Z0-9_\-\/\.\:]+")  # endpoints / paths


def extract_claim_tokens(answer: str) -> List[str]:
    ans = answer or ""
    toks = set()
    for m in _NUM_TOKEN_RE.finditer(ans):
        toks.add(m.group(0).strip().lower())
    for m in _CODE_RE.finditer(ans):
        toks.add(m.group(0).strip())
    for m in _PATH_RE.finditer(ans):
        toks.add(m.group(0).strip())
    return list(toks)


def hallucinated(answer: str, evidence_text: str) -> bool:
    ans = (answer or "").strip()
    if not ans:
        return False
    ev = (evidence_text or "").lower()
    toks = extract_claim_tokens(ans)
    if not toks:
        return False
    miss = [t for t in toks if t.lower() not in ev]
    # any missing hard token leads to hallucination
    return len(miss) > 0


def parse_sample(s: Dict[str, Any]) -> Dict[str, Any]:
    q = s.get("query") or s.get("q")
    if not q:
        raise ValueError("missing query")

    category = s.get("category") or s.get("tag") or "unknown"
    should_refuse = bool(s.get("should_refuse", False))

    gold_doc_ids = []
    if "gold_doc_ids" in s and isinstance(s["gold_doc_ids"], list):
        gold_doc_ids = [str(x) for x in s["gold_doc_ids"] if str(x)]
    elif s.get("gold_doc_id"):
        gold_doc_ids = [str(s["gold_doc_id"])]

    return {
        "id": s.get("id", ""),
        "category": category,
        "query": q,
        "gold_doc_ids": gold_doc_ids,
        "should_refuse": should_refuse,
    }


def fetch_full_evidence(base_url: str, chunk_ids: List[str]) -> str:
    texts = []
    for cid in chunk_ids[:8]:
        j = safe_get_json(base_url + "/chunk?chunk_id=" + requests.utils.quote(cid), timeout=5, admin=True)
        if j.get("ok") and j.get("text"):
            texts.append(str(j["text"]))
    return "\n\n".join(texts)


def run_once(args, run_name: str, cfg_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # optionally override server config 
    if cfg_override:
        _, err, _ = post_json(args.base_url + "/set_config", cfg_override, timeout=10, admin=True)
        if err:
            print(f"[WARN] set_config failed: {err} override={cfg_override}")

    # load samples
    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(parse_sample(json.loads(line)))

    # fetch server config once
    cfg = safe_get_json(args.base_url + "/config", timeout=5, admin=True)
    last_ingest = cfg.get("last_ingest") or {}

    # metrics
    total = 0
    answerable_total = 0
    answered_total = 0  # answerable & not refused
    recall_hits = 0
    top1_hits = 0
    refuse_correct = 0

    # new metrics
    citation_precision_sum = 0.0
    halluc_cnt = 0

    lat_ms: List[int] = []
    failures: List[Dict[str, Any]] = []

    by_cat = defaultdict(lambda: {
        "total": 0,
        "answerable_n": 0,
        "refusal_n": 0,
        "answered_n": 0,
        "recall_hit": 0,
        "top1_hit": 0,
        "refusal_correct": 0,
        "citation_precision_sum": 0.0,
        "halluc_cnt": 0,
    })

    for s in samples:
        total += 1
        cat = s["category"]
        by_cat[cat]["total"] += 1

        q = s["query"]
        golds = s["gold_doc_ids"]
        should_refuse = s["should_refuse"]

        if should_refuse:
            by_cat[cat]["refusal_n"] += 1
        else:
            by_cat[cat]["answerable_n"] += 1
            answerable_total += 1

        payload = {"q": q, "topk": args.k, "min_score": args.min_score, "debug": False}
        resp, err, dt = post_json(args.base_url + "/ask", payload, timeout=30)
        if err:
            failures.append({
                "id": s.get("id"),
                "category": cat,
                "query": q,
                "gold_doc_ids": golds,
                "error": err,
            })
            lat_ms.append(dt)
            continue

        lat_ms.append(int(resp.get("stats", {}).get("total_ms", dt)))

        refused = bool(resp.get("refused", False))
        citations = resp.get("citations", []) or []
        doc_ids = [c.get("doc_id") for c in citations[:args.k] if c.get("doc_id")]
        chunk_ids = [c.get("chunk_id") for c in citations[:args.k] if c.get("chunk_id")]
        answer = resp.get("answer", "") or ""

        # refusal correctness
        if refused == should_refuse:
            refuse_correct += 1
            by_cat[cat]["refusal_correct"] += 1
        if not should_refuse and not refused:
            by_cat[cat]["answered_n"] += 1
            answered_total += 1

        # retrieval metrics only for answerable samples
        if not should_refuse:
            hit_recall = any(g in doc_ids for g in golds) if golds else False
            hit_top1 = (doc_ids[0] in set(golds)) if (golds and doc_ids) else False
            if hit_recall:
                recall_hits += 1
                by_cat[cat]["recall_hit"] += 1
            if hit_top1:
                top1_hits += 1
                by_cat[cat]["top1_hit"] += 1

            # CitationPrecision: among returned citations, what fraction comes from gold docs
            if doc_ids:
                in_gold = sum(1 for d in doc_ids if d in set(golds))
                cp = in_gold / len(doc_ids)
            else:
                cp = 0.0
            citation_precision_sum += cp
            by_cat[cat]["citation_precision_sum"] += cp

            # HallucinationRate (heuristic)
            if not refused:
                ev_text = fetch_full_evidence(args.base_url, chunk_ids) if args.use_full_chunk_for_hallucination else "\n".join([c.get("snippet", "") for c in citations])
                if hallucinated(answer, ev_text):
                    halluc_cnt += 1
                    by_cat[cat]["halluc_cnt"] += 1

            # failure cases
            if refused or (not hit_recall):
                failures.append({
                    "id": s.get("id"),
                    "category": cat,
                    "query": q,
                    "gold_doc_ids": golds,
                    "refused": refused,
                    "top_doc_ids": doc_ids[:args.k],
                    "trace_id": resp.get("trace_id"),
                    "refusal_reason": resp.get("refusal_reason"),
                })
        else:
            # should refuse but answered -> failure
            if not refused:
                failures.append({
                    "id": s.get("id"),
                    "category": cat,
                    "query": q,
                    "gold_doc_ids": [],
                    "refused": refused,
                    "top_doc_ids": doc_ids[:args.k],
                    "trace_id": resp.get("trace_id"),
                    "refusal_reason": resp.get("refusal_reason"),
                })

    recall = recall_hits / answerable_total if answerable_total else 0.0
    top1 = top1_hits / answerable_total if answerable_total else 0.0
    refuse_acc = refuse_correct / total if total else 0.0
    citation_precision = citation_precision_sum / answerable_total if answerable_total else 0.0
    halluc_rate = halluc_cnt / answered_total if answered_total else 0.0

    avg_ms = statistics.mean(lat_ms) if lat_ms else 0.0
    p95_ms = p95(lat_ms)

    # write markdown report
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# ProdTraceRAG Eval Report\n\n")
        f.write(f"- timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- run_name: {run_name}\n")
        f.write(f"- base_url: {args.base_url}\n")
        f.write(f"- dataset: {args.input}\n")
        f.write(f"- k: {args.k}\n")
        f.write(f"- min_score(req): {args.min_score}\n")
        f.write(f"- server_config: {json.dumps(cfg, ensure_ascii=False)}\n\n")

        f.write("## Overall\n")
        f.write(f"- Recall@{args.k}: {recall:.3f}\n")
        f.write(f"- Top1: {top1:.3f}\n")
        f.write(f"- RefusalAcc: {refuse_acc:.3f}\n")
        f.write(f"- CitationPrecision@{args.k} (avg): {citation_precision:.3f}\n")
        f.write(f"- HallucinationRate (heuristic): {halluc_rate:.3f}\n\n")

        f.write("## Latency\n")
        f.write(f"- avg_ms: {avg_ms:.1f}\n")
        f.write(f"- p95_ms: {p95_ms}\n\n")

        f.write("## By Category\n")
        f.write("| category | total | answerable | refusal | Recall@k | Top1 | RefusalAcc | CitePrec@k | HallucRate |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for cat, m in sorted(by_cat.items(), key=lambda x: x[0]):
            ans_n = m["answerable_n"]
            ref_n = m["refusal_n"]
            answered_n = m["answered_n"]
            recall_cat = (m["recall_hit"] / ans_n) if ans_n else 0.0
            top1_cat = (m["top1_hit"] / ans_n) if ans_n else 0.0
            refuse_cat = (m["refusal_correct"] / m["total"]) if m["total"] else 0.0
            citep_cat = (m["citation_precision_sum"] / ans_n) if ans_n else 0.0
            hall_cat = (m["halluc_cnt"] / answered_n) if answered_n else 0.0
            f.write(f"| {cat} | {m['total']} | {ans_n} | {ref_n} | {recall_cat:.3f} | {top1_cat:.3f} | {refuse_cat:.3f} | {citep_cat:.3f} | {hall_cat:.3f} |\n")
        f.write("\n")

        f.write("## Failures (Top 40)\n")
        for x in failures[:40]:
            f.write(
                f"- **{x.get('id','')}** [{x.get('category')}] Q: {x.get('query')} | "
                f"gold: {x.get('gold_doc_ids')} | refused: {x.get('refused')} | "
                f"top: {x.get('top_doc_ids')} | trace: {x.get('trace_id')} | "
                f"reason/error: {x.get('refusal_reason') or x.get('error')}\n"
            )

    print(f"[OK] report saved to: {args.out}")
    print(f"Overall: Recall@{args.k}={recall:.3f}, Top1={top1:.3f}, RefusalAcc={refuse_acc:.3f}, "
          f"CitePrec@{args.k}={citation_precision:.3f}, HallucRate={halluc_rate:.3f}, avg={avg_ms:.1f}ms, p95={p95_ms}ms")

    return {
        "run_name": run_name,
        "recall": recall,
        "top1": top1,
        "refusal_acc": refuse_acc,
        "citation_precision": citation_precision,
        "hallucination_rate": halluc_rate,
        "avg_ms": avg_ms,
        "p95_ms": p95_ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://127.0.0.1:8000")
    ap.add_argument("--input", default="eval/eval_set.jsonl")
    ap.add_argument("--out", default="reports/report_latest.md")
    ap.add_argument("--run_name", default="manual")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--min_score", type=float, default=0.25)

    # hallucination check
    ap.add_argument("--use_full_chunk_for_hallucination", action="store_true")

    # ablation
    ap.add_argument("--ablation", action="store_true")

    # CI gate
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--min_recall", type=float, default=0.85)
    ap.add_argument("--min_refusal_acc", type=float, default=0.85)
    ap.add_argument("--min_citation_precision", type=float, default=0.70)
    ap.add_argument("--max_hallucination_rate", type=float, default=0.20)

    args = ap.parse_args()

    if not args.ablation:
        m = run_once(args, args.run_name, cfg_override=None)
        if args.gate:
            failed = []
            if m["recall"] < args.min_recall:
                failed.append(f"Recall@k {m['recall']:.3f} < {args.min_recall:.3f}")
            if m["refusal_acc"] < args.min_refusal_acc:
                failed.append(f"RefusalAcc {m['refusal_acc']:.3f} < {args.min_refusal_acc:.3f}")
            if m["citation_precision"] < args.min_citation_precision:
                failed.append(f"CitationPrecision {m['citation_precision']:.3f} < {args.min_citation_precision:.3f}")
            if m["hallucination_rate"] > args.max_hallucination_rate:
                failed.append(f"HallucinationRate {m['hallucination_rate']:.3f} > {args.max_hallucination_rate:.3f}")
            if failed:
                print("[FAIL] Gate not satisfied:")
                for x in failed:
                    print(" - " + x)
                raise SystemExit(1)
        return

    # ablation runs
    base = run_once(args, args.run_name + "_full", cfg_override={
        "enable_bm25": True,
        "enable_expansion": True,
        "enable_diversity": True,
    })
    abls = [
        ("no_bm25", {"enable_bm25": False, "enable_expansion": True, "enable_diversity": True}),
        ("no_expansion", {"enable_bm25": True, "enable_expansion": False, "enable_diversity": True}),
        ("no_diversity", {"enable_bm25": True, "enable_expansion": True, "enable_diversity": False}),
    ]
    print("\n=== Ablation (delta vs full) ===")
    for name, cfg in abls:
        out_path = args.out.replace(".md", f"_{name}.md")
        args2 = argparse.Namespace(**{**vars(args), "out": out_path})
        m = run_once(args2, args.run_name + "_" + name, cfg_override=cfg)
        print(f"- {name}: ΔRecall={m['recall']-base['recall']:+.3f}, ΔTop1={m['top1']-base['top1']:+.3f}, "
              f"ΔRefAcc={m['refusal_acc']-base['refusal_acc']:+.3f}, "
              f"ΔCitePrec={m['citation_precision']-base['citation_precision']:+.3f}, "
              f"ΔHall={m['hallucination_rate']-base['hallucination_rate']:+.3f}, "
              f"Δp95={m['p95_ms']-base['p95_ms']:+.1f}ms")


if __name__ == "__main__":
    main()
