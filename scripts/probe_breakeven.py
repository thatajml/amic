"""Milestone 0: break-even probe (brief 9.3, plan Day 1).

Measures the cost terms on this machine and checks whether the AMIC inequality can hold:

    C_feat + C_T0 + P(T1)*C_T1 + P(esc)*(C_ret + C_gen)  <  (1 - r) * C_retrieve_total

C_emb cancels out (it is paid by both arms; under P2 AMIC's signal bus reuses it for retrieval).
C_retrieve_total = vector-store query + the extra prompt-processing time the injected memory
tokens cost the LLM (prefill). Both are what a SKIP saves.

Usage:
    python scripts/probe_breakeven.py                 # everything (needs Ollama for C_gen)
    python scripts/probe_breakeven.py --no-gen        # skip LLM timing
    python scripts/probe_breakeven.py --chroma http   # use the docker Chroma server
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

DIM = 384
RNG = np.random.default_rng(0)


def timeit(fn: Callable[[], object], reps: int, warmup: int = 3) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000)
    xs.sort()
    return {
        "p50_ms": statistics.median(xs),
        "p95_ms": xs[min(len(xs) - 1, int(0.95 * len(xs)))],
        "mean_ms": statistics.fmean(xs),
        "n": reps,
    }


def unit(n: int) -> np.ndarray:
    v = RNG.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# --------------------------------------------------------------------------- C_emb
def probe_embedding(model_name: str) -> dict[str, float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    q = "Why does my for loop print the last item twice?"
    return timeit(lambda: model.encode(q, normalize_embeddings=True), reps=50)


# --------------------------------------------------------------------------- C_feat, C_T0, C_T1
DEICTIC = re.compile(
    r"\b(again|last time|you said|earlier|before|remember|my (progress|mistake|code|answer)s?|"
    r"we (did|covered|discussed)|previous(ly)?)\b",
    re.I,
)


def probe_gate() -> dict[str, dict[str, float]]:
    from sklearn.linear_model import LogisticRegression

    query = "Can you explain again what I got wrong with list indexing last time?"
    q = unit(1)[0]
    centroids = unit(8)
    bloom = np.zeros(4096, dtype=bool)
    bloom[RNG.integers(0, 4096, 300)] = True
    m0 = unit(6)

    def features() -> np.ndarray:
        s1 = [bool(DEICTIC.search(query)), len(query.split()), query.endswith("?")]
        sims = centroids @ q
        top2 = np.partition(sims, -2)[-2:]
        m0_cov = float((m0 @ q).max())
        toks = re.findall(r"[a-z_]+", query.lower())
        bloom_hit = any(bloom[hash(t) % 4096] for t in toks)
        f = np.zeros(24, dtype=np.float32)
        f[:3] = s1
        f[3:6] = (top2.max(), top2.max() - top2.min(), m0_cov)
        f[6] = bloom_hit
        return f

    def tier0() -> str:
        if not centroids.size:
            return "SKIP"
        if DEICTIC.search(query):
            return "DEEP"
        return "ABSTAIN"

    X = RNG.standard_normal((500, 24))
    y = (X[:, 0] + RNG.standard_normal(500) > 0).astype(int)
    lr = LogisticRegression().fit(X, y)
    f = features().reshape(1, -1)

    return {
        "C_feat": timeit(features, reps=2000),
        "C_T0": timeit(tier0, reps=5000),
        "C_T1": timeit(lambda: lr.predict_proba(f), reps=2000),
    }


# --------------------------------------------------------------------------- C_ret
def make_chroma(mode: str, host: str, port: int):
    import chromadb

    if mode == "http":
        return chromadb.HttpClient(host=host, port=port)
    return chromadb.EphemeralClient()


def probe_retrieval(mode: str, host: str, port: int, sizes: list[int]) -> dict[str, dict]:
    client = make_chroma(mode, host, port)
    out: dict[str, dict] = {"rtt_heartbeat": timeit(client.heartbeat, reps=30)}
    for n in sizes:
        name = f"probe_{n}_{uuid.uuid4().hex[:6]}"
        col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        vecs = unit(n)
        tenants = [f"u{i % 20}" for i in range(n)]  # 20 tenants share one collection
        for start in range(0, n, 2000):
            sl = slice(start, start + 2000)
            col.add(
                ids=[f"id{i}" for i in range(start, min(start + 2000, n))],
                embeddings=vecs[sl].tolist(),
                documents=[
                    f"turn {i}: student asked about loops and lists"
                    for i in range(start, min(start + 2000, n))
                ],
                metadatas=[{"tenant_id": t, "tier": "M1"} for t in tenants[sl]],
            )
        q = unit(1)[0].tolist()
        res = {}
        for label, k in [("shallow_k3", 3), ("deep_k30", 30)]:
            res[label] = timeit(
                lambda k=k, col=col, q=q: col.query(
                    query_embeddings=[q],
                    n_results=k,
                    where={"tenant_id": "u0"},
                    include=["documents", "metadatas", "distances", "embeddings"],
                ),
                reps=30,
            )
        # MMR over 30 candidates -> 12, the DEEP post-processing cost
        cand = unit(30)
        qv = np.asarray(q, dtype=np.float32)
        res["mmr_30_to_12"] = timeit(lambda qv=qv, cand=cand: mmr(qv, cand, 12, 0.5), reps=500)
        out[str(n)] = res
        client.delete_collection(name)
    return out


def mmr(q: np.ndarray, cand: np.ndarray, k: int, lam: float) -> list[int]:
    rel = cand @ q
    chosen: list[int] = []
    rest = list(range(len(cand)))
    while rest and len(chosen) < k:
        div = (cand[rest] @ cand[chosen].T).max(axis=1) if chosen else np.zeros(len(rest))
        best = rest[int(np.argmax(lam * rel[rest] - (1 - lam) * div))]
        chosen.append(best)
        rest.remove(best)
    return chosen


# --------------------------------------------------------------------------- C_gen
FILLER = (
    "Previous session note: the learner confused list.append with list.extend, "
    "struggled with off-by-one errors in range(), and prefers short worked examples. "
)


def probe_generation(
    model: str, base_url: str, mem_tokens: list[int], reps: int
) -> dict[str, dict]:
    import ollama

    client = ollama.Client(host=base_url)
    question = "Explain the difference between a list and a tuple in Python in three sentences."
    words_per_token = 0.75
    out: dict[str, dict] = {}
    client.generate(model=model, prompt="hi", options={"num_predict": 1})  # load model
    for mt in mem_tokens:
        n_words = int(mt * words_per_token)
        memory = (FILLER * (n_words // len(FILLER.split()) + 1)).split()[:n_words]
        prompt = (
            f"Relevant memory:\n{' '.join(memory)}\n\n" if mt else ""
        ) + f"Student: {question}"
        runs = []
        for i in range(reps + 1):  # rep 0 is a warmup for this prompt length
            r = client.generate(
                model=model,
                # nonce FIRST: Ollama reuses the KV cache for a shared prefix, so a suffix
                # nonce would leave the memory block cached and hide its prefill cost
                prompt=f"[run {uuid.uuid4().hex}]" + chr(10) + prompt,
                options={"temperature": 0, "seed": 0, "num_predict": 128},
            )
            if i == 0:
                continue
            runs.append(
                {
                    "total_ms": r["total_duration"] / 1e6,
                    "prefill_ms": r["prompt_eval_duration"] / 1e6,
                    "decode_ms": r["eval_duration"] / 1e6,
                    "prompt_tokens": r["prompt_eval_count"],
                    "completion_tokens": r["eval_count"],
                }
            )
        out[str(mt)] = {k: statistics.median(x[k] for x in runs) for k in runs[0]}
    return out


# --------------------------------------------------------------------------- inequality
def breakeven(results: dict, p_t1: float, n_ref: str) -> list[dict]:
    g = results["gate"]
    gate_cost = g["C_feat"]["p50_ms"] + g["C_T0"]["p50_ms"] + p_t1 * g["C_T1"]["p50_ms"]
    ret = results["retrieval"][n_ref]
    c_store = ret["deep_k30"]["p50_ms"] + ret["mmr_30_to_12"]["p50_ms"]
    gen = results.get("generation")
    c_prefill = 0.0
    mem_tok = 0
    if gen:
        keys = sorted(gen, key=int)
        lo, hi = gen[keys[0]], gen[keys[-1]]
        c_prefill = hi["prefill_ms"] - lo["prefill_ms"]
        mem_tok = hi["prompt_tokens"] - lo["prompt_tokens"]
    rows = []
    for r in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        saved = (1 - r) * (c_store + c_prefill)
        rows.append(
            {
                "r": r,
                "gate_ms": gate_cost,
                "saved_store_ms": (1 - r) * c_store,
                "saved_prefill_ms": (1 - r) * c_prefill,
                "saved_total_ms": saved,
                "net_ms": saved - gate_cost,
                "holds": saved > gate_cost,
                "saved_memory_tokens_per_query": (1 - r) * mem_tok,
            }
        )
    return rows


def render_md(results: dict, rows: list[dict], p_t1: float, n_ref: str) -> str:
    g, ret = results["gate"], results["retrieval"]
    L = [
        "# Milestone 0: break-even probe",
        "",
        f"Measured {results['meta']['ts']} on `{results['meta']['machine']}`, "
        f"Chroma mode `{results['meta']['chroma_mode']}`. "
        "p50 values in ms. **These are local measurements from a probe with synthetic data, "
        "not experiment results.**",
        "",
        "## Cost terms",
        "",
        "| Term | p50 ms | p95 ms |",
        "|---|---|---|",
        f"| C_emb (MiniLM, CPU, 1 query) | {results['embedding']['p50_ms']:.2f} | {results['embedding']['p95_ms']:.2f} |",
        f"| C_feat (signals excl. embedding) | {g['C_feat']['p50_ms']:.4f} | {g['C_feat']['p95_ms']:.4f} |",
        f"| C_T0 (rules) | {g['C_T0']['p50_ms']:.4f} | {g['C_T0']['p95_ms']:.4f} |",
        f"| C_T1 (LR predict_proba, 24-d) | {g['C_T1']['p50_ms']:.4f} | {g['C_T1']['p95_ms']:.4f} |",
    ]
    L.append(
        f"| Chroma round trip (heartbeat, no search) | {ret['rtt_heartbeat']['p50_ms']:.2f} | {ret['rtt_heartbeat']['p95_ms']:.2f} |"
    )
    for n, r in ret.items():
        if n == "rtt_heartbeat":
            continue
        L.append(
            f"| C_ret SHALLOW k=3 @ {n} records | {r['shallow_k3']['p50_ms']:.2f} | {r['shallow_k3']['p95_ms']:.2f} |"
        )
        L.append(
            f"| C_ret DEEP k=30 (+MMR→12) @ {n} records | {r['deep_k30']['p50_ms'] + r['mmr_30_to_12']['p50_ms']:.2f} | {r['deep_k30']['p95_ms'] + r['mmr_30_to_12']['p95_ms']:.2f} |"
        )
    gen = results.get("generation")
    if gen:
        L += [
            "",
            "## Generation (Ollama, median of reps, 128-token cap)",
            "",
            "| Memory tokens | prompt tok | prefill ms | decode ms | total ms |",
            "|---|---|---|---|---|",
        ]
        for mt, v in sorted(gen.items(), key=lambda kv: int(kv[0])):
            L.append(
                f"| {mt} | {v['prompt_tokens']} | {v['prefill_ms']:.0f} | {v['decode_ms']:.0f} | {v['total_ms']:.0f} |"
            )
    L += [
        "",
        f"## Inequality (DEEP retrieval at {n_ref} records, assumed P(T1) = {p_t1}, no escalation)",
        "",
        "r = fraction of queries that still retrieve. Saved = (1−r)·(store query + memory prefill).",
        "",
        "| r | gate ms | saved store ms | saved prefill ms | net ms | holds | memory tokens saved / query |",
        "|---|---|---|---|---|---|---|",
    ]
    for x in rows:
        L.append(
            f"| {x['r']:.1f} | {x['gate_ms']:.3f} | {x['saved_store_ms']:.2f} | {x['saved_prefill_ms']:.1f} | "
            f"{x['net_ms']:.1f} | {'yes' if x['holds'] else '**no**'} | {x['saved_memory_tokens_per_query']:.0f} |"
        )
    return "\n".join(L) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--chroma", choices=["local", "http"], default="local")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--sizes", default="100,1000,10000")
    ap.add_argument("--embedding", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--mem-tokens", default="0,300,1000")
    ap.add_argument("--gen-reps", type=int, default=3)
    ap.add_argument("--p-t1", type=float, default=0.7, help="assumed share reaching Tier 1")
    ap.add_argument("--out", default="runs/probe.json")
    ap.add_argument("--md", default="docs/results/probe.md")
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(",")]
    results: dict = {
        "meta": {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "machine": f"{platform.system()} {platform.machine()} / {platform.processor() or 'cpu'}",
            "chroma_mode": a.chroma,
            "args": vars(a),
        }
    }
    print("embedding ...")
    results["embedding"] = probe_embedding(a.embedding)
    print("gate ...")
    results["gate"] = probe_gate()
    print("retrieval ...")
    results["retrieval"] = probe_retrieval(a.chroma, a.host, a.port, sizes)
    if not a.no_gen:
        print("generation ...")
        results["generation"] = probe_generation(
            a.model, a.ollama, [int(x) for x in a.mem_tokens.split(",")], a.gen_reps
        )
    n_ref = str(max(sizes))
    rows = breakeven(results, a.p_t1, n_ref)
    results["breakeven"] = rows

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    md = render_md(results, rows, a.p_t1, n_ref)
    Path(a.md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.md).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
