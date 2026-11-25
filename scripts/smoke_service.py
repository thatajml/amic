"""Live service smoke test: the API and the consolidation worker as SEPARATE processes, on a real
Redis (sessions, memory version, M3 profiles, the WriteIntent stream) and a Chroma server, with
the real embedder, tutor model and fact extraction (Ollama).

    python scripts/smoke_service.py --redis redis://localhost:6390/14 --chroma-port 8010

Needs: Redis at --redis, `chroma run --port <chroma-port>` and Ollama. Uses a fresh collection,
learner id and Redis stream; removes the learner at the end. Steps:
1. a new learner asks a question -> Tier 0 R1 SKIP (no memory yet);
2. the worker consolidates it (M1 + M3; observed via GET /learners/{id});
3. `/remember ...` -> route WRITE, acknowledged without an LLM call, then stored by the worker;
4. "what was my project again?" -> R3 DEEP with the remembered note retrieved;
5. DELETE /learners/{id} -> every tier empty, and the next request is R1 SKIP again.
Writes runs/smoke_service.json with the observations and timings.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx


def wait(pred, timeout: float, what: str) -> float:
    t = time.perf_counter()
    while time.perf_counter() - t < timeout:
        if pred():
            return (time.perf_counter() - t) * 1000
        time.sleep(0.2)
    raise SystemExit(f"timed out waiting for {what}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis", default="redis://localhost:6390/14")
    ap.add_argument("--chroma-port", type=int, default=8010)
    ap.add_argument("--port", type=int, default=8077)
    a = ap.parse_args()
    run = uuid.uuid4().hex[:8]
    env = {
        **os.environ,
        "AMIC_SESSION_BACKEND": "redis",
        "AMIC_REDIS_URL": a.redis,
        "AMIC_CHROMA_CLIENT": "http",
        "AMIC_CHROMA_PORT": str(a.chroma_port),
        "AMIC_CHROMA_COLLECTION": f"smoke_{run}",
        "AMIC_METRICS_DB_URL": f"sqlite:///./data/smoke_{run}.db",
        "AMIC_ARM": "amic",
        "AMIC_WRITEPATH__QUEUE": "redis",
        "AMIC_WRITEPATH__STREAM": f"amic:smoke:{run}",
        "AMIC_WRITEPATH__EXTRACT_FACTS": "true",
        "AMIC_ROUTES__SHALLOW_STORES": '["M3", "M2"]',
        "AMIC_ROUTES__DEEP_STORES": '["M3", "M1", "M2"]',
        "PYTHONIOENCODING": "utf-8",
    }
    py = sys.executable
    logs = Path("runs") / f"smoke_service_{run}"
    logs.mkdir(parents=True, exist_ok=True)
    api = subprocess.Popen(
        [py, "-m", "uvicorn", "amic.service.api:app", "--port", str(a.port)],
        env=env,
        stdout=(logs / "api.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    worker = subprocess.Popen(
        [py, "-m", "amic.writepath.worker"],
        env=env,
        stdout=(logs / "worker.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{a.port}"
    cli = httpx.Client(base_url=base, timeout=180)
    tid, out = f"smoke-{run}", {"run": run}
    try:

        def up() -> bool:
            try:
                return cli.get("/health").status_code == 200
            except httpx.HTTPError:
                return False

        out["api_start_ms"] = wait(up, 180, "the API")

        def mem() -> dict:
            return cli.get(f"/learners/{tid}").json()

        def chat(q: str) -> dict:
            t = time.perf_counter()
            r = cli.post("/chat", json={"tenant_id": tid, "session_id": "s1", "query": q})
            r.raise_for_status()
            j = r.json()
            d = j["decision"]
            print(
                f"> {q}\n  {d['route']} by {d['resolved_by']} ({d['rationale'][:70]}); "
                f"{len(j['items'])} items; {(time.perf_counter() - t) * 1000:.0f} ms"
            )
            return {**j, "ms": (time.perf_counter() - t) * 1000}

        r1 = chat("How do I loop over a list and print each item?")
        out["first"] = {"route": r1["decision"]["route"], "by": r1["decision"]["rationale"][:2]}
        out["consolidate_turn_ms"] = wait(lambda: mem()["episodes"] >= 1, 240, "the worker (turn)")
        out["after_turn"] = mem()
        r2 = chat("/remember I am building a weather data logger for my allotment")
        out["remember"] = {"route": r2["decision"]["route"], "answer": r2["answer"], "ms": r2["ms"]}
        out["consolidate_remember_ms"] = wait(
            lambda: mem()["episodes"] >= 2 and mem()["facts"] >= 1, 240, "the worker (remember)"
        )
        out["after_remember"] = mem()
        r3 = chat("What was the project I told you about last time?")
        texts = " ".join(i["text"] for i in r3["items"])
        out["recall"] = {
            "route": r3["decision"]["route"],
            "items": len(r3["items"]),
            "tiers": sorted({i["tier"] for i in r3["items"]}),
            "note_retrieved": "weather data logger" in texts,
            "answer_mentions_it": "weather" in r3["answer"].lower(),
        }
        out["erase"] = cli.delete(f"/learners/{tid}").json()
        # the recall turn may still be queued or in flight: the worker must drop / purge it
        time.sleep(3)
        out["settle_ms"] = wait(
            lambda: mem()["episodes"] == 0 and mem()["facts"] == 0 and not mem()["profile"],
            120,
            "the erasure to hold against queued writes",
        )
        time.sleep(5)  # and stay empty once the worker has drained the stream
        out["after_erase"] = mem()
        r4 = chat("How do I loop over a list and print each item?")
        out["after_erase_route"] = r4["decision"]["rationale"][:2]
        ok = (
            out["first"]["by"] == "R1"
            and out["remember"]["route"] == "WRITE"
            and out["recall"]["note_retrieved"]
            and out["after_erase"]["episodes"] == 0
            and out["after_erase"]["facts"] == 0
            and not out["after_erase"]["profile"]
            and out["after_erase_route"] == "R1"
        )
        out["ok"] = ok
        print(json.dumps(out, indent=2))
        Path("runs/smoke_service.json").write_text(json.dumps(out, indent=2))
        if not ok:
            sys.exit(1)
    finally:
        cli.delete(f"/learners/{tid}")
        for p in (api, worker):
            p.terminate()
        for p in (api, worker):
            p.wait(timeout=30)


if __name__ == "__main__":
    main()
