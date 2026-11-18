# Milestone 0: break-even probe
n> **Reference only.** Server mode adds a ~44 ms per-query transport stall on Windows/Docker; see `probe.md` for the canonical numbers.

Measured 2026-11-18T08:27:37+00:00 on `Windows AMD64 / Intel64 Family 6 Model 191 Stepping 2, GenuineIntel`, Chroma mode `http`. p50 values in ms. **These are local measurements from a probe with synthetic data, not experiment results.**

## Cost terms

| Term | p50 ms | p95 ms |
|---|---|---|
| C_emb (MiniLM, CPU, 1 query) | 6.96 | 7.69 |
| C_feat (signals excl. embedding) | 0.0195 | 0.0362 |
| C_T0 (rules) | 0.0012 | 0.0013 |
| C_T1 (LR predict_proba, 24-d) | 0.1633 | 0.3091 |
| Chroma round trip (heartbeat, no search) | 0.73 | 1.19 |
| C_ret SHALLOW k=3 @ 100 records | 48.45 | 52.55 |
| C_ret DEEP k=30 (+MMR→12) @ 100 records | 48.97 | 52.59 |
| C_ret SHALLOW k=3 @ 1000 records | 48.16 | 49.79 |
| C_ret DEEP k=30 (+MMR→12) @ 1000 records | 48.71 | 51.11 |
| C_ret SHALLOW k=3 @ 10000 records | 48.94 | 52.51 |
| C_ret DEEP k=30 (+MMR→12) @ 10000 records | 52.87 | 56.12 |

## Generation (Ollama, median of reps, 128-token cap)

| Memory tokens | prompt tok | prefill ms | decode ms | total ms |
|---|---|---|---|---|
| 0 | 51 | 30 | 1566 | 1609 |
| 300 | 367 | 114 | 2097 | 2242 |
| 1000 | 1104 | 341 | 1764 | 2177 |

## Inequality (DEEP retrieval at 10000 records, assumed P(T1) = 0.7, no escalation)

r = fraction of queries that still retrieve. Saved = (1−r)·(store query + memory prefill).

| r | gate ms | saved store ms | saved prefill ms | net ms | holds | memory tokens saved / query |
|---|---|---|---|---|---|---|
| 0.3 | 0.135 | 37.01 | 217.8 | 254.7 | yes | 737 |
| 0.4 | 0.135 | 31.72 | 186.7 | 218.3 | yes | 632 |
| 0.5 | 0.135 | 26.44 | 155.6 | 181.9 | yes | 526 |
| 0.6 | 0.135 | 21.15 | 124.4 | 145.5 | yes | 421 |
| 0.7 | 0.135 | 15.86 | 93.3 | 109.1 | yes | 316 |
| 0.8 | 0.135 | 10.57 | 62.2 | 72.7 | yes | 211 |
| 0.9 | 0.135 | 5.29 | 31.1 | 36.3 | yes | 105 |
