# Milestone 0: break-even probe

Measured 2026-11-18T08:29:02+00:00 on `Windows AMD64 / Intel64 Family 6 Model 191 Stepping 2, GenuineIntel`, Chroma mode `local`. p50 values in ms. **These are local measurements from a probe with synthetic data, not experiment results.**

## Cost terms

| Term | p50 ms | p95 ms |
|---|---|---|
| C_emb (MiniLM, CPU, 1 query) | 5.51 | 7.54 |
| C_feat (signals excl. embedding) | 0.0174 | 0.0379 |
| C_T0 (rules) | 0.0015 | 0.0016 |
| C_T1 (LR predict_proba, 24-d) | 0.1783 | 0.2728 |
| Chroma round trip (heartbeat, no search) | 0.00 | 0.00 |
| C_ret SHALLOW k=3 @ 100 records | 1.08 | 1.29 |
| C_ret DEEP k=30 (+MMR→12) @ 100 records | 1.72 | 2.36 |
| C_ret SHALLOW k=3 @ 1000 records | 1.83 | 2.39 |
| C_ret DEEP k=30 (+MMR→12) @ 1000 records | 3.12 | 5.01 |
| C_ret SHALLOW k=3 @ 10000 records | 6.18 | 10.12 |
| C_ret DEEP k=30 (+MMR→12) @ 10000 records | 7.80 | 11.66 |

## Generation (Ollama, median of reps, 128-token cap)

| Memory tokens | prompt tok | prefill ms | decode ms | total ms |
|---|---|---|---|---|
| 0 | 49 | 30 | 1574 | 1629 |
| 300 | 369 | 114 | 2099 | 2243 |
| 1000 | 1104 | 342 | 1762 | 2185 |

## Inequality (DEEP retrieval at 10000 records, assumed P(T1) = 0.7, no escalation)

r = fraction of queries that still retrieve. Saved = (1−r)·(store query + memory prefill).

| r | gate ms | saved store ms | saved prefill ms | net ms | holds | memory tokens saved / query |
|---|---|---|---|---|---|---|
| 0.3 | 0.144 | 5.46 | 218.7 | 224.1 | yes | 738 |
| 0.4 | 0.144 | 4.68 | 187.5 | 192.0 | yes | 633 |
| 0.5 | 0.144 | 3.90 | 156.2 | 160.0 | yes | 528 |
| 0.6 | 0.144 | 3.12 | 125.0 | 128.0 | yes | 422 |
| 0.7 | 0.144 | 2.34 | 93.7 | 95.9 | yes | 317 |
| 0.8 | 0.144 | 1.56 | 62.5 | 63.9 | yes | 211 |
| 0.9 | 0.144 | 0.78 | 31.2 | 31.9 | yes | 105 |

## Reading (2026-11-18)

- **Go.** The gate (~0.14 ms, excluding the embedding, which both arms pay) is two to three orders of
  magnitude below what a skipped DEEP retrieval saves at every r tested.
- **Savings come from the prompt, not the vector store.** A skipped DEEP retrieval saves ~8 ms of
  vector search (at 10⁴ records) but ~310 ms of LLM prefill for ~1 000 memory tokens. The paper
  should lead with memory/prompt tokens and prefill time.
- **End-to-end latency gains will be modest.** Decoding (~1.6–2.1 s for ≤128 tokens) dominates
  and is unaffected by the gate, so expect single-digit to low-double-digit % latency reduction.
  Decode-time variance between answers is larger than the 300 vs 1 000 token prefill difference,
  which is why the "total ms" column isn't monotonic.
- **Chroma runs embedded.** Through the Docker-hosted server on Windows, every query POST paid a
  flat ~44 ms transport stall (GET round trip 0.7 ms, in-process search ~1 ms), so it measured the
  transport, not retrieval. Server-mode numbers are kept in `probe_http.md` for reference only.
  The system and all experiments use the embedded `PersistentClient`.
- `P(T1) = 0.7` is an assumption here; the real value comes from Day-4 Tier-0 coverage.
