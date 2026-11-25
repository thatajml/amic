"""Replay verify-and-escalate over stored answers and verdicts (pre-registered E4).

For a run whose answers were generated without escalation, a verifier mode decides which SKIP
answers would have been escalated. An escalated answer is byte for byte the always-SHALLOW
answer for the same probe and seed (identical prompt and seed; generation cache), so its pairwise
verdict is the always-SHALLOW verdict. Nothing is regenerated or re-judged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MODES = ("lexical", "llm", "both")


def _jsonl(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_json(path, lines=True)
    return None if df.empty else df


def replay_escalation(
    run: str,
    mode: str,
    escalate_to: str = "always_shallow",
    runs_dir: Path = Path("runs"),
) -> pd.DataFrame | None:
    """Pairwise rows of `run` with escalation replayed under `mode` (adds an `escalated` column).

    None when the run, its verify replay or the escalation target's verdicts are missing.
    """
    if mode not in MODES:
        raise ValueError(mode)
    pw = _jsonl(runs_dir / run / "pairwise_vs_always.jsonl")
    vr = _jsonl(runs_dir / run / "verify_replay.jsonl")
    tgt = _jsonl(runs_dir / escalate_to / "pairwise_vs_always.jsonl")
    if pw is None or vr is None or tgt is None:
        return None
    m = pw.merge(vr[["probe_id", "seed", "lexical", "llm"]], on=["probe_id", "seed"], how="left")
    lex = m.lexical.notna() & (m.lexical.astype(str) != "None")
    llm = m.llm.fillna(False).astype(bool)
    fail = {"lexical": lex, "llm": llm, "both": lex | llm}[mode]
    m["escalated"] = (m.route == "SKIP") & fail
    tv = tgt.set_index(["probe_id", "seed"]).result
    keys = list(zip(m.probe_id, m.seed, strict=True))
    esc_result = pd.Series([tv.get(k) for k in keys], index=m.index)
    if esc_result[m.escalated].isna().any():
        raise ValueError(f"{escalate_to} has no verdict for some escalated rows of {run}")
    m.loc[m.escalated, "result"] = esc_result[m.escalated]
    return m
