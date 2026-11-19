"""LoCoMo (Maharana et al., 2024) as an external check on gate DECISIONS (not answer quality).

Labels come from LoCoMo's own QA categories, so no generation/judge is needed:
    1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop  -> memory-required
    5 adversarial (asks about things never said)           -> memory-not-required analogue
Depth: single-hop -> SHALLOW, otherwise DEEP.

Caveat (plan R10): category 5 is a WEAK negative. Answering "that was never mentioned"
confidently still benefits from checking memory, so LoCoMo results are reported separately and
are not pooled with the education set.
"""

from __future__ import annotations

import json
import random
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from amic.eval.datasets.edu import Probe

PATH = Path("data/locomo/locomo10.json")
CATEGORY = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}


def parse_dt(s: str) -> datetime:
    """'1:56 pm on 8 May, 2023' -> aware datetime."""
    s = re.sub(r"\s+", " ", s.strip())
    return datetime.strptime(s, "%I:%M %p on %d %B, %Y").replace(tzinfo=UTC)


def load(path: Path = PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def sessions(conv: dict[str, Any]) -> list[tuple[str, datetime, list[dict[str, Any]]]]:
    c = conv["conversation"]
    out = []
    for k in c:
        m = re.fullmatch(r"session_(\d+)", k)
        if m and c[k]:
            out.append((k, parse_dt(c[f"{k}_date_time"]), c[k]))
    return sorted(out, key=lambda x: int(x[0].split("_")[1]))


def tenant_of(conv: dict[str, Any]) -> str:
    return f"locomo-{conv['sample_id']}"


def turn_text(t: dict[str, Any]) -> str:
    txt = f"{t['speaker']}: {t['text']}"
    if t.get("blip_caption"):
        txt += f" [shared an image: {t['blip_caption']}]"
    return txt


def seed(comps: Any, convs: list[dict[str, Any]], write_memory: bool = True) -> int:
    """One M1 record per dialogue turn, timestamped from the session date.
    write_memory=False re-seeds only the session metadata."""
    n = 0
    for conv in convs:
        tenant = tenant_of(conv)
        if write_memory:
            comps.episodic.delete_tenant(tenant)
        comps.sessions.erase_tenant(tenant)
        for sid, start, turns in sessions(conv):
            comps.sessions.touch(tenant, f"{tenant}-{sid}", start)
            if not write_memory:
                comps.sessions.touch(
                    tenant, f"{tenant}-{sid}", start + timedelta(minutes=len(turns))
                )
                continue
            texts = [turn_text(t) for t in turns]
            vecs = comps.embedder.encode_many(texts)
            for j, (t, txt) in enumerate(zip(turns, texts, strict=True)):
                comps.episodic.add(
                    tenant,
                    f"{tenant}-{t.get('dia_id', f'{sid}-{j}')}",
                    txt,
                    vecs[j],
                    start + timedelta(minutes=j),
                    {"session_id": f"{tenant}-{sid}", "kind": "dialogue"},
                )
                n += 1
            comps.sessions.touch(tenant, f"{tenant}-{sid}", start + timedelta(minutes=len(turns)))
    return n


def probes(convs: list[dict[str, Any]], per_conv: int = 35, seed_: int = 5) -> list[Probe]:
    """Stratified sample of QA items per conversation (category proportions preserved)."""
    rng = random.Random(seed_)
    out: list[Probe] = []
    for conv in convs:
        tenant = tenant_of(conv)
        ts = sessions(conv)[-1][1] + timedelta(days=1)
        by_cat: dict[int, list[dict[str, Any]]] = {}
        for q in conv["qa"]:
            by_cat.setdefault(int(q["category"]), []).append(q)
        total = sum(len(v) for v in by_cat.values())
        for cat, qs in sorted(by_cat.items()):
            k = max(1, round(per_conv * len(qs) / total))
            for q in rng.sample(qs, min(k, len(qs))):
                out.append(
                    Probe(
                        probe_id=f"{tenant}-q{len(out) + 1:04d}",
                        learner_id=tenant,
                        category=f"locomo_{CATEGORY[cat]}",
                        intended_memory=cat != 5,
                        query=q["question"],
                        ts=ts,
                    )
                )
    return out


def labels_frame(ps: list[Probe]) -> Any:
    import pandas as pd

    rows = []
    for p in ps:
        rows.append(
            {
                "probe_id": p.probe_id,
                "learner_id": p.learner_id,
                "split": "locomo",
                "category": p.category,
                "intended_memory": p.intended_memory,
                "query": p.query,
                "has_m0": False,
                "label_required": p.intended_memory,
                "depth": None
                if not p.intended_memory
                else ("SHALLOW" if p.category == "locomo_single_hop" else "DEEP"),
            }
        )
    return pd.DataFrame(rows)
