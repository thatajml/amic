from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

from amic.eval.datasets import edu, locomo

_spec = importlib.util.spec_from_file_location("label_oracle", Path("scripts/label_oracle.py"))
label_oracle = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(label_oracle)  # type: ignore[union-attr]


def _raw_history(p: edu.Persona) -> dict:
    turns = [
        {"role": "student", "text": "hi, my loop breaks"},
        {"role": "tutor", "text": "show me the code"},
        {"role": "student", "text": "for i in range(len(x)+1)"},
        {"role": "tutor", "text": "range goes one too far"},
    ]
    return {"sessions": [{"turns": turns} for _ in p.plan], "facts": ["a", "b", "c", "d"]}


def test_personas_are_deterministic_and_well_formed():
    a, b = edu.sample_personas(5), edu.sample_personas(5)
    assert [x.learner_id for x in a] == [f"edu-00{i}" for i in range(1, 6)]
    assert [x.plan for x in a] == [x.plan for x in b]
    for p in a:
        assert 3 <= len(p.plan) <= 5
        assert sum(len(s["misconceptions"]) for s in p.plan) >= 2
        starts = [s["start"] for s in p.plan]
        assert starts == sorted(starts) and p.probe_time > starts[-1]


def test_parse_history_and_exchanges():
    p = edu.sample_personas(1)[0]
    lr = edu.parse_history(p, _raw_history(p))
    assert len(lr.sessions) == len(p.plan)
    assert edu.exchanges(lr.sessions[0]) == [
        ("hi, my loop breaks", "show me the code"),
        ("for i in range(len(x)+1)", "range goes one too far"),
    ]
    bad = _raw_history(p)
    bad["sessions"] = bad["sessions"][:-1]
    with pytest.raises(ValueError):
        edu.parse_history(p, bad)


def test_parse_probes_validates_mix():
    p = edu.sample_personas(1)[0]
    lr = edu.parse_history(p, _raw_history(p))
    items = []
    for cat, (n, _) in edu.PROBE_MIX.items():
        for i in range(n):
            it = {"category": cat, "query": f"{cat} q{i}", "relevant_facts": [0, "x"]}
            if cat == "m0_followup":
                it["m0_context"] = [
                    {"role": "student", "text": "what is a set"},
                    {"role": "tutor", "text": "an unordered collection"},
                ]
            items.append(it)
    probes = edu.parse_probes(lr, {"probes": items})
    assert len(probes) == edu.PROBES_PER_LEARNER
    assert all(pr.relevant_facts == [0] for pr in probes)
    assert all(bool(pr.m0_context) == (pr.category == "m0_followup") for pr in probes)
    with pytest.raises(ValueError):
        edu.parse_probes(lr, {"probes": [i for i in items if i["category"] != "explicit_recall"]})


def test_seed_learner_writes_m1_and_session_meta(comps):
    p = edu.sample_personas(1)[0]
    lr = edu.parse_history(p, _raw_history(p))
    n = edu.seed_learner(comps, lr)
    assert n == 2 * len(lr.sessions) == comps.episodic.count(lr.learner_id)
    meta = comps.sessions.tenant_meta(lr.learner_id)
    assert meta["n_sessions"] == len(lr.sessions)
    assert meta["last_session_id"] == lr.sessions[-1].session_id


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ((4, 4, 4), (False, None)),
        ((3, 4, 4), (True, "SHALLOW")),  # cheapest sufficient route
        ((3, 3.5, 4.5), (True, "DEEP")),
        ((2, 2.5, 2.5), (False, None)),  # below margin
        ((1, 5, 3), (True, "SHALLOW")),
    ],
)
def test_decide_label(scores, expected):
    assert label_oracle.decide_label(*scores) == expected


def test_split_by_learner_is_disjoint_and_stable():
    ids = [f"edu-{i:03d}" for i in range(40)]
    s = label_oracle.split_of(ids)
    assert s == label_oracle.split_of(list(reversed(ids)))
    counts = {k: list(s.values()).count(k) for k in ("train", "val", "test")}
    assert counts == {"train": 24, "val": 8, "test": 8}


def test_locomo_date_parse():
    assert locomo.parse_dt("1:56 pm on 8 May, 2023") == datetime(2023, 5, 8, 13, 56, tzinfo=UTC)
    assert locomo.parse_dt("10:37 am on  27 June, 2023").hour == 10


@pytest.mark.skipif(not locomo.PATH.exists(), reason="LoCoMo not downloaded")
def test_locomo_probes_stratified():
    convs = locomo.load()
    ps = locomo.probes(convs[:2], per_conv=35)
    cats = {p.category for p in ps}
    assert "locomo_adversarial" in cats and "locomo_single_hop" in cats
    df = locomo.labels_frame(ps)
    assert (~df[df.category == "locomo_adversarial"].label_required).all()
    assert (df[df.category == "locomo_single_hop"].depth == "SHALLOW").all()
