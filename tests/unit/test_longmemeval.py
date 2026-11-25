from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from amic.eval.datasets import longmemeval as lme


def _q(qid: str, qtype: str, answer: object = "Paris") -> dict:
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": "Where did I go?",
        "question_date": "2023/05/30 (Tue) 23:40",
        "answer": answer,
        "answer_session_ids": ["s1"],
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 10:00"],
        # the same session id twice, as in the real data
        "haystack_session_ids": ["s1", "s1"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I went to Paris"},
                {"role": "assistant", "content": "Nice!"},
            ],
            [{"role": "user", "content": "hello"}],
        ],
    }


def test_parse_date():
    assert lme.parse_date("2023/05/30 (Tue) 23:40") == datetime(2023, 5, 30, 23, 40, tzinfo=UTC)


def test_records_have_unique_ids_even_with_repeated_session_ids():
    recs = list(lme.records(_q("a1", "single-session-user")))
    ids = [r[0] for r in recs]
    assert len(ids) == 3 and len(set(ids)) == 3
    assert recs[0][1] == "User: I went to Paris"


def test_labels_depth_and_abstention():
    qs = [
        _q("a1", "single-session-user"),
        _q("a2", "multi-session"),
        _q("a3_abs", "temporal-reasoning"),
    ]
    df = lme.labels_frame(qs).set_index("probe_id")
    assert df.loc["lme-a1", "depth"] == "SHALLOW" and bool(df.loc["lme-a1", "label_required"])
    assert df.loc["lme-a2", "depth"] == "DEEP"
    assert not df.loc["lme-a3_abs", "label_required"] and pd.isna(df.loc["lme-a3_abs", "depth"])


def test_gold_is_text_and_meta_uses_the_last_session_end():
    q = _q("a1", "knowledge-update", answer=3)
    assert lme.gold(q) == "3"
    meta = lme.session_meta(q)
    assert meta["n_sessions"] == 2.0
    end = datetime(2023, 5, 21, 10, 1, tzinfo=UTC)  # 1 turn -> +1 minute
    assert meta["last_seen"] == end.timestamp()


def test_subset_keeps_every_abstention_question():
    qs = [_q(f"x{i}", t) for i, t in enumerate(lme.TYPES * 3)] + [_q("y_abs", "multi-session")]
    sub = lme.stratified_subset(qs, per_type=2)
    assert "y_abs" in sub and len(sub) == 1 + 2 * len(lme.TYPES)
