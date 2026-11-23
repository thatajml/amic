"""Educator Console backend: reads every tier of one learner, edits bump the memory version."""

from __future__ import annotations

from amic.educator import EducatorConsole
from amic.writepath.extract import ConflictResolver, ExtractedFact
from tests.conftest import T0


def test_console_reads_and_edits_one_learner(comps, seed_history):
    seed_history(comps, "ivy", [("my dict lookup keeps giving an error", "use .get()")])
    ConflictResolver(comps.semantic, comps.embedder).resolve(
        "ivy", [ExtractedFact("goal", "goal", "pass the exam", 0.9)], T0
    )
    con = EducatorConsole(comps)
    assert {r["tenant_id"] for r in con.learners()} == {"ivy"}
    assert con.profile("ivy") is None  # no worker ran: no profile yet
    con.add_note("ivy", "preference", "short worked examples")
    prof = con.profile("ivy")
    assert "short worked examples" in prof["block"]
    v = comps.sessions.mem_version("ivy")
    con.set_mastery("ivy", "dictionaries", 0.1)
    assert comps.sessions.mem_version("ivy") > v
    assert con.profile("ivy")["mastery"][0]["concept"] == "dictionaries"
    facts = con.facts("ivy")
    assert [f["text"] for f in facts] == ["pass the exam"] and facts[0]["current"]
    con.close_fact("ivy", facts[0]["id"])
    assert con.facts("ivy", include_closed=False) == []
    assert con.facts("ivy")[0]["closed_by"] == "educator"
    assert len(con.episodes("ivy")) == 1
    assert con.forgetting_preview("ivy")["episodes_compacted"] == 0
    out = con.erase("ivy")
    assert out["deleted_records"] == 1 and out["deleted_profile"] == 1
    assert con.learners() == []


def test_decisions_come_from_the_learner_traces(comps):
    from amic.app import build_arm, make_context

    build_arm("never", comps).answer(make_context(comps, "jo", "s", "what is a set?", "never"))
    build_arm("never", comps).answer(make_context(comps, "kim", "s", "what is a set?", "never"))
    rows = EducatorConsole(comps).decisions("jo")
    assert len(rows) == 1 and rows[0]["route"] == "SKIP"
