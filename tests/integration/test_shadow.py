"""Shadow mode: the shadow arm replays each request on the same context, writes nothing, and
logs a comparison row; the served response is unchanged."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from amic.service.api import create_app


def test_shadow_runs_after_the_response_and_never_writes(comps, seed_history, tmp_path):
    comps.settings.shadow.enabled = True
    comps.settings.shadow.log_path = str(tmp_path / "shadow.jsonl")
    seed_history(comps, "una", [("I keep mixing up append and extend", "...")])
    with TestClient(create_app(comps)) as client:
        r = client.post(
            "/chat",
            json={
                "tenant_id": "una",
                "session_id": "live",
                "query": "what is a tuple?",
                "arm": "never",
            },
        )
        assert r.status_code == 200 and r.json()["decision"]["route"] == "SKIP"
    rows = [json.loads(x) for x in (tmp_path / "shadow.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["served"]["route"] == "SKIP" and row["shadow"]["route"] == "DEEP"
    assert row["shadow"]["n_items"] >= 1  # the shadow saw the seeded memory, not this turn
    # exactly one turn pair in M0 (the served one) and one new M1 record (the served write)
    assert len(comps.sessions.get_turns("live")) == 2
    assert comps.episodic.count("una") == 2
    arms = [t.arm for t in comps.telemetry.traces]
    assert arms == ["never", "shadow-always"]
