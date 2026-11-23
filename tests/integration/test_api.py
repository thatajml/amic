from __future__ import annotations

from fastapi.testclient import TestClient

from amic.service.api import create_app


def test_chat_health_and_background_write(comps):
    with TestClient(create_app(comps)) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post(
            "/chat",
            json={
                "tenant_id": "zoe",
                "session_id": "s1",
                "query": "what is a dict?",
                "arm": "always",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["decision"]["route"] == "DEEP" and body["trace_id"]
        # TestClient runs background tasks before returning: the turn is now in M1
        assert comps.episodic.count("zoe") == 1
        r2 = client.post(
            "/chat",
            json={"tenant_id": "zoe", "session_id": "s1", "query": "and a set?", "arm": "never"},
        )
        assert r2.json()["decision"]["route"] == "SKIP"
        # "what is a dict?" is evidence on a curriculum concept: the worker created an M3 profile
        assert comps.profiles.get("zoe") is not None
        assert client.delete("/learners/zoe").json() == {
            "deleted_records": 2,
            "deleted_facts": 0,
            "deleted_profile": 1,
        }
        assert comps.episodic.count("zoe") == 0 and comps.profiles.get("zoe") is None


def test_bad_request(comps):
    with TestClient(create_app(comps)) as client:
        assert (
            client.post(
                "/chat", json={"tenant_id": "", "session_id": "s", "query": "x"}
            ).status_code
            == 422
        )
