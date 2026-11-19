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
        assert client.delete("/learners/zoe").json() == {"deleted_records": 2}
        assert comps.episodic.count("zoe") == 0


def test_bad_request(comps):
    with TestClient(create_app(comps)) as client:
        assert (
            client.post(
                "/chat", json={"tenant_id": "", "session_id": "s", "query": "x"}
            ).status_code
            == 422
        )


def test_unknown_model_is_a_400_and_health_lists_models(comps):
    with TestClient(create_app(comps)) as client:
        assert client.get("/health").json()["models"] == [comps.llm.model]
        r = client.post(
            "/chat",
            json={
                "tenant_id": "a",
                "session_id": "s",
                "query": "hi",
                "arm": "never",
                "model": "nope",
            },
        )
        assert r.status_code == 400 and "not available" in r.json()["detail"]
