"""Chat UI accounts: name + PIN registry and the profile created for a new user."""

import sqlite3

import pytest

from amic.chatui import TeeTelemetry, UserRegistry, save_name_to_profile
from amic.memory.stores.profile import ProfileStore, SqliteKV


def test_register_then_login():
    r = UserRegistry(":memory:")
    uid = r.register("Ala Ajmal", "1234")
    assert uid.startswith("u-ala-ajmal-")
    assert r.login("  ala   AJMAL ", "1234") == (uid, "Ala Ajmal")


def test_wrong_pin_or_unknown_name_is_refused():
    r = UserRegistry(":memory:")
    r.register("Sam", "4321")
    assert r.login("Sam", "0000") is None
    assert r.login("Nobody", "4321") is None


@pytest.mark.parametrize("name,pin", [("", "1234"), ("   ", "1234"), ("Sam", "123")])
def test_bad_input_is_rejected(name, pin):
    with pytest.raises(ValueError):
        UserRegistry(":memory:").register(name, pin)


def test_duplicate_name_is_rejected_case_insensitively():
    r = UserRegistry(":memory:")
    r.register("Sam", "1234")
    with pytest.raises(ValueError, match="taken"):
        r.register("sam", "9999")


def test_pin_is_not_stored_in_plaintext(tmp_path):
    path = str(tmp_path / "u.sqlite")
    UserRegistry(path).register("Sam", "secret-pin")
    raw = b"".join(
        bytes(c) if isinstance(c, (bytes, bytearray)) else str(c).encode()
        for row in sqlite3.connect(path).execute("SELECT * FROM users")
        for c in row
    )
    assert b"secret-pin" not in raw


def test_name_note_creates_the_profile():
    profiles = ProfileStore(SqliteKV(":memory:"))
    assert profiles.get("u-sam-1") is None
    save_name_to_profile(profiles, "u-sam-1", "Sam")
    prof = profiles.get("u-sam-1")
    assert prof is not None and "Name is Sam" in prof.render()
    save_name_to_profile(profiles, "u-sam-1", "Sam")  # idempotent
    assert profiles.get("u-sam-1").notes["personal"] == {"name": "Name is Sam"}


def test_set_pin_keeps_the_learner_id():
    r = UserRegistry(":memory:")
    uid = r.register("Ala", "1111")
    assert r.set_pin(" ala ", "2222") == "Ala"
    assert r.login("Ala", "1111") is None
    assert r.login("Ala", "2222") == (uid, "Ala")
    with pytest.raises(ValueError, match="No account"):
        r.set_pin("Nobody", "2222")
    with pytest.raises(ValueError, match="at least"):
        r.set_pin("Ala", "1")


def test_admin_cli_lists_and_resets(tmp_path, capsys):
    from amic.chatui_admin import main

    db = str(tmp_path / "u.sqlite")
    uid = UserRegistry(db).register("Ala", "1111")
    assert main(["--db", db, "reset-pin", "Ala", "--pin", "9999"]) == 0
    assert UserRegistry(db).login("Ala", "9999") == (uid, "Ala")
    assert main(["--db", db, "reset-pin", "Nobody", "--pin", "9999"]) == 1
    capsys.readouterr()
    assert main(["--db", db, "list"]) == 0
    assert uid in capsys.readouterr().out


def test_tee_telemetry_feeds_both_stores_and_annotates_both():
    from datetime import UTC, datetime

    from amic.contracts.models import DecisionTrace, LatencySplit
    from amic.telemetry.store import InMemoryTelemetry, SqliteTelemetry

    mem, shared = InMemoryTelemetry(), SqliteTelemetry("sqlite:///:memory:")
    tee = TeeTelemetry(mem, shared)
    tr = DecisionTrace(
        request_id="r1",
        ts=datetime.now(UTC),
        tenant_id="u-1",
        session_id="s",
        query="hi",
        route="SKIP",
        resolved_by="tier1",
        arm="amic",
        config_hash="x",
        latency=LatencySplit(),
    )
    tee.record(tr)
    tee.annotate("r1", baseline_memory_tokens=123)
    assert tee.traces[0].baseline_memory_tokens == 123
    assert shared.fetch(limit=5)[0].baseline_memory_tokens == 123
