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


# ------------------------------------------------------------------ hardening (edge-case fixes)
def test_names_are_unicode_normalised_and_invisible_characters_dropped():
    from amic.chatui import normalize_name

    assert normalize_name("Zoé  Q") == normalize_name("Zoé Q") == "Zoé Q"
    assert normalize_name("Sam​") == "Sam"
    r = UserRegistry(":memory:")
    r.register("Zoé Q", "1234")
    with pytest.raises(ValueError, match="taken"):
        r.register("Zoé Q", "1234")  # same name typed with a combining accent
    r.register("Sam", "1234")
    with pytest.raises(ValueError, match="taken"):
        r.register("Sam​", "1234")  # an invisible character does not make a new name


def test_mixed_alphabet_and_overlong_names_are_rejected():
    r = UserRegistry(":memory:")
    with pytest.raises(ValueError, match="alphabet"):
        r.register("Sаm", "1234")  # Latin S + Cyrillic a
    with pytest.raises(ValueError, match="at most"):
        r.register("a" * 41, "1234")
    assert r.register("山田太郎", "1234")  # one script is fine


def test_pin_spaces_are_ignored_and_blank_pins_rejected():
    r = UserRegistry(":memory:")
    for bad in ("    ", " 12 ", "123 "):
        with pytest.raises(ValueError, match="at least"):
            r.register("X", bad)
    uid = r.register("Sam", " 1234 ")  # stored trimmed
    assert r.login("Sam", "1234") == (uid, "Sam")
    assert r.login("Sam", "1234  ") == (uid, "Sam")  # a pasted trailing space still works


def test_legacy_pin_with_spaces_still_logs_in():
    import hashlib
    import secrets

    r = UserRegistry(":memory:")
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(b"12 34", salt=salt, n=2**14, r=8, p=1)
    r._db.execute("INSERT INTO users VALUES ('u-old-1','Old','old',?,?,0)", (salt, h))
    assert r.login("Old", "12 34") == ("u-old-1", "Old")


def test_lockout_after_repeated_wrong_pins_and_unlock():
    from amic.chatui import LOCK_SECONDS, MAX_FAILS, LoginLocked

    r = UserRegistry(":memory:")
    now = [1000.0]
    r._clock = lambda: now[0]
    r.register("Sam", "1234")
    for _ in range(MAX_FAILS):
        assert r.login("Sam", "0000") is None
    with pytest.raises(LoginLocked) as e:
        r.login("Sam", "1234")  # even the right PIN is refused while locked
    assert 0 < e.value.seconds <= LOCK_SECONDS + 1
    now[0] += LOCK_SECONDS + 1
    assert r.login("Sam", "1234") is not None  # lock expired
    for _ in range(MAX_FAILS):
        r.login("Sam", "0000")
    r.unlock("sam")
    assert r.login("Sam", "1234") is not None
    # unknown names are throttled the same way (no hint whether an account exists)
    for _ in range(MAX_FAILS):
        r.login("Ghost", "0000")
    with pytest.raises(LoginLocked):
        r.login("Ghost", "0000")


def test_a_correct_login_resets_the_failure_count():
    from amic.chatui import MAX_FAILS

    r = UserRegistry(":memory:")
    r.register("Sam", "1234")
    for _ in range(MAX_FAILS - 1):
        r.login("Sam", "0000")
    assert r.login("Sam", "1234") is not None
    for _ in range(MAX_FAILS - 1):
        assert r.login("Sam", "0000") is None  # counter started over: still not locked
    assert r.login("Sam", "1234") is not None


def test_plain_escapes_markdown():
    from amic.chatui import plain

    out = plain("[x](http://e.example) **b** :red[hi] # H")
    for ch in "[]()*:#":
        assert all(out[i - 1] == "\\" for i, c in enumerate(out) if c == ch and i)
    assert plain("Sam") == "Sam"


def test_tee_telemetry_keeps_a_bounded_panel_list():
    from datetime import UTC, datetime

    from amic.contracts.models import DecisionTrace, LatencySplit
    from amic.telemetry.store import InMemoryTelemetry

    tee = TeeTelemetry(InMemoryTelemetry(), InMemoryTelemetry())
    for i in range(2 * TeeTelemetry.KEEP + 50):
        tee.record(
            DecisionTrace(
                request_id=f"r{i}",
                ts=datetime.now(UTC),
                tenant_id="u",
                session_id="s",
                query="q",
                route="SKIP",
                resolved_by="tier1",
                arm="amic",
                config_hash="x",
                latency=LatencySplit(),
            )
        )
    assert len(tee.traces) <= 2 * TeeTelemetry.KEEP
    assert tee.traces[-1].request_id == f"r{2 * TeeTelemetry.KEEP + 49}"


def test_admin_unlock_and_chat_prompt_and_replies():
    from amic.generation.prompt import PromptBuilder
    from amic.writepath.commands import Command, acknowledgement
    from amic.writepath.writer import record_text

    assert "never guess" in PromptBuilder.for_profile("tutor_chat").system
    assert "never guess" not in PromptBuilder.for_profile("tutor").system  # experiments unchanged
    assert "close" in acknowledgement(Command("forget", "teal"))
    assert len(record_text("q" * 50_000, "a")) < 2_200
