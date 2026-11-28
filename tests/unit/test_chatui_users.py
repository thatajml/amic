"""Chat UI accounts: name + PIN registry and the profile created for a new user."""

import sqlite3

import pytest

from amic.chatui import UserRegistry, save_name_to_profile
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
