"""Chat UI navigation: a login page first, the chat page once signed in or a guest."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ENTRY = str(Path(__file__).resolve().parents[2] / "ui" / "chat.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("AMIC_CHAT_USERS_DB", str(tmp_path / "users.sqlite"))
    monkeypatch.setenv("AMIC_SESSION_BACKEND", "sqlite")
    monkeypatch.setenv("AMIC_SESSION_DB", str(tmp_path / "sessions.sqlite"))
    monkeypatch.setenv("AMIC_PROFILE__DB", str(tmp_path / "profiles.sqlite"))
    monkeypatch.setenv("AMIC_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("AMIC_CHROMA_CLIENT", "embedded")
    monkeypatch.setenv("AMIC_WRITEPATH__QUEUE", "inline")
    st.cache_resource.clear()  # the registry and stack are cached per process
    return AppTest.from_file(ENTRY, default_timeout=120).run()


def _titles(at):
    return [t.value for t in at.title]


def test_fresh_session_shows_only_the_login_page(app):
    assert not app.exception
    assert _titles(app) == ["Welcome to AMIC"]
    assert [t.label for t in app.tabs] == ["Sign in", "Create account"]
    assert not app.sidebar.selectbox  # no model settings before login


def test_wrong_pin_stays_on_the_login_page(app):
    next(t for t in app.text_input if t.label == "Choose a name").set_value("Sam")
    next(t for t in app.text_input if t.label == "Choose a PIN").set_value("1234")
    next(t for t in app.text_input if t.label == "Repeat the PIN").set_value("1234")
    next(b for b in app.button if b.label == "Create account").click().run()
    assert _titles(app) == ["Hi, Sam"]
    next(b for b in app.sidebar.button if b.label == "Sign out").click().run()
    next(t for t in app.text_input if t.label == "Name").set_value("Sam")
    next(t for t in app.text_input if t.label == "PIN").set_value("0000")
    next(b for b in app.button if b.label == "Sign in").click().run()
    assert app.error[0].value.startswith("Name or PIN is not right")
    assert _titles(app) == ["Welcome to AMIC"]
    next(t for t in app.text_input if t.label == "PIN").set_value("1234")
    next(b for b in app.button if b.label == "Sign in").click().run()
    assert _titles(app) == ["Hi, Sam"]


def test_account_lands_on_the_chat_page_and_sign_out_returns(app):
    next(t for t in app.text_input if t.label == "Choose a name").set_value("Sam")
    next(t for t in app.text_input if t.label == "Choose a PIN").set_value("1234")
    next(t for t in app.text_input if t.label == "Repeat the PIN").set_value("1234")
    next(b for b in app.button if b.label == "Create account").click().run()
    assert not app.exception
    assert _titles(app) == ["Hi, Sam"]
    next(b for b in app.sidebar.button if b.label == "Sign out").click().run()
    assert _titles(app) == ["Welcome to AMIC"]


def test_guest_gets_the_chat_page_without_saving(app):
    next(b for b in app.button if b.label == "Continue as guest").click().run()
    assert not app.exception
    assert _titles(app) == ["AMIC chat"]
    assert "Guest" in app.sidebar.info[0].value


def test_mismatched_pins_do_not_create_an_account(app):
    next(t for t in app.text_input if t.label == "Choose a name").set_value("Sam")
    next(t for t in app.text_input if t.label == "Choose a PIN").set_value("1234")
    next(t for t in app.text_input if t.label == "Repeat the PIN").set_value("1243")
    next(b for b in app.button if b.label == "Create account").click().run()
    assert "different" in app.error[0].value
    assert "user" not in app.session_state


def test_staff_links_are_on_the_login_page_only(app):
    assert [b.label for b in app.get("link_button")] == ["Dashboard", "Educator Console"]
    next(b for b in app.button if b.label == "Continue as guest").click().run()
    assert not app.get("link_button")
