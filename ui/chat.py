"""AMIC chat: talk to any LLM with AMIC deciding what memory goes into each prompt.

    streamlit run ui/chat.py

Entry point and router. Not signed in -> only the login page; signed in, guest or a seeded learner
-> the chat page (ui/app_pages/chat.py). Accounts save every turn and an M3 learner profile to
their own memory; guests save nothing; seeded learners are read-only unless you turn writes on, so
chatting never changes the memory the experiments use.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="AMIC chat", layout="wide")

signed = any(st.session_state.get(k) for k in ("user", "guest", "seeded"))
page = st.navigation(
    [
        st.Page(
            "app_pages/chat.py",
            title="Chat",
            icon=":material/chat:",
            default=signed,
            help="Talk to the tutor and see each memory-gate decision.",
        )
        if signed
        else st.Page(
            "app_pages/login.py",
            title="Login",
            icon=":material/login:",
            default=True,
            help="Sign in, create an account, or continue as a guest.",
        ),
    ],
    position="hidden",
)
page.run()
