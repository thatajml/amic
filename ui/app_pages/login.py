"""Login page of the chat UI: sign in, create an account, or continue as a guest."""

from __future__ import annotations

import streamlit as st
from chat_common import (
    DASHBOARD_URL,
    EDUCATOR_URL,
    get_registry,
    learner_options,
    stack_problem,
)

from amic.chatui import LoginLocked, normalize_name

_, centre, _ = st.columns([1, 2, 1])
with centre:
    st.title("Welcome to AMIC")
    st.caption(
        "Sign in so the tutor remembers you across sessions: your turns and learner profile are "
        "saved to your own memory."
    )
    tab_in, tab_new = st.tabs(["Sign in", "Create account"])
    with tab_in, st.form("sign_in"):
        name = st.text_input("Name", max_chars=40)
        pin = st.text_input("PIN", type="password")
        if st.form_submit_button("Sign in", type="primary", width="stretch"):
            try:
                found = get_registry().login(name, pin)
            except LoginLocked as e:
                st.error(
                    f"Too many wrong PINs. Try again in {(e.seconds + 59) // 60} min, "
                    "or ask staff to unlock the account."
                )
            else:
                if found:
                    st.session_state.user = found
                    st.rerun()
                st.error("Name or PIN is not right. If you forgot your PIN, ask staff to reset it.")
    with tab_new, st.form("create_account"):
        name = st.text_input("Choose a name", max_chars=40)
        pin = st.text_input("Choose a PIN", type="password", help="At least 4 characters.")
        pin2 = st.text_input("Repeat the PIN", type="password")
        st.caption("Remember your PIN: only staff can reset it.")
        if st.form_submit_button("Create account", type="primary", width="stretch"):
            try:
                if pin.strip() != pin2.strip():
                    raise ValueError("The two PINs are different. Type the same PIN twice.")
                uid = get_registry().register(name, pin)
            except ValueError as e:
                st.error(str(e))
            else:
                st.session_state.user = (uid, normalize_name(name))
                st.rerun()
    if problem := stack_problem():
        st.warning(
            "The memory services are not reachable, so chatting will fail until they are "
            "started (`up-amic.cmd`).",
            icon=":material/cloud_off:",
        )
        st.caption(problem)
    st.divider()
    if st.button("Continue as guest", width="stretch", help="Nothing is saved."):
        st.session_state.guest = True
        st.rerun()
    if opts := learner_options():
        with st.expander("Seeded learners (research data)"):
            label = st.selectbox("Seeded learner", list(opts))
            if st.button("Use this learner"):
                st.session_state.seeded = opts[label]
                st.rerun()

    st.divider()
    st.caption("Staff")
    s1, s2 = st.columns(2)
    s1.link_button("Dashboard", DASHBOARD_URL, icon=":material/monitoring:", width="stretch")
    s2.link_button("Educator Console", EDUCATOR_URL, icon=":material/school:", width="stretch")
