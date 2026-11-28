"""Login page of the chat UI: sign in, create an account, or continue as a guest."""

from __future__ import annotations

import streamlit as st
from chat_common import get_registry, learner_options

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
            if found := get_registry().login(name, pin):
                st.session_state.user = found
                st.rerun()
            st.error("Name or PIN is not right.")
    with tab_new, st.form("create_account"):
        name = st.text_input("Choose a name", max_chars=40)
        pin = st.text_input("Choose a PIN", type="password", help="At least 4 characters.")
        if st.form_submit_button("Create account", type="primary", width="stretch"):
            try:
                uid = get_registry().register(name, pin)
            except ValueError as e:
                st.error(str(e))
            else:
                st.session_state.user = (uid, " ".join(name.split()))
                st.rerun()
    st.caption("There is no PIN recovery: a forgotten PIN means a new account.")
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
