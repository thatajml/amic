"""Educator Console (brief L1): what AMIC remembers about each learner, and the learner's
data rights. A thin Streamlit view over amic.educator.EducatorConsole.

    streamlit run ui/educator.py
    AMIC_CONFIG=configs/ablations/m3.yaml streamlit run ui/educator.py   # extra YAML, comma list

Reads the same stores as the service (Settings from .env / AMIC_* variables).
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from amic.app import build_components
from amic.contracts.config import load_settings
from amic.educator import EducatorConsole
from amic.memory.stores.profile import CONCEPT_PATTERNS

st.set_page_config(page_title="AMIC Educator Console", layout="wide")


@st.cache_resource
def console() -> EducatorConsole:
    cfg = [p for p in os.environ.get("AMIC_CONFIG", "").split(",") if p]
    s = load_settings(cfg or None)
    return EducatorConsole(build_components(s, llm=object()))  # type: ignore[arg-type]


con = console()
st.title("Educator Console")
learners = pd.DataFrame(con.learners())
if learners.empty:
    st.info("No learner has long-term memory yet.")
    st.stop()

with st.sidebar:
    st.header("Learners")
    query = st.text_input("Filter", "")
    shown = learners[learners.tenant_id.str.contains(query, case=False)] if query else learners
    tenant = st.selectbox("Learner", shown.tenant_id.tolist())
    st.dataframe(shown, hide_index=True, width="stretch")

row = learners.set_index("tenant_id").loc[tenant]
c1, c2, c3 = st.columns(3)
c1.metric("Episodes (M1)", int(row.episodes))
c2.metric("Current facts (M2)", int(row.facts))
c3.metric("Profile (M3)", "yes" if row.profile else "no", row.updated or None)

tab_p, tab_f, tab_e, tab_d, tab_r = st.tabs(
    ["Profile", "Facts", "Episodes", "Gate decisions", "Data rights"]
)

with tab_p:
    prof = con.profile(tenant)
    if prof is None:
        st.info("No profile yet: it is built by the consolidation worker as the learner chats.")
    else:
        st.subheader("What the tutor sees (M3 block)")
        st.code(prof["block"], language=None)
        m = pd.DataFrame(prof["mastery"])
        if not m.empty:
            st.subheader("Concept mastery (EWMA of evidence; 0.5 = no evidence)")
            st.bar_chart(m[m.evidence > 0].set_index("concept")["mastery"], horizontal=True)
            st.dataframe(m, hide_index=True, width="stretch")
        st.subheader("Open misconceptions")
        st.write(prof["misconceptions"] or "none")
        st.subheader("Notes")
        st.json(prof["notes"])
    with st.expander("Correct the profile"):
        kind = st.selectbox("Add a note", ["goal", "preference", "project", "personal"])
        text = st.text_input("Text", key="note")
        if st.button("Add note") and text.strip():
            con.add_note(tenant, kind, text.strip())
            st.rerun()
        concept = st.selectbox("Set mastery of", list(CONCEPT_PATTERNS))
        value = st.slider("Mastery", 0.0, 1.0, 0.5, 0.05)
        if st.button("Set mastery"):
            con.set_mastery(tenant, concept, value)
            st.rerun()

with tab_f:
    show_closed = st.toggle("Show closed facts (history)", value=False)
    facts = pd.DataFrame(con.facts(tenant, include_closed=show_closed))
    if facts.empty:
        st.info("No facts.")
    else:
        st.dataframe(facts.drop(columns=["id"]), hide_index=True, width="stretch")
        current = facts[facts.current]
        if not current.empty:
            pick = st.selectbox(
                "Close a wrong or outdated fact (kept for audit, never retrieved again)",
                current.id.tolist(),
                format_func=lambda i: current.set_index("id").loc[i, "text"],
            )
            if st.button("Close fact"):
                con.close_fact(tenant, pick)
                st.rerun()

with tab_e:
    st.dataframe(pd.DataFrame(con.episodes(tenant, 50)), hide_index=True, width="stretch")

with tab_d:
    dec = pd.DataFrame(con.decisions(tenant, 200))
    if dec.empty:
        st.info("No recorded decisions for this learner.")
    else:
        st.write(dec.route.value_counts(normalize=True).rename("share"))
        st.dataframe(dec, hide_index=True, width="stretch")

with tab_r:
    st.subheader("Forgetting (what tonight's job would do)")
    st.json(con.forgetting_preview(tenant))
    st.subheader("Erase this learner")
    st.caption("Deletes M1 episodes, M2 facts, the M3 profile and session state; the M4 sketch")
    st.caption("and cached gate decisions are invalidated. This cannot be undone.")
    if st.checkbox(f"I understand: erase everything about {tenant}") and st.button("Erase"):
        st.json(con.erase(tenant))
        st.cache_resource.clear()
