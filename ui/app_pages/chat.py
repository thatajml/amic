"""Chat page: the tutor, the gate's decision for each answer, and the account sidebar."""

from __future__ import annotations

import uuid

import streamlit as st
from chat_common import (
    app_settings,
    get_stack,
    users_collection,
)

from amic.chatui import (
    CUSTOM,
    OLLAMA,
    PRESETS,
    PROVIDERS,
    decision_summary,
    find_trace,
    list_ollama_models,
    plain,
    save_name_to_profile,
)
from amic.writepath.commands import parse_command

ROUTE_COLOR = {"SKIP": "gray", "SHALLOW": "blue", "DEEP": "violet"}
EXAMPLES = [
    "What is a Python list?",
    "What did I get wrong last time?",
    "Can you give me an exercise that fits my project?",
]

user = st.session_state.get("user")  # (learner id, display name) once signed in
seeded_id = st.session_state.get("seeded")
mode = "user" if user else "seeded" if seeded_id else "guest"
ollama_url = app_settings().models.ollama_base_url


def sign_out() -> None:
    for k in ("user", "guest", "seeded", "who", "messages", "profile_saved"):
        st.session_state.pop(k, None)


# ----------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Account")
    if user:
        st.success(f"Signed in as **{plain(user[1])}**", icon=":material/account_circle:")
        st.caption("Your turns and learner profile are saved to your own memory.")
    elif mode == "seeded":
        st.info(f"Seeded learner **{seeded_id}**", icon=":material/science:")
    else:
        st.info("Guest: nothing is saved.", icon=":material/person:")
    st.button(
        "Sign out" if user else "Back to login",
        icon=":material/logout:",
        on_click=sign_out,
        width="stretch",
    )
    write = False
    if mode == "seeded":
        write = st.toggle(
            "Save new turns to long-term memory",
            value=False,
            help="Off keeps the seeded memory unchanged (safe for experiments).",
        )

    st.header("Tutor model")
    provider = st.selectbox("Provider", PROVIDERS)
    base_url = key_env = api_key = ""
    if provider == OLLAMA:
        base_url = ollama_url
        models = list_ollama_models(ollama_url)
        if models:
            model = st.selectbox(
                "Model", models, index=models.index("llama3.1:8b") if "llama3.1:8b" in models else 0
            )
        else:
            st.warning(f"Ollama is not reachable at {ollama_url}. Start it with `ollama serve`.")
            model = st.text_input("Model", "llama3.1:8b")
    else:
        preset = PRESETS[provider]
        base_url = st.text_input(
            "API base URL",
            preset.base_url,
            disabled=provider not in (CUSTOM, "LM Studio / vLLM (local)"),
        )
        model = st.text_input("Model", preset.default_model, placeholder="model name")
        key_env = st.text_input(
            "API key from env / .env", preset.key_env, help="Name of the variable holding the key."
        )
        api_key = st.text_input(
            "...or paste a key (this session only, never saved)", type="password"
        )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.header("Memory gate")
    arm = st.radio(
        "Arm",
        ["amic", "always", "never"],
        horizontal=True,
        help="amic = gated; always = retrieve every turn (DEEP); never = no memory.",
    )
    lam = st.select_slider(
        "λ (cost of a wrong skip ÷ cost of a needless retrieval)",
        [1.0, 2.0, 4.0, 8.0, 16.0],
        value=4.0,
        help="Higher λ = retrieve more readily. Threshold τ* = 1/(1+λ).",
    )
    st.caption(f"τ* = {1 / (1 + lam):.2f}")

# a fresh conversation whenever the learner changes
who = (mode, user[0] if user else seeded_id or "")
if st.session_state.get("who") != who:
    st.session_state.who = who
    st.session_state.messages = []
    st.session_state.sid = uuid.uuid4().hex[:8]
    st.session_state.newcomer = f"ui-guest-{uuid.uuid4().hex[:6]}"
learner = who[1] or st.session_state.newcomer
collection = users_collection() if user else "edu_v1"
write = True if user else write  # accounts always save; guests never do

# ----------------------------------------------------------------------------------- main
st.title(f"Hi, {plain(user[1])}" if user else "AMIC chat")
st.caption(
    "Any LLM answers; AMIC decides whether it gets the learner's memory "
    "(SKIP / SHALLOW / DEEP) before each turn."
)

try:
    comps, arms, tel = get_stack(
        provider, model, base_url, key_env, api_key, lam, temperature, collection
    )
except Exception as e:  # noqa: BLE001
    st.error(f"Could not start AMIC: {e}", icon=":material/error:")
    st.caption("If the memory services are down, start the stack with `up-amic.cmd`.")
    st.button("Retry", icon=":material/refresh:")
    st.stop()

if user and st.session_state.get("profile_saved") != user[0]:
    # the learner's M3 profile exists from the first visit (idempotent)
    save_name_to_profile(comps.profiles, user[0], user[1])
    st.session_state.profile_saved = user[0]

# In deployment the write path builds a learner's sketch ahead of time; build it on selection so
# the first turn's gate timing is not inflated by the one-off build (the experiment runner does
# the same).
comps.sketches.get(learner)


def show_panel(sm: dict) -> None:
    route = sm["route"]
    p = f"{sm['p']:.3f}" if sm.get("p") is not None else "–"
    band = f"[{sm['band'][0]:.2f}, {sm['band'][1]:.2f}]" if sm.get("band") else "–"
    st.markdown(
        f":{ROUTE_COLOR.get(route, 'gray')}-badge[{route}] decided by "
        f"**{sm['resolved_by']}** · p = {p} · band {band}"
    )
    with st.expander("Why, and what it cost"):
        st.write(sm["rationale"])
        if sm.get("latency_ms"):
            c = st.columns(4)
            c[0].metric("Memory tokens", sm["memory_tokens"])
            c[1].metric("Prompt tokens", sm["prompt_tokens"])
            c[2].metric("Gate ms", sm["latency_ms"]["gate"])
            c[3].metric("Total ms", f"{sm['latency_ms']['total']:.0f}")
            st.caption(" · ".join(f"{k} {v}" for k, v in sm["latency_ms"].items()))
        if sm.get("signals"):
            st.caption("Strongest signals: " + ", ".join(f"`{k}`={v}" for k, v in sm["signals"]))
        for it in sm["items"]:
            st.markdown(f"- `{it['score']:.2f}` {it['text'][:240]}")
        if not sm["items"]:
            st.caption("No memory retrieved.")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("summary"):
            show_panel(m["summary"])

msgs = st.session_state.messages
cols = st.columns(len(EXAMPLES))
clicked = next((q for c, q in zip(cols, EXAMPLES, strict=True) if c.button(q)), None)
query = st.chat_input("Ask the tutor...") or clicked
if query is not None and not query.strip():
    st.toast("Type a message first.", icon=":material/edit:")  # a blank turn is not sent or saved
    query = None
query = query.strip() if query else query
can_save = write and mode != "guest"

if query and parse_command(query) and not can_save:
    # /remember and /forget only act on saved memory: say so instead of promising and not saving
    msgs.append({"role": "user", "content": query})
    msgs.append(
        {
            "role": "assistant",
            "content": (
                "Guests cannot save memory, so `/remember` and `/forget` do nothing here. "
                "Go back to the login page and create an account to use them."
                if mode == "guest"
                else "Memory commands need saving to be on: turn on "
                "**Save new turns to long-term memory** in the sidebar first."
            ),
        }
    )
    st.rerun()

if query:
    from amic.app import make_context

    msgs.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.status(f":shimmer[{model} is thinking]", type="compact") as status:
            ctx = make_context(comps, learner, f"ui-{st.session_state.sid}-{learner}", query, arm)  # type: ignore[arg-type]
            try:
                resp, job = arms[arm].answer_deferred(ctx, write=can_save)
            except Exception as e:  # noqa: BLE001
                status.update(label="Failed", state="error")
                st.error(f"The tutor call failed: {e}")
                # keep the exchange visible (an error reply) instead of a dangling question
                msgs.append(
                    {"role": "assistant", "content": f"The tutor call failed: {plain(str(e))}"}
                )
                st.stop()
            job()  # shadow baseline for the dashboard; writes memory only when allowed above
            status.update(label="Answered", state="complete")
        sm = decision_summary(resp, find_trace(tel.traces, resp))
        st.markdown(resp.answer)
        show_panel(sm)
    msgs.append({"role": "assistant", "content": resp.answer, "summary": sm})

# session totals, after this turn's answer so they are never one turn behind
if msgs:
    n = sum(1 for m in msgs if m.get("summary"))
    skipped = sum(1 for m in msgs if m.get("summary", {}).get("route") == "SKIP")
    mem = sum(m.get("summary", {}).get("memory_tokens", 0) for m in msgs)
    st.sidebar.divider()
    st.sidebar.metric("Turns", n)
    st.sidebar.metric("Memory skipped", f"{skipped}/{n}")
    st.sidebar.metric("Memory tokens used", mem)
