"""AMIC chat: talk to any LLM with AMIC deciding what memory goes into each prompt.

    streamlit run ui/chat.py

Pick a tutor model (local Ollama or any OpenAI-compatible API), a learner and an arm. Every answer
shows the gate's decision: route, which tier decided, p against the threshold band, signals,
retrieved memory, latency split and tokens. Memory is read-only unless you turn writes on, so
chatting never changes the seeded memory the experiments use.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from amic.chatui import (
    CUSTOM,
    OLLAMA,
    PRESETS,
    PROVIDERS,
    USERS_COLLECTION,
    UserRegistry,
    decision_summary,
    find_trace,
    list_ollama_models,
    make_gateway,
    save_name_to_profile,
)

st.set_page_config(page_title="AMIC chat", layout="wide")

ROUTE_COLOR = {"SKIP": "gray", "SHALLOW": "blue", "DEEP": "violet"}
EXAMPLES = [
    "What is a Python list?",
    "What did I get wrong last time?",
    "Can you give me an exercise that fits my project?",
]
SIGN_IN, CREATE, GUEST, SEEDED = "Sign in", "Create account", "Guest", "Seeded learner"


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embedder(name: str, device: str):  # noqa: ANN201
    from amic.memory.embedder import STEmbedder

    return STEmbedder(name, device)


@st.cache_resource(show_spinner="Starting AMIC...")
def get_stack(
    provider: str,
    model: str,
    base_url: str,
    key_env: str,
    api_key: str,
    lam: float,
    temperature: float,
    collection: str,
):  # noqa: ANN201
    from amic.app import build_arm, build_components
    from amic.contracts.config import load_settings
    from amic.telemetry.store import InMemoryTelemetry

    s = load_settings(None, chroma_collection=collection, gate={"lam": lam})
    llm = make_gateway(
        provider,
        model,
        base_url=base_url,
        key_env=key_env,
        api_key=api_key,
        temperature=temperature,
        max_tokens=s.models.max_answer_tokens,
    )
    tel = InMemoryTelemetry()
    comps = build_components(
        s,
        embedder=get_embedder(s.models.embedding, s.models.embedding_device),
        llm=llm,
        telemetry=tel,
    )
    arms = {name: build_arm(name, comps) for name in ("amic", "always", "never")}  # type: ignore[arg-type]
    return comps, arms, tel


@st.cache_resource
def get_registry() -> UserRegistry:
    return UserRegistry()


def learner_options() -> dict[str, str]:
    """Label -> learner id. Seeded learners come from data/edu/learners.jsonl if present."""
    opts: dict[str, str] = {}
    if Path("data/edu/learners.jsonl").exists():
        from amic.eval.datasets import edu

        for lr in edu.load_learners():
            opts[f"{lr.learner_id} · {lr.level} · {lr.goal[:40]}"] = lr.learner_id
    return opts


# ----------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Tutor model")
    provider = st.selectbox("Provider", PROVIDERS)
    base_url = key_env = api_key = ""
    if provider == OLLAMA:
        models = list_ollama_models()
        if models:
            model = st.selectbox(
                "Model", models, index=models.index("llama3.1:8b") if "llama3.1:8b" in models else 0
            )
        else:
            st.warning("Ollama is not reachable at localhost:11434. Start it with `ollama serve`.")
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

    st.header("Learner")
    user = st.session_state.get("user")  # (learner id, display name) once signed in
    opts = learner_options()
    write = False
    label = ""
    if user:
        mode = "user"
        st.success(f"Signed in as **{user[1]}**", icon=":material/account_circle:")
        st.caption("Your turns and profile are saved to your own memory.")
        if st.button("Sign out", icon=":material/logout:"):
            del st.session_state["user"]
            st.rerun()
    else:
        mode = (
            st.segmented_control(
                "Who is chatting",
                [SIGN_IN, CREATE, GUEST, *([SEEDED] if opts else [])],
                default=SIGN_IN,
                label_visibility="collapsed",
            )
            or GUEST
        )
        if mode in (SIGN_IN, CREATE):
            with st.form(f"auth_{mode}"):
                name = st.text_input("Name", max_chars=40)
                pin = st.text_input("PIN", type="password", help="At least 4 characters.")
                if st.form_submit_button(mode, width="stretch"):
                    try:
                        if mode == CREATE:
                            uid = get_registry().register(name, pin)
                            dn = " ".join(name.split())
                            profiles = get_stack(
                                provider,
                                model,
                                base_url,
                                key_env,
                                api_key,
                                4.0,
                                0.0,
                                USERS_COLLECTION,
                            )[0].profiles
                            save_name_to_profile(profiles, uid, dn)
                            st.session_state.user = (uid, dn)
                        elif found := get_registry().login(name, pin):
                            st.session_state.user = found
                        else:
                            st.error("Name or PIN is not right.")
                        if "user" in st.session_state:
                            st.rerun()
                    except ValueError as e:
                        st.error(str(e))
        elif mode == SEEDED:
            label = st.selectbox("Seeded learner", list(opts))
            write = st.toggle(
                "Save new turns to long-term memory",
                value=False,
                help="Off keeps the seeded memory unchanged (safe for experiments).",
            )
        else:
            st.caption("A guest has no memory and nothing is saved.")

# a fresh conversation whenever the learner changes
who = user[0] if user else (opts.get(label, "") if mode == SEEDED else "")
if st.session_state.get("who") != (mode, who):
    st.session_state.who = (mode, who)
    st.session_state.messages = []
    st.session_state.sid = uuid.uuid4().hex[:8]
    st.session_state.newcomer = f"ui-guest-{uuid.uuid4().hex[:6]}"
learner = who or st.session_state.newcomer
collection = USERS_COLLECTION if user else "edu_v1"
write = True if user else write  # accounts always save; guests never do

# ----------------------------------------------------------------------------------- main
st.title(f"Hi, {user[1]}" if user else "AMIC chat")
st.caption(
    "Any LLM answers; AMIC decides whether it gets the learner's memory "
    "(SKIP / SHALLOW / DEEP) before each turn."
)

try:
    comps, arms, tel = get_stack(
        provider, model, base_url, key_env, api_key, lam, temperature, collection
    )
except Exception as e:  # noqa: BLE001
    st.error(f"Could not start the tutor model: {e}")
    st.stop()

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

if query:
    from amic.app import make_context

    msgs.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.status(f":shimmer[{model} is thinking]", type="compact") as status:
            ctx = make_context(comps, learner, f"ui-{st.session_state.sid}-{learner}", query, arm)  # type: ignore[arg-type]
            try:
                resp, job = arms[arm].answer_deferred(ctx)
            except Exception as e:  # noqa: BLE001
                status.update(label="Failed", state="error")
                st.error(f"The tutor call failed: {e}")
                st.stop()
            if write and (user or mode == SEEDED):  # guests are never written
                job()
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
