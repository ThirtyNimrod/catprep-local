import streamlit as st
import sys
import runpy
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Suppress noisy "missing ScriptRunContext" warnings from the self-launch path
import logging
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from agents.graph import build_graph
from utils.kg_visualizer import (
    build_pyvis_html,
    get_graph_stats,
    filter_triples,
    get_unique_values,
)
import streamlit.components.v1 as components
import os

# --- Page Config ---
st.set_page_config(page_title="CAT Prep Assistant", layout="wide", page_icon="🐱")

# --- App State Initialization ---
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.app_graph = build_graph(memory=st.session_state.memory)
    st.session_state.thread_id = "catprep_session_user_1"
    st.session_state.config = {"configurable": {"thread_id": st.session_state.thread_id}}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hello! I am your CAT Prep Assistant. I can help you with study plans, practice questions, or mock test feedback.", "avatar": "🐱"}
    ]

if "current_graph_triples" not in st.session_state:
    st.session_state.current_graph_triples = []

if "all_graph_triples" not in st.session_state:
    st.session_state.all_graph_triples = []

# --- Custom CSS ---
st.markdown("""
<style>
    .stChatInput { padding-bottom: 0.5rem; }
    .title-text { font-size: 2rem; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 0.5rem; margin-top: -1rem; }
    .block-container { padding-top: 1rem; padding-bottom: 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-text">CAT Prep Assistant 📚</div>', unsafe_allow_html=True)

# --- Layout: Two Columns (KG left, Chat right) ---
col_graph, col_chat = st.columns([1, 1.2])

with col_chat:
    st.subheader("💬 Chat")
    
    # Render chat history
    chat_container = st.container(height=450)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar=msg.get("avatar")):
                st.markdown(msg["content"])
                
    # Input box
    if prompt := st.chat_input("Ask for a study plan, practice questions..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👤"})
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
                
        # Run backend
        with chat_container:
            with st.chat_message("ai", avatar="🐱"):
                status_placeholder = st.empty()
                status_placeholder.markdown("🧠 *Thinking...*")
                
                inputs = {"messages": [HumanMessage(content=prompt)]}
                final_response = "I couldn't generate a response."
                
                try:
                    for event in st.session_state.app_graph.stream(inputs, config=st.session_state.config, stream_mode="values"):
                        # Extract UI messages
                        msgs = event.get("messages", [])
                        if msgs and isinstance(msgs[-1], AIMessage):
                            final_response = msgs[-1].content
                        
                        # Extract graph triples (accumulate across turns)
                        if "active_graph_context" in event:
                            triples = event["active_graph_context"]
                            if triples:
                                st.session_state.current_graph_triples = triples
                                # Accumulate — deduplicate by converting to set of tuples
                                existing = {tuple(t) for t in st.session_state.all_graph_triples}
                                for t in triples:
                                    tt = tuple(t)
                                    if tt not in existing:
                                        st.session_state.all_graph_triples.append(list(t))
                                        existing.add(tt)
                                
                    status_placeholder.markdown(final_response)
                    st.session_state.messages.append({"role": "ai", "content": final_response, "avatar": "🐱"})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    status_placeholder.markdown(f"**{error_msg}**")
                    st.session_state.messages.append({"role": "ai", "content": error_msg, "avatar": "🐱"})
                    
        st.rerun()

with col_graph:
    st.subheader("🕸️ Knowledge Graph")

    # Use accumulated triples across turns
    all_triples = [tuple(t) for t in st.session_state.all_graph_triples]

    if all_triples:
        # --- Graph Stats ---
        stats = get_graph_stats(all_triples)
        m1, m2, m3 = st.columns(3)
        m1.metric("Nodes", stats["node_count"])
        m2.metric("Edges", stats["edge_count"])
        m3.metric("Predicates", len(stats["predicates"]))

        # --- Triplet Filters (inspired by yFiles filter-panel) ---
        subjects, predicates, objects = get_unique_values(all_triples)

        with st.expander("🔍 Triplet Filters", expanded=False):
            fc1, fc2, fc3 = st.columns(3)
            sel_subject = fc1.selectbox("Subject", [""] + subjects, format_func=lambda x: "(Any)" if x == "" else x)
            sel_predicate = fc2.selectbox("Predicate", [""] + predicates, format_func=lambda x: "(Any)" if x == "" else x)
            sel_object = fc3.selectbox("Object", [""] + objects, format_func=lambda x: "(Any)" if x == "" else x)

        # --- Search / Highlight ---
        search_node = st.text_input("🔎 Highlight node", placeholder="Type a node name...")

        # Apply filters
        display_triples = filter_triples(
            all_triples,
            subject=sel_subject or None,
            predicate=sel_predicate or None,
            object_=sel_object or None,
        )

        if display_triples:
            try:
                html = build_pyvis_html(
                    display_triples,
                    highlight_node=search_node or None,
                    height="520px",
                )
                components.html(html, height=560, scrolling=True)
            except Exception as e:
                st.error(f"Could not render graph: {e}")
        else:
            st.warning("No triples match the current filters.")

        # --- Clear Graph Button ---
        if st.button("🗑️ Clear Graph"):
            st.session_state.all_graph_triples = []
            st.session_state.current_graph_triples = []
            st.rerun()
    else:
        st.info("Ask a domain-specific question to see the AI retrieve context from the Knowledge Graph!")


# --- Self-launch: `python main_ui.py` starts Streamlit ---
# Guard: only run the launcher when NOT already inside a Streamlit runtime.
_inside_streamlit = False
try:
    from streamlit.runtime import exists as _st_runtime_exists
    _inside_streamlit = _st_runtime_exists()
except Exception:
    pass

if __name__ == "__main__" and not _inside_streamlit:
    _orig_get_event_loop = asyncio.get_event_loop

    def _patched_get_event_loop():
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    asyncio.get_event_loop = _patched_get_event_loop
    sys.argv = ["streamlit", "run", __file__]
    runpy.run_module("streamlit", run_name="__main__")

