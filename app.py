"""
Human-in-the-Loop Tree-of-Thought Streamlit App (Clean Version)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import cohere
from typing import List, Dict

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Tree-of-Thought Decision Support",
    page_icon="🌳",
    layout="wide"
)

# -----------------------------
# COHERE CLIENT (SAFE INIT)
# -----------------------------
@st.cache_resource
def get_cohere_client():
    return cohere.Client(st.secrets["COHERE_API_KEY"])

co = get_cohere_client()

# -----------------------------
# IMPORT MODULES
# -----------------------------
from rag_engine import RAGEngine
from tot_engine import ToTEngine
from scorer import BranchScorer
from confidence import ConfidenceCalculator
from memory_manager import MemoryManager
from tree_visualizer import TreeVisualizer
from utils import validate_query


# -----------------------------
# SESSION STATE INIT
# -----------------------------
def init_state():
    defaults = {
        "rag_engine": RAGEngine(),
        "tot_engine": ToTEngine(),
        "scorer": BranchScorer(),
        "confidence_calc": ConfidenceCalculator(),
        "memory_manager": MemoryManager(),
        "visualizer": TreeVisualizer(),

        "branches": [],
        "branch_scores": [],
        "score_details": [],
        "branch_confidences": [],
        "selected_branch": None,
        "recommended_branch": None,

        "context": None,
        "question": None,
        "document_ready": False
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# -----------------------------
# SIDEBAR
# -----------------------------
def sidebar():
    with st.sidebar:
        st.title("📄 Document")

        file = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])

        if file and not st.session_state.document_ready:
            with st.spinner("Processing..."):
                chunks = st.session_state.rag_engine.process_document(file)
                st.session_state.document_ready = True
                st.success(f"Loaded {len(chunks)} chunks")

        st.divider()
        st.subheader("💾 Memory")

        sessions = st.session_state.memory_manager.list_sessions()
        if sessions:
            st.dataframe(pd.DataFrame(sessions), use_container_width=True)

        if st.button("Clear Memory"):
            st.session_state.memory_manager.clear_memory()
            st.success("Cleared!")
            st.rerun()


# -----------------------------
# QUERY INPUT
# -----------------------------
def query_box():
    st.title("🌳 Tree-of-Thought Reasoning System")

    q = st.text_area("Ask your question", height=100)
    qtype = st.radio("Type", ["Yes/No", "Open-ended"], horizontal=True)

    return q, qtype


# -----------------------------
# BRANCH GENERATION
# -----------------------------
def generate_branches(question, context, qtype):
    engine = st.session_state.tot_engine
    engine.create_tree(question, context)

    if qtype == "Yes/No":
        return list(engine.generate_yes_no_branches(question, context).values())
    else:
        return engine.generate_open_ended_branches(question, context, num_branches=3)


# -----------------------------
# SCORING
# -----------------------------
def score_branches(branches, context, question):
    scorer = st.session_state.scorer
    engine = st.session_state.tot_engine

    scores, details = [], []

    for root in branches:
        nodes = engine.get_branch_nodes(root)
        node_objs = [engine.get_node(n) for n in nodes]

        result = scorer.score_branch(node_objs, context, question)

        scores.append(result["total_score"])
        details.append(result["breakdown"])

    return scores, details


# -----------------------------
# CONFIDENCE
# -----------------------------
def compute_confidence(scores):
    calc = st.session_state.confidence_calc
    conf = calc.calculate_confidence(scores)
    best = calc.get_recommended_branch(scores)
    return conf, best


# -----------------------------
# BRANCH UI
# -----------------------------
def show_branches(branches, scores, confs, recommended):
    st.subheader("🌿 Reasoning Branches")

    cols = st.columns(len(branches))

    for i, (b, s, c) in enumerate(zip(branches, scores, confs)):
        with cols[i]:

            st.markdown(f"### Branch {i+1}")
            node = st.session_state.tot_engine.get_node(b)

            if node:
                st.write(node.text[:180] + "...")

            st.metric("Score", round(s, 3))
            st.metric("Confidence", f"{c:.1f}%")

            if i == recommended:
                st.success("🏆 Recommended")

            if st.button(f"Select {i+1}", key=f"s{i}"):
                st.session_state.selected_branch = i

            if st.button(f"Expand {i+1}", key=f"e{i}"):
                st.session_state.tot_engine.expand_branch(
                    b,
                    st.session_state.question,
                    st.session_state.context
                )
                st.rerun()


# -----------------------------
# SCORE BREAKDOWN
# -----------------------------
def breakdown():
    i = st.session_state.selected_branch
    if i is None:
        return

    st.subheader("📊 Score Breakdown")

    d = st.session_state.score_details[i]

    cols = st.columns(4)
    keys = ["relevance", "depth", "logic", "evidence"]

    for j, k in enumerate(keys):
        with cols[j]:
            st.metric(k, d[k]["score"])


# -----------------------------
# TREE VIEW (simple placeholder)
# -----------------------------
def tree_view():
    st.subheader("🌳 Tree View")

    tree = st.session_state.tot_engine.get_tree_structure()

    if len(tree.nodes) == 0:
        st.info("No tree yet")
    else:
        st.success("Tree generated")


# -----------------------------
# FINAL DECISION
# -----------------------------
def decision():
    i = st.session_state.selected_branch
    if i is None:
        return

    st.subheader("✅ Final Decision")

    conf = st.session_state.branch_confidences[i]

    st.info(f"Selected Branch: {i+1} | Confidence: {conf:.1f}%")

    if st.button("Finalize Decision"):
        st.session_state.memory_manager.save_reasoning_session(
            query=st.session_state.question,
            context=st.session_state.context,
            branches=[{"id": b} for b in st.session_state.branches],
            scores={},
            confidences={},
            selected_branch=i,
            final_decision=f"Branch {i+1}"
        )
        st.success("Saved!")


# -----------------------------
# MAIN
# -----------------------------
def main():
    init_state()
    sidebar()

    q, qtype = query_box()

    if st.button("Generate Reasoning", disabled=not st.session_state.document_ready):

        if not validate_query(q):
            st.error("Invalid question")
            return

        with st.spinner("Thinking..."):
            context = st.session_state.rag_engine.get_context_for_query(q)

            st.session_state.context = context
            st.session_state.question = q

            branches = generate_branches(q, context, qtype)
            scores, details = score_branches(branches, context, q)
            confs, best = compute_confidence(scores)

            st.session_state.branches = branches
            st.session_state.branch_scores = scores
            st.session_state.score_details = details
            st.session_state.branch_confidences = confs
            st.session_state.recommended_branch = best

    if st.session_state.branches:
        with st.expander("Context"):
            st.write(st.session_state.context)

        show_branches(
            st.session_state.branches,
            st.session_state.branch_scores,
            st.session_state.branch_confidences,
            st.session_state.recommended_branch
        )

        breakdown()
        tree_view()
        decision()


if __name__ == "__main__":
    main()
