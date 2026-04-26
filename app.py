"""
Main Streamlit application for Human-in-the-Loop Tree-of-Thought Framework.
Provides interactive UI for document-grounded decision support.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import plotly.graph_objects as go

# Import custom modules
from rag_engine import RAGEngine
from tot_engine import ToTEngine
from scorer import BranchScorer
from confidence import ConfidenceCalculator
from memory_manager import MemoryManager
from tree_visualizer import TreeVisualizer
from utils import validate_query, format_score

# Page configuration
st.set_page_config(
    page_title="Tree-of-Thought Decision Support",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .score-card {
        padding: 1rem;
        border-radius: 10px;
        background: #f0f2f6;
        margin: 0.5rem 0;
    }
    .branch-card {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ddd;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .branch-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .selected-branch {
        border-color: #4CAF50;
        background: #f0fff0;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FFC107;
        font-weight: bold;
    }
    .confidence-low {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables."""
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = RAGEngine()
    if 'tot_engine' not in st.session_state:
        st.session_state.tot_engine = ToTEngine()
    if 'scorer' not in st.session_state:
        st.session_state.scorer = BranchScorer()
    if 'confidence_calc' not in st.session_state:
        st.session_state.confidence_calc = ConfidenceCalculator()
    if 'memory_manager' not in st.session_state:
        st.session_state.memory_manager = MemoryManager()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = TreeVisualizer()
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'branches' not in st.session_state:
        st.session_state.branches = []
    if 'branch_scores' not in st.session_state:
        st.session_state.branch_scores = []
    if 'branch_confidences' not in st.session_state:
        st.session_state.branch_confidences = []
    if 'selected_branch' not in st.session_state:
        st.session_state.selected_branch = None
    if 'context' not in st.session_state:
        st.session_state.context = None
    if 'question' not in st.session_state:
        st.session_state.question = None
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False

# Sidebar for document upload and memory management
def render_sidebar():
    """Render sidebar with controls."""
    with st.sidebar:
        st.markdown("## 📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload PDF or TXT file",
            type=['pdf', 'txt'],
            help="Upload a document to ground the reasoning"
        )
        
        if uploaded_file and not st.session_state.document_processed:
            with st.spinner("Processing document..."):
                try:
                    chunks = st.session_state.rag_engine.process_document(uploaded_file)
                    st.success(f"✅ Processed {len(chunks)} chunks")
                    st.session_state.document_processed = True
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        st.markdown("## 💾 Memory Management")
        
        # Display sessions
        sessions = st.session_state.memory_manager.list_sessions()
        if sessions:
            st.write("**Previous Sessions:**")
            session_df = pd.DataFrame(sessions)
            st.dataframe(session_df[['session_id', 'query', 'final_decision']], 
                        use_container_width=True)
        
        # Clear memory button
        if st.button("🗑️ Clear All Memory", type="secondary"):
            st.session_state.memory_manager.clear_memory()
            st.success("Memory cleared!")
            st.rerun()
        
        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.info("""
        **Human-in-the-Loop Tree-of-Thought Framework**
        
        This system combines:
        - RAG for document grounding
        - Multi-branch reasoning
        - Automatic scoring
        - Confidence estimation
        - Human oversight
        """)

# Main query input area
def render_query_input():
    """Render the main query input area."""
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown("## 🌳 Tree-of-Thought Decision Support")
    st.markdown("Ask questions grounded in your documents with multi-branch reasoning")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_area(
            "💭 Ask your question:",
            placeholder="Example: Based on the document, should we proceed with this strategy?",
            height=100
        )
    with col2:
        st.write("")
        st.write("")
        question_type = st.radio(
            "Question Type:",
            ["Yes/No", "Open-ended"],
            horizontal=True
        )
    
    return question, question_type

# Generate reasoning branches
def generate_branches(question: str, context: str, question_type: str):
    """Generate reasoning branches based on question type."""
    tot_engine = st.session_state.tot_engine
    
    # Create tree root
    tot_engine.create_tree(question, context)
    
    # Generate branches
    if question_type == "Yes/No":
        branches = tot_engine.generate_yes_no_branches(question, context)
        return list(branches.values())
    else:
        branches = tot_engine.generate_open_ended_branches(question, context, num_branches=3)
        return branches

# Score branches
def score_branches(branches: List[str], context: str, question: str):
    """Score all branches."""
    tot_engine = st.session_state.tot_engine
    scorer = st.session_state.scorer
    
    branch_scores = []
    score_details = []
    
    for branch_root in branches:
        # Get all nodes in branch
        branch_nodes_ids = tot_engine.get_branch_nodes(branch_root)
        branch_nodes = [tot_engine.get_node(node_id) for node_id in branch_nodes_ids]
        
        # Score branch
        scoring_result = scorer.score_branch(branch_nodes, context, question)
        branch_scores.append(scoring_result['total_score'])
        score_details.append(scoring_result['breakdown'])
    
    return branch_scores, score_details

# Calculate confidences
def calculate_confidences(scores: List[float]):
    """Calculate confidence percentages."""
    confidence_calc = st.session_state.confidence_calc
    confidences = confidence_calc.calculate_confidence(scores)
    recommended = confidence_calc.get_recommended_branch(scores)
    return confidences, recommended

# Render branches for human selection
def render_branches(branches: List[str], scores: List[float], 
                   confidences: List[float], recommended: int,
                   score_details: List[Dict]):
    """Render branches for human selection."""
    st.markdown("## 🌿 Reasoning Branches")
    st.markdown("Select the branch you find most convincing:")
    
    cols = st.columns(len(branches))
    
    for idx, (branch_root, score, confidence) in enumerate(zip(branches, scores, confidences)):
        with cols[idx]:
            # Determine confidence class
            if confidence >= 60:
                confidence_class = "confidence-high"
            elif confidence >= 30:
                confidence_class = "confidence-medium"
            else:
                confidence_class = "confidence-low"
            
            # Branch card styling
            is_selected = (st.session_state.selected_branch == idx)
            card_class = "branch-card selected-branch" if is_selected else "branch-card"
            
            # Display branch
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            st.markdown(f"### Branch {idx + 1}")
            
            # Show branch content preview
            branch_node = st.session_state.tot_engine.get_node(branch_root)
            if branch_node:
                st.markdown(f"**Reasoning:** {branch_node.text[:200]}...")
            
            # Score and confidence
            st.markdown(f"**Score:** {score:.3f}")
            st.markdown(f'**Confidence:** <span class="{confidence_class}">{confidence:.1f}%</span>', 
                       unsafe_allow_html=True)
            
            # Recommendation badge
            if idx == recommended:
                st.markdown("🏆 **Recommended**")
            
            # Selection button
            if st.button(f"Select Branch {idx + 1}", key=f"select_{idx}"):
                st.session_state.selected_branch = idx
                st.rerun()
            
            # Expand branch button
            if st.button(f"🔍 Expand Branch {idx + 1}", key=f"expand_{idx}"):
                new_node = st.session_state.tot_engine.expand_branch(
                    branch_root, 
                    st.session_state.question,
                    st.session_state.context
                )
                # Re-score after expansion
                new_branch_nodes = st.session_state.tot_engine.get_branch_nodes(branch_root)
                new_branch_nodes_objs = [st.session_state.tot_engine.get_node(nid) 
                                        for nid in new_branch_nodes]
                new_score = st.session_state.scorer.score_branch(
                    new_branch_nodes_objs, 
                    st.session_state.context,
                    st.session_state.question
                )
                scores[idx] = new_score['total_score']
                score_details[idx] = new_score['breakdown']
                st.success("Branch expanded and rescored!")
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# Render score breakdown
def render_score_breakdown(score_details: List[Dict], selected_idx: int):
    """Render detailed score breakdown for selected branch."""
    if selected_idx is not None and score_details:
        st.markdown("## 📊 Score Breakdown")
        
        breakdown = score_details[selected_idx]
        
        # Create metrics
        cols = st.columns(4)
        metrics = ['relevance', 'depth', 'logic', 'evidence']
        
        for idx, metric in enumerate(metrics):
            with cols[idx]:
                score = breakdown[metric]['score']
                contribution = breakdown[metric]['contribution']
                st.metric(
                    label=metric.capitalize(),
                    value=f"{score:.2f}",
                    delta=f"contrib: {contribution:.2f}"
                )
        
        # Explanation
        with st.expander("View Detailed Explanation"):
            for metric in metrics:
                st.markdown(f"**{metric.capitalize()}**")
                st.markdown(f"- Score: {breakdown[metric]['score']:.3f}")
                st.markdown(f"- Weight: {breakdown[metric]['weight']}")
                st.markdown(f"- Contribution: {breakdown[metric]['contribution']:.3f}")
                st.markdown(f"- Explanation: {breakdown[metric]['explanation']}")
                st.markdown("---")

# Render tree visualization
def render_tree_visualization():
    """Render interactive tree visualization."""
    st.markdown("## 🌳 Tree Visualization")
    
    tot_engine = st.session_state.tot_engine
    tree = tot_engine.get_tree_structure()
    
    if len(tree.nodes) > 0:
        # Calculate node scores for visualization
        node_scores = {}
        for node_id in tree.nodes:
            node = tot_engine.get_node(node_id)
            if node and node.score:
                node_scores[node_id] = node.score
        
        # Create figure
        fig = st.session_state.visualizer.create_tree_figure(
            tree, 
            node_scores,
            st.session_state.branches[st.session_state.selected_branch] 
            if st.session_state.selected_branch is not None else None
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show hierarchy
        if st.session_state.selected_branch is not None:
            with st.expander("View Branch Hierarchy"):
                branch_root = st.session_state.branches[st.session_state.selected_branch]
                hierarchy = st.session_state.visualizer.create_hierarchy_view(tree, branch_root)
                st.text(hierarchy)
    else:
        st.info("No tree to display yet. Ask a question to generate reasoning branches.")

# Make final decision
def make_decision():
    """Make final decision based on selected branch."""
    if st.session_state.selected_branch is not None:
        st.markdown("## ✅ Final Decision")
        
        # Get selected branch
        selected_idx = st.session_state.selected_branch
        selected_branch = st.session_state.branches[selected_idx]
        confidence = st.session_state.branch_confidences[selected_idx]
        
        # Display decision interface
        st.markdown(f"""
        <div class="score-card">
            <h3>Selected Branch: {selected_idx + 1}</h3>
            <p><strong>Confidence:</strong> {confidence:.1f}%</p>
            <p><strong>Decision:</strong> Based on the reasoning and evidence, 
            this branch provides the most compelling argument.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Override option
        override = st.checkbox("Override recommended branch")
        if override:
            new_selection = st.selectbox(
                "Choose different branch:",
                options=range(len(st.session_state.branches)),
                format_func=lambda x: f"Branch {x + 1}"
            )
            if st.button("Confirm Override"):
                st.session_state.selected_branch = new_selection
                st.rerun()
        
        # Finalize button
        if st.button("🎯 Finalize Decision", type="primary"):
            # Save to memory
            session_id = st.session_state.memory_manager.save_reasoning_session(
                query=st.session_state.question,
                context=st.session_state.context,
                branches=[{
                    'id': branch,
                    'text': st.session_state.tot_engine.get_node(branch).text[:200]
                } for branch in st.session_state.branches],
                scores={f"Branch_{i}": score for i, score in enumerate(st.session_state.branch_scores)},
                confidences={f"Branch_{i}": conf for i, conf in enumerate(st.session_state.branch_confidences)},
                selected_branch=selected_idx,
                final_decision=f"Selected Branch {selected_idx + 1} with {confidence:.1f}% confidence"
            )
            st.success(f"Decision saved! Session ID: {session_id}")
            st.balloons()

# Main app
def main():
    """Main application entry point."""
    initialize_session_state()
    render_sidebar()
    
    # Query input
    question, question_type = render_query_input()
    
    # Process query
    if st.button("🧠 Generate Reasoning", type="primary", disabled=not st.session_state.document_processed):
        if not validate_query(question):
            st.error("Please enter a valid question (minimum 3 characters)")
            return
        
        with st.spinner("Generating reasoning branches..."):
            # Get context from RAG
            context = st.session_state.rag_engine.get_context_for_query(question, top_k=5)
            st.session_state.context = context
            st.session_state.question = question
            
            # Generate branches
            branches = generate_branches(question, context, question_type)
            st.session_state.branches = branches
            
            # Score branches
            scores, score_details = score_branches(branches, context, question)
            st.session_state.branch_scores = scores
            st.session_state.score_details = score_details
            
            # Calculate confidences
            confidences, recommended = calculate_confidences(scores)
            st.session_state.branch_confidences = confidences
            st.session_state.recommended_branch = recommended
            
            st.success(f"Generated {len(branches)} reasoning branches!")
    
    # Display results if branches exist
    if st.session_state.branches:
        # Show context
        with st.expander("📚 Retrieved Context"):
            st.write(st.session_state.context)
        
        # Render branches for selection
        render_branches(
            st.session_state.branches,
            st.session_state.branch_scores,
            st.session_state.branch_confidences,
            st.session_state.recommended_branch,
            st.session_state.score_details
        )
        
        # Show score breakdown for selected branch
        if st.session_state.selected_branch is not None:
            render_score_breakdown(st.session_state.score_details, st.session_state.selected_branch)
        
        # Show tree visualization
        render_tree_visualization()
        
        # Make final decision
        make_decision()

if __name__ == "__main__":
    main()