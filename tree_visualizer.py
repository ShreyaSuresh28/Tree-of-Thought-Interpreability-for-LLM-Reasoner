"""
Tree visualization module for interactive display of reasoning trees.
"""

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, Any, List
import numpy as np

class TreeVisualizer:
    """Interactive tree visualization for reasoning branches."""
    
    def __init__(self):
        """Initialize the tree visualizer."""
        pass
    
    def create_tree_figure(self, tree: nx.DiGraph, node_scores: Dict[str, float],
                          selected_branch: str = None) -> go.Figure:
        """
        Create a Plotly figure for tree visualization.
        
        Args:
            tree: NetworkX directed graph
            node_scores: Dictionary mapping node IDs to scores
            selected_branch: ID of selected branch node
            
        Returns:
            Plotly figure object
        """
        if len(tree.nodes) == 0:
            return go.Figure()
        
        # Calculate layout
        pos = nx.spring_layout(tree, k=2, iterations=50)
        
        # Extract node positions and colors
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_ids = []
        
        for node in tree.nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_ids.append(node)
            
            # Get node data
            node_data = tree.nodes[node]
            
            # Determine color based on score and selection
            if selected_branch and node.startswith(selected_branch):
                node_colors.append('#4CAF50')  # Green for selected branch
            elif node in node_scores:
                score = node_scores[node]
                # Color gradient from red (low) to green (high)
                color = self._get_score_color(score)
                node_colors.append(color)
            else:
                node_colors.append('#9E9E9E')  # Gray for unscores nodes
            
            # Create hover text
            hover_text = f"Node: {node}<br>"
            hover_text += f"Text: {node_data.get('text', '')[:100]}..."
            if node in node_scores:
                hover_text += f"<br>Score: {node_scores[node]:.2f}"
            node_text.append(hover_text)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        
        for edge in tree.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='none'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=30,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            text=node_ids,
            textposition="top center",
            hoverinfo='text',
            hovertext=node_text,
            name='Nodes'
        ))
        
        # Update layout
        fig.update_layout(
            title="Tree-of-Thought Reasoning Structure",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600
        )
        
        return fig
    
    def _get_score_color(self, score: float) -> str:
        """
        Get color based on score.
        
        Args:
            score: Score value (0-1)
            
        Returns:
            Hex color code
        """
        if score >= 0.8:
            return '#4CAF50'  # Green
        elif score >= 0.6:
            return '#8BC34A'  # Light green
        elif score >= 0.4:
            return '#FFC107'  # Yellow
        elif score >= 0.2:
            return '#FF9800'  # Orange
        else:
            return '#F44336'  # Red
    
    def render_tree_sidebar(self, tree: nx.DiGraph, node_scores: Dict[str, float],
                           branches: List[str], selected_branch: int) -> None:
        """
        Render tree information in sidebar.
        
        Args:
            tree: NetworkX directed graph
            node_scores: Node scores dictionary
            branches: List of branch root IDs
            selected_branch: Index of selected branch
        """
        st.sidebar.subheader("🌳 Tree Structure")
        
        st.sidebar.write(f"**Total Nodes:** {len(tree.nodes)}")
        st.sidebar.write(f"**Total Branches:** {len(branches)}")
        
        if branches:
            st.sidebar.write("**Branch Selection:**")
            branch_names = [f"Branch {i+1}" for i in range(len(branches))]
            selected_name = st.sidebar.radio(
                "Choose branch to view:",
                branch_names,
                index=selected_branch if 0 <= selected_branch < len(branches) else 0
            )
            
            branch_idx = branch_names.index(selected_name)
            
            # Show branch statistics
            branch_root = branches[branch_idx]
            if branch_root in node_scores:
                st.sidebar.metric("Branch Score", f"{node_scores[branch_root]:.2f}")
    
    def create_hierarchy_view(self, tree: nx.DiGraph, root_id: str) -> str:
        """
        Create a text-based hierarchy view of a branch.
        
        Args:
            tree: NetworkX directed graph
            root_id: Root node ID
            
        Returns:
            Formatted hierarchy string
        """
        if root_id not in tree:
            return "Branch not found"
        
        def build_hierarchy(node_id, depth=0):
            indent = "  " * depth
            node_data = tree.nodes[node_id]
            text = node_data.get('text', '')[:100]
            result = f"{indent}├─ [{node_id}] {text}...\n"
            
            # Find children
            children = [n for n in tree.neighbors(node_id) if tree.has_edge(node_id, n)]
            for child in children:
                result += build_hierarchy(child, depth + 1)
            
            return result
        
        return build_hierarchy(root_id)
    
    def create_simple_tree_display(self, tree: nx.DiGraph, 
                                  node_scores: Dict[str, float]) -> None:
        """
        Create a simple tree display using Streamlit columns.
        
        Args:
            tree: NetworkX directed graph
            node_scores: Node scores dictionary
        """
        if len(tree.nodes) == 0:
            st.info("No tree structure available yet. Start by asking a question.")
            return
        
        # Get root node
        roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
        if not roots:
            return
        
        root = roots[0]
        
        # Display root
        with st.container():
            st.markdown(f"**Root:** {tree.nodes[root].get('text', '')[:200]}...")
            
            # Display children (branches)
            children = [n for n in tree.neighbors(root)]
            
            if children:
                cols = st.columns(len(children))
                for idx, child in enumerate(children):
                    with cols[idx]:
                        score = node_scores.get(child, 0.0)
                        color = "green" if score >= 0.7 else "orange" if score >= 0.4 else "red"
                        st.markdown(f"""
                        <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>
                            <b>Branch {idx+1}</b><br>
                            Score: <span style='color:{color}'>{score:.2f}</span><br>
                            <small>{tree.nodes[child].get('text', '')[:100]}...</small>
                        </div>
                        """, unsafe_allow_html=True)