"""
Tree visualization with click support for explainability.
"""

import networkx as nx
import plotly.graph_objects as go
from typing import Dict


class TreeVisualizer:

    def __init__(self):
        pass

    # -----------------------------
    # HIERARCHICAL LAYOUT
    # -----------------------------
    def _hierarchical_layout(self, tree, root):
        pos = {}

        def dfs(node, depth=0, x=0):
            children = list(tree.successors(node))

            if not children:
                pos[node] = (x, -depth)
                return x + 1

            child_x = x
            child_positions = []

            for child in children:
                child_x = dfs(child, depth + 1, child_x)
                child_positions.append(pos[child][0])

            avg_x = sum(child_positions) / len(child_positions)
            pos[node] = (avg_x, -depth)

            return child_x

        dfs(root)
        return pos

    # -----------------------------
    # MAIN FIGURE
    # -----------------------------
    def create_tree_figure(self, tree, node_scores, selected_branch=None):

        if len(tree.nodes) == 0:
            return go.Figure()

        roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
        root = roots[0] if roots else list(tree.nodes)[0]

        pos = self._hierarchical_layout(tree, root)

        node_x, node_y, node_colors, node_text, node_ids = [], [], [], [], []

        for node in tree.nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_ids.append(node)

            node_data = tree.nodes[node]
            node_type = node_data.get("branch_type", "unknown")

            # COLOR LOGIC
            if node == root:
                color = "#2196F3"
            elif node_type == "evidence":
                color = "#9C27B0"
            elif selected_branch and node in nx.descendants(tree, selected_branch):
                color = "#4CAF50"
            elif node in node_scores:
                color = self._get_score_color(node_scores[node])
            else:
                color = "#BDBDBD"

            node_colors.append(color)

            # HOVER TEXT
            hover = f"""
            Node: {node}
            Type: {node_type}
            Depth: {node_data.get('depth', 0)}
            Text: {node_data.get('text', '')[:120]}...
            """

            if node in node_scores:
                hover += f"\nScore: {node_scores[node]:.3f}"

            node_text.append(hover)

        # EDGES
        edge_x, edge_y = [], []

        for edge in tree.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig = go.Figure()

        # Edges
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=1.5, color='#888'),
            hoverinfo='none'
        ))

        # Nodes (🔥 CLICK ENABLED)
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=32,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            text=node_ids,
            textposition="top center",
            customdata=node_ids,  # 🔥 IMPORTANT
            hoverinfo='text',
            hovertext=node_text
        ))

        fig.update_layout(
            title="🌳 Explainable Tree-of-Thought",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=650
        )

        return fig

    def _get_score_color(self, score):
        if score >= 0.8:
            return '#4CAF50'
        elif score >= 0.6:
            return '#8BC34A'
        elif score >= 0.4:
            return '#FFC107'
        elif score >= 0.2:
            return '#FF9800'
        else:
            return '#F44336'
