"""
Tree-of-Thought reasoning engine with real LLM-based reasoning (Cohere).
"""

import networkx as nx
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import cohere


class ThoughtNode:
    """Represents a node in the Tree-of-Thought."""

    def __init__(self, id: str, text: str, parent_id: Optional[str] = None,
                 depth: int = 0, branch_type: str = "reasoning"):
        self.id = id
        self.text = text
        self.parent_id = parent_id
        self.depth = depth
        self.branch_type = branch_type
        self.children = []
        self.score = None
        self.created_at = datetime.now().isoformat()


class ToTEngine:
    """Tree-of-Thought reasoning engine with LLM."""

    def __init__(self):
        self.tree = nx.DiGraph()
        self.nodes = {}
        self.root_id = None

        # 🔑 ADD YOUR API KEY HERE
        self.co = cohere.Client("API_KEY")

    # =========================
    # TREE CREATION
    # =========================
    def create_tree(self, question: str, context: str) -> str:
        self.tree = nx.DiGraph()
        self.nodes = {}

        root_text = f"Question: {question}\n\nContext: {context[:500]}..."
        self.root_id = self.add_node(root_text, branch_type="root")

        return self.root_id

    def add_node(self, text: str, parent_id: Optional[str] = None,
                 branch_type: str = "reasoning") -> str:

        node_id = str(uuid.uuid4())[:8]

        depth = self.nodes[parent_id].depth + 1 if parent_id else 0

        node = ThoughtNode(node_id, text, parent_id, depth, branch_type)

        # Simple heuristic score (for visualization)
        node.score = min(len(text) / 500, 1.0)

        self.nodes[node_id] = node
        self.tree.add_node(node_id, **node.__dict__)

        if parent_id:
            self.tree.add_edge(parent_id, node_id)
            self.nodes[parent_id].children.append(node_id)

        return node_id

    # =========================
    # BRANCH GENERATION
    # =========================
    def generate_yes_no_branches(self, question: str, context: str) -> Dict[str, str]:

        yes_reasoning = self._generate_reasoning_path(
            question, context, "YES (supporting decision)"
        )
        no_reasoning = self._generate_reasoning_path(
            question, context, "NO (opposing decision)"
        )

        yes_id = self.add_node(yes_reasoning, self.root_id, branch_type="yes")
        no_id = self.add_node(no_reasoning, self.root_id, branch_type="no")

        self._add_evidence_nodes(yes_id, context)
        self._add_evidence_nodes(no_id, context)

        return {'yes': yes_id, 'no': no_id}

    def generate_open_ended_branches(self, question: str, context: str,
                                    num_branches: int = 3) -> List[str]:

        branch_ids = []

        for i in range(num_branches):
            perspective = self._get_perspective(i)

            reasoning = self._generate_reasoning_path(
                question, context, perspective
            )

            branch_id = self.add_node(reasoning, self.root_id, branch_type="reasoning")
            branch_ids.append(branch_id)

            self._add_evidence_nodes(branch_id, context)

        return branch_ids

    # =========================
    # LLM REASONING
    # =========================
    def _generate_reasoning_path(self, question: str, context: str, stance: str = "") -> str:

        prompt = f"""
You are an AI reasoning system using Tree-of-Thought.

Question:
{question}

Context:
{context[:800]}

Perspective:
{stance if stance else "Neutral"}

Task:
- Break down reasoning step by step
- Refer to context evidence
- Identify supporting or conflicting points
- Provide logical explanation

Output format:

Reasoning Steps:
1.
2.
3.

Conclusion:
"""

        try:
            response = self.co.generate(
                model="command-r-plus",
                prompt=prompt,
                max_tokens=300,
                temperature=0.7
            )

            return response.generations[0].text.strip()

        except Exception as e:
            return f"[ERROR] LLM reasoning failed: {str(e)}"

    # =========================
    # EXPANSION (DEEPER THINKING)
    # =========================
    def expand_branch(self, parent_id: str, question: str, context: str) -> str:

        parent_node = self.nodes[parent_id]

        deeper_reasoning = self._generate_deeper_reasoning(
            parent_node.text, question, context
        )

        return self.add_node(deeper_reasoning, parent_id, branch_type="reasoning")

    def _generate_deeper_reasoning(self, current_reasoning: str,
                                  question: str, context: str) -> str:

        prompt = f"""
You are refining reasoning in a Tree-of-Thought system.

Question:
{question}

Context:
{context[:800]}

Previous reasoning:
{current_reasoning[:500]}

Task:
- Expand reasoning depth
- Add new insights
- Identify possible flaws
- Strengthen conclusion

Output:
"""

        try:
            response = self.co.generate(
                model="command-r-plus",
                prompt=prompt,
                max_tokens=250,
                temperature=0.7
            )

            return response.generations[0].text.strip()

        except Exception as e:
            return f"[ERROR] Expansion failed: {str(e)}"

    # =========================
    # EVIDENCE HANDLING
    # =========================
    def _add_evidence_nodes(self, parent_id: str, context: str):

        sentences = context.split('.')

        evidence_points = [
            s.strip() for s in sentences if len(s.strip()) > 50
        ][:3]

        for i, evidence in enumerate(evidence_points):
            evidence_text = f"Evidence {i+1}: {evidence}"
            self.add_node(evidence_text, parent_id, branch_type="evidence")

    # =========================
    # PERSPECTIVES
    # =========================
    def _get_perspective(self, index: int) -> str:
        perspectives = [
            "Conservative Approach",
            "Optimistic Outlook",
            "Risk-Aware Analysis",
            "Evidence-Focused",
            "Balanced Perspective"
        ]
        return perspectives[index % len(perspectives)]

    # =========================
    # TREE UTILITIES
    # =========================
    def get_branch_nodes(self, branch_root_id: str) -> List[str]:

        nodes = [branch_root_id]

        def collect(node_id):
            for child in self.nodes[node_id].children:
                nodes.append(child)
                collect(child)

        collect(branch_root_id)
        return nodes

    def get_tree_structure(self) -> nx.DiGraph:
        return self.tree

    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        return self.nodes.get(node_id)
