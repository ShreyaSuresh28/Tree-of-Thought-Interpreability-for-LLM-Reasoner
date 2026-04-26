"""
Advanced Scoring System for Tree-of-Thought reasoning.
Hybrid approach: Rule-based + Semantic similarity
"""

from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer, util
import numpy as np


class BranchScorer:
    """Advanced scorer for evaluating reasoning branches."""

    def __init__(self):
        self.weights = {
            'relevance': 0.35,
            'depth': 0.20,
            'logic': 0.25,
            'evidence': 0.20
        }

        # 🔥 Load embedding model (VERY IMPORTANT)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    # =========================
    # MAIN SCORING FUNCTION
    # =========================
    def score_branch(self, branch_nodes: List[Any], context: str,
                     question: str) -> Dict[str, Any]:

        relevance_score = self._calculate_relevance(branch_nodes, context)
        depth_score = self._calculate_depth(branch_nodes)
        logic_score = self._calculate_logic(branch_nodes, question)
        evidence_score = self._calculate_evidence(branch_nodes, context)

        total_score = (
            relevance_score * self.weights['relevance'] +
            depth_score * self.weights['depth'] +
            logic_score * self.weights['logic'] +
            evidence_score * self.weights['evidence']
        )

        breakdown = {
            'relevance': self._format_metric(relevance_score, 'relevance'),
            'depth': self._format_metric(depth_score, 'depth'),
            'logic': self._format_metric(logic_score, 'logic'),
            'evidence': self._format_metric(evidence_score, 'evidence')
        }

        return {
            'total_score': float(total_score),
            'breakdown': breakdown
        }

    # =========================
    # RELEVANCE (SEMANTIC 🔥)
    # =========================
    def _calculate_relevance(self, branch_nodes: List[Any], context: str) -> float:

        if not branch_nodes or not context:
            return 0.0

        branch_text = " ".join([node.text for node in branch_nodes])

        # 🔥 Semantic similarity using embeddings
        emb1 = self.model.encode(branch_text, convert_to_tensor=True)
        emb2 = self.model.encode(context, convert_to_tensor=True)

        similarity = util.cos_sim(emb1, emb2).item()

        return float(max(0.0, min(1.0, similarity)))

    # =========================
    # DEPTH
    # =========================
    def _calculate_depth(self, branch_nodes: List[Any]) -> float:

        depth = len(branch_nodes)

        if depth <= 1:
            return 0.2
        elif depth == 2:
            return 0.5
        elif 3 <= depth <= 6:
            return 1.0
        else:
            return 0.8

    # =========================
    # LOGIC (IMPROVED)
    # =========================
    def _calculate_logic(self, branch_nodes: List[Any], question: str) -> float:

        if len(branch_nodes) < 2:
            return 0.3

        combined_text = " ".join([node.text.lower() for node in branch_nodes])

        logical_indicators = [
            'therefore', 'because', 'thus', 'hence',
            'consequently', 'implies', 'leads to', 'as a result'
        ]

        connector_count = sum(
            1 for indicator in logical_indicators if indicator in combined_text
        )

        score = 0.5 + min(0.3, connector_count * 0.1)

        # 🔥 Check semantic alignment with question
        q_words = set(question.lower().split())
        b_words = set(combined_text.split())

        overlap = len(q_words.intersection(b_words)) / max(len(q_words), 1)

        score += overlap * 0.2

        return float(min(1.0, score))

    # =========================
    # EVIDENCE (IMPROVED)
    # =========================
    def _calculate_evidence(self, branch_nodes: List[Any], context: str) -> float:

        evidence_nodes = [
            node for node in branch_nodes if node.branch_type == 'evidence'
        ]

        if not evidence_nodes:
            return 0.3

        evidence_text = " ".join([node.text for node in evidence_nodes])

        # 🔥 Semantic similarity with context
        emb1 = self.model.encode(evidence_text, convert_to_tensor=True)
        emb2 = self.model.encode(context, convert_to_tensor=True)

        similarity = util.cos_sim(emb1, emb2).item()

        return float(min(1.0, similarity))

    # =========================
    # HELPER
    # =========================
    def _format_metric(self, score: float, name: str) -> Dict[str, Any]:

        explanations = {
            'relevance': 'Semantic similarity between reasoning and context',
            'depth': 'Number of reasoning steps',
            'logic': 'Logical coherence and flow',
            'evidence': 'Grounding in retrieved evidence'
        }

        return {
            'score': float(score),
            'weight': self.weights[name],
            'contribution': float(score * self.weights[name]),
            'explanation': explanations[name]
        }
