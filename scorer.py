"""
Scoring system for reasoning branches.
Computes multiple metrics and combines them into overall branch scores.
"""

from typing import Dict, List, Any
from utils import calculate_overlap

class BranchScorer:
    """Scorer for evaluating reasoning branches."""
    
    def __init__(self):
        """Initialize the branch scorer."""
        self.weights = {
            'relevance': 0.35,
            'depth': 0.20,
            'logic': 0.25,
            'evidence': 0.20
        }
        
    def score_branch(self, branch_nodes: List[Any], context: str, 
                     question: str) -> Dict[str, Any]:
        """
        Score a reasoning branch using multiple metrics.
        
        Args:
            branch_nodes: List of nodes in the branch
            context: Retrieved context
            question: Original question
            
        Returns:
            Dictionary containing scores and breakdown
        """
        # Calculate individual scores
        relevance_score = self._calculate_relevance(branch_nodes, context)
        depth_score = self._calculate_depth(branch_nodes)
        logic_score = self._calculate_logic(branch_nodes, question)
        evidence_score = self._calculate_evidence(branch_nodes, context)
        
        # Calculate weighted total
        total_score = (
            relevance_score * self.weights['relevance'] +
            depth_score * self.weights['depth'] +
            logic_score * self.weights['logic'] +
            evidence_score * self.weights['evidence']
        )
        
        # Prepare breakdown
        breakdown = {
            'relevance': {
                'score': relevance_score,
                'weight': self.weights['relevance'],
                'contribution': relevance_score * self.weights['relevance'],
                'explanation': 'How well the reasoning addresses the query'
            },
            'depth': {
                'score': depth_score,
                'weight': self.weights['depth'],
                'contribution': depth_score * self.weights['depth'],
                'explanation': 'Depth and thoroughness of reasoning steps'
            },
            'logic': {
                'score': logic_score,
                'weight': self.weights['logic'],
                'contribution': logic_score * self.weights['logic'],
                'explanation': 'Logical consistency and coherence'
            },
            'evidence': {
                'score': evidence_score,
                'weight': self.weights['evidence'],
                'contribution': evidence_score * self.weights['evidence'],
                'explanation': 'Use of and grounding in retrieved evidence'
            }
        }
        
        return {
            'total_score': total_score,
            'breakdown': breakdown
        }
    
    def _calculate_relevance(self, branch_nodes: List[Any], 
                            context: str) -> float:
        """
        Calculate relevance score based on context overlap.
        
        Args:
            branch_nodes: List of branch nodes
            context: Retrieved context
            
        Returns:
            Relevance score (0-1)
        """
        if not branch_nodes or not context:
            return 0.0
        
        # Combine all node texts
        branch_text = " ".join([node.text for node in branch_nodes])
        
        # Calculate overlap with context
        overlap = calculate_overlap(branch_text, context)
        
        # Also check if key terms from context appear in branch
        context_words = set(context.lower().split())
        branch_words = set(branch_text.lower().split())
        
        if context_words:
            term_coverage = len(branch_words.intersection(context_words)) / len(context_words)
        else:
            term_coverage = 0.0
        
        # Combine metrics
        relevance = (overlap * 0.6 + term_coverage * 0.4)
        
        return min(1.0, relevance)
    
    def _calculate_depth(self, branch_nodes: List[Any]) -> float:
        """
        Calculate depth score based on reasoning steps.
        
        Args:
            branch_nodes: List of branch nodes
            
        Returns:
            Depth score (0-1)
        """
        if not branch_nodes:
            return 0.0
        
        # Optimal depth is 3-5 levels
        depth = len(branch_nodes)
        
        if depth <= 1:
            return 0.2
        elif depth == 2:
            return 0.5
        elif 3 <= depth <= 5:
            return 1.0
        else:
            return 0.8  # Too deep, might be overcomplicating
    
    def _calculate_logic(self, branch_nodes: List[Any], 
                        question: str) -> float:
        """
        Calculate logical strength score.
        
        Args:
            branch_nodes: List of branch nodes
            question: Original question
            
        Returns:
            Logic score (0-1)
        """
        if len(branch_nodes) < 2:
            return 0.3
        
        score = 0.5  # Base score
        
        # Check for logical flow words
        logical_indicators = ['therefore', 'because', 'thus', 'hence', 
                             'consequently', 'implies', 'leads to']
        
        combined_text = " ".join([node.text.lower() for node in branch_nodes])
        
        # Count logical connectors
        connector_count = sum(1 for indicator in logical_indicators 
                            if indicator in combined_text)
        
        score += min(0.3, connector_count * 0.1)
        
        # Check if conclusion addresses the question
        if any(word in combined_text for word in question.lower().split()[:3]):
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_evidence(self, branch_nodes: List[Any], 
                           context: str) -> float:
        """
        Calculate evidence coverage score.
        
        Args:
            branch_nodes: List of branch nodes
            context: Retrieved context
            
        Returns:
            Evidence score (0-1)
        """
        if not context:
            return 0.0
        
        # Look for evidence nodes
        evidence_nodes = [node for node in branch_nodes 
                         if node.branch_type == 'evidence']
        
        if not evidence_nodes:
            return 0.3
        
        # Check how much of the context is cited
        context_sentences = context.split('.')
        cited_count = 0
        
        combined_text = " ".join([node.text for node in evidence_nodes])
        
        for sentence in context_sentences:
            if len(sentence.strip()) > 20:  # Only substantial sentences
                # Check if key words from sentence appear in evidence
                sentence_words = set(sentence.lower().split())
                cited_words = set(combined_text.lower().split())
                
                if len(sentence_words.intersection(cited_words)) > len(sentence_words) * 0.3:
                    cited_count += 1
        
        if context_sentences:
            coverage = cited_count / len(context_sentences)
        else:
            coverage = 0.0
        
        return min(1.0, coverage * 2)  # Multiply by 2 to reward good coverage