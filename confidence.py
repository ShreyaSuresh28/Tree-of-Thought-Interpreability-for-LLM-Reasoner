"""
Confidence calculation module.
Computes confidence percentages based on branch scores.
"""

from typing import List, Dict, Any

class ConfidenceCalculator:
    """Calculate confidence scores for reasoning branches."""
    
    def __init__(self):
        """Initialize the confidence calculator."""
        pass
    
    def calculate_confidence(self, branch_scores: List[float]) -> List[float]:
        """
        Calculate confidence percentages for all branches.
        
        Args:
            branch_scores: List of branch scores (0-1)
            
        Returns:
            List of confidence percentages (0-100)
        """
        if not branch_scores:
            return []
        
        max_score = max(branch_scores)
        
        if max_score == 0:
            return [0.0] * len(branch_scores)
        
        confidences = [(score / max_score) * 100 for score in branch_scores]
        
        return confidences
    
    def get_recommended_branch(self, branch_scores: List[float]) -> int:
        """
        Get index of recommended branch.
        
        Args:
            branch_scores: List of branch scores
            
        Returns:
            Index of branch with highest score
        """
        if not branch_scores:
            return -1
        
        return branch_scores.index(max(branch_scores))
    
    def calculate_confidence_interval(self, scores: List[float]) -> Dict[str, float]:
        """
        Calculate confidence interval for branch scores.
        
        Args:
            scores: List of branch scores
            
        Returns:
            Dictionary with mean, std_dev, and confidence interval
        """
        import numpy as np
        
        if not scores:
            return {'mean': 0, 'std_dev': 0, 'ci_lower': 0, 'ci_upper': 0}
        
        mean = np.mean(scores)
        std_dev = np.std(scores)
        
        # 95% confidence interval
        ci_lower = max(0, mean - 1.96 * std_dev)
        ci_upper = min(1, mean + 1.96 * std_dev)
        
        return {
            'mean': mean,
            'std_dev': std_dev,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level description.
        
        Args:
            confidence: Confidence percentage
            
        Returns:
            Confidence level string
        """
        if confidence >= 80:
            return "Very High"
        elif confidence >= 60:
            return "High"
        elif confidence >= 40:
            return "Medium"
        elif confidence >= 20:
            return "Low"
        else:
            return "Very Low"
    
    def generate_confidence_explanation(self, branch_score: float, 
                                       max_score: float) -> str:
        """
        Generate explanation for confidence score.
        
        Args:
            branch_score: Score of current branch
            max_score: Highest score among all branches
            
        Returns:
            Explanation string
        """
        ratio = branch_score / max_score if max_score > 0 else 0
        
        if ratio >= 0.9:
            return "This branch has nearly the maximum possible score, indicating high confidence in its reasoning."
        elif ratio >= 0.7:
            return "This branch scores well compared to alternatives, showing good confidence."
        elif ratio >= 0.5:
            return "This branch shows moderate confidence; consider reviewing alternatives."
        elif ratio >= 0.3:
            return "Low confidence; the reasoning may need additional evidence or restructuring."
        else:
            return "Very low confidence; this branch may be insufficient for decision-making."