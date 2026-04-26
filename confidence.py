"""
Advanced Confidence calculation module.
Uses softmax normalization + uncertainty estimation.
"""

from typing import List, Dict
import numpy as np


class ConfidenceCalculator:
    """Advanced confidence scoring for reasoning branches."""

    def __init__(self):
        pass

    # =========================
    # MAIN CONFIDENCE FUNCTION
    # =========================
    def calculate_confidence(self, branch_scores: List[float]) -> List[float]:
        """
        Convert scores into probability-like confidence using softmax.

        Args:
            branch_scores: List of branch scores

        Returns:
            Confidence percentages (sum ≈ 100)
        """
        if not branch_scores:
            return []

        scores = np.array(branch_scores)

        # 🔥 Softmax normalization (VERY IMPORTANT)
        exp_scores = np.exp(scores - np.max(scores))  # stability trick
        probabilities = exp_scores / exp_scores.sum()

        return (probabilities * 100).tolist()

    # =========================
    # BEST BRANCH
    # =========================
    def get_recommended_branch(self, branch_scores: List[float]) -> int:
        if not branch_scores:
            return -1

        return int(np.argmax(branch_scores))

    # =========================
    # UNCERTAINTY ANALYSIS 🔥
    # =========================
    def calculate_uncertainty(self, branch_scores: List[float]) -> float:
        """
        Calculate uncertainty using entropy.

        High entropy → low confidence
        Low entropy → high confidence
        """
        if not branch_scores:
            return 1.0

        probs = np.array(self.calculate_confidence(branch_scores)) / 100

        entropy = -np.sum(probs * np.log(probs + 1e-9))

        return float(entropy)

    # =========================
    # CONFIDENCE LEVEL
    # =========================
    def get_confidence_level(self, confidence: float) -> str:

        if confidence >= 75:
            return "Very High"
        elif confidence >= 55:
            return "High"
        elif confidence >= 35:
            return "Medium"
        elif confidence >= 20:
            return "Low"
        else:
            return "Very Low"

    # =========================
    # CONFIDENCE EXPLANATION 🔥
    # =========================
    def generate_confidence_explanation(self, branch_scores: List[float],
                                        selected_idx: int) -> str:

        if not branch_scores:
            return "No confidence available."

        confidences = self.calculate_confidence(branch_scores)
        selected_conf = confidences[selected_idx]

        uncertainty = self.calculate_uncertainty(branch_scores)

        explanation = f"""
Selected Branch Confidence: {selected_conf:.2f}%

- This confidence is computed using softmax normalization over all branch scores.
- The system compares all reasoning paths and assigns probability-like confidence.
- Uncertainty (entropy): {uncertainty:.3f}

Interpretation:
"""

        if uncertainty < 0.5:
            explanation += "The system is highly certain about this decision."
        elif uncertainty < 1.0:
            explanation += "Moderate certainty — alternative branches exist."
        else:
            explanation += "High uncertainty — reasoning paths are equally competitive."

        return explanation.strip()

    # =========================
    # CONFIDENCE INTERVAL
    # =========================
    def calculate_confidence_interval(self, scores: List[float]) -> Dict[str, float]:

        if not scores:
            return {'mean': 0, 'std_dev': 0, 'ci_lower': 0, 'ci_upper': 0}

        scores = np.array(scores)

        mean = float(np.mean(scores))
        std_dev = float(np.std(scores))

        ci_lower = max(0, mean - 1.96 * std_dev)
        ci_upper = min(1, mean + 1.96 * std_dev)

        return {
            'mean': mean,
            'std_dev': std_dev,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
