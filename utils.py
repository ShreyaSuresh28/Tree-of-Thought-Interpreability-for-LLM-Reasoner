"""
Advanced utility functions for Tree-of-Thought framework.
Includes tokenization, chunking, text cleaning, and semantic helpers.
"""

import tiktoken
from typing import List
import re
import numpy as np


# =========================
# TOKEN COUNTING
# =========================
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


# =========================
# SMART CHUNKING 🔥
# =========================
def chunk_text_by_tokens(text: str, max_tokens: int = 500, overlap: int = 100) -> List[str]:
    """
    Chunk text with sentence awareness (better than raw token split).
    """

    if not text:
        return []

    # Split into sentences first (IMPORTANT)
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(current_chunk.strip())

            # overlap handling
            overlap_text = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_text + " " + sentence
            current_tokens = count_tokens(current_chunk)
        else:
            current_chunk += " " + sentence
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# =========================
# TEXT CLEANING 🔥
# =========================
def clean_text(text: str) -> str:
    """
    Clean text but preserve structure for LLM.
    """

    if not text:
        return ""

    # Remove weird characters but keep structure
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\n]', '', text)

    # Normalize spaces but keep paragraphs
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


# =========================
# IMPROVED OVERLAP 🔥
# =========================
def calculate_overlap(text1: str, text2: str) -> float:
    """
    Improved semantic-like overlap using weighted scoring.
    """

    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)

    precision = len(intersection) / len(words1)
    recall = len(intersection) / len(words2)

    if precision + recall == 0:
        return 0.0

    # F1-style score (better than Jaccard)
    return 2 * (precision * recall) / (precision + recall)


# =========================
# KEYWORD EXTRACTION 🔥
# =========================
def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    Extract important words (simple but effective).
    """

    words = re.findall(r'\w+', text.lower())

    stopwords = {
        'the', 'is', 'in', 'and', 'to', 'of', 'a', 'for',
        'on', 'with', 'as', 'by', 'an', 'be', 'this', 'that'
    }

    words = [w for w in words if w not in stopwords and len(w) > 3]

    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    return [w[0] for w in sorted_words[:top_k]]


# =========================
# SCORE FORMAT
# =========================
def format_score(score: float) -> str:
    return f"{score * 100:.2f}%"


# =========================
# QUERY VALIDATION 🔥
# =========================
def validate_query(query: str) -> bool:
    """
    Stronger validation for user queries.
    """

    if not query:
        return False

    query = query.strip()

    if len(query) < 5:
        return False

    # Must contain at least one meaningful word
    if not re.search(r'\w+', query):
        return False

    return True


# =========================
# NORMALIZE SCORES 🔥
# =========================
def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores safely.
    """

    if not scores:
        return []

    arr = np.array(scores)

    if arr.max() == arr.min():
        return [0.5] * len(scores)

    normalized = (arr - arr.min()) / (arr.max() - arr.min())

    return normalized.tolist()
