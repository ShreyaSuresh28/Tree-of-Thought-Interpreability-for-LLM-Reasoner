"""
Utility functions for the Tree-of-Thought framework.
Provides helper functions for text processing, tokenization, and common operations.
"""

import tiktoken
from typing import List, Dict, Any
import re

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: Input text to count tokens for
        model: Model name for tokenization (default: gpt-3.5-turbo)
    
    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

def chunk_text_by_tokens(text: str, max_tokens: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into chunks based on token count with overlap.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move start position for overlap
        start += max_tokens - overlap
    
    return chunks

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text to clean
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
    return text.strip()

def calculate_overlap(text1: str, text2: str) -> float:
    """
    Calculate word overlap between two texts.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Overlap score (0-1)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def format_score(score: float) -> str:
    """
    Format score for display.
    
    Args:
        score: Score to format (0-1)
    
    Returns:
        Formatted percentage string
    """
    return f"{score * 100:.1f}%"

def validate_query(query: str) -> bool:
    """
    Validate user query.
    
    Args:
        query: User query to validate
    
    Returns:
        True if query is valid, False otherwise
    """
    if not query or len(query.strip()) < 3:
        return False
    return True