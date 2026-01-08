"""
Text preprocessing utilities
"""
import re
from typing import List
from bs4 import BeautifulSoup


def clean_html(text: str) -> str:
    """
    Remove HTML tags from text
    
    Args:
        text: HTML text
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text(separator=' ', strip=True)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_keywords(text: str, max_words: int = 100) -> str:
    """
    Extract keywords from text
    
    Args:
        text: Input text
        max_words: Maximum number of words
    
    Returns:
        Extracted keywords
    """
    words = text.split()
    return ' '.join(words[:max_words])


def create_embedding_text(
    title: str,
    content: str,
    tags: List[str],
    difficulty: str = ""
) -> str:
    """
    Create embedding text from problem metadata
    
    Args:
        title: Problem title
        content: Problem description
        tags: Problem tags
        difficulty: Difficulty level
    
    Returns:
        Combined embedding text
    """
    # Clean and normalize
    title_clean = normalize_whitespace(title)
    content_clean = clean_html(content) if content else ""
    content_clean = normalize_whitespace(content_clean)
    
    # Extract first 200 words from content
    content_keywords = extract_keywords(content_clean, max_words=200)
    
    # Combine components
    parts = []
    
    if difficulty:
        parts.append(f"[{difficulty}]")
    
    parts.append(title_clean)
    
    if tags:
        parts.append(f"Topics: {', '.join(tags)}")
    
    if content_keywords:
        parts.append(content_keywords)
    
    return " | ".join(parts)
