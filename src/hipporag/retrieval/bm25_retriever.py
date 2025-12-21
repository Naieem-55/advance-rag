"""
BM25 Retriever for HippoRAG
Provides sparse retrieval to complement dense retrieval
"""

import numpy as np
from typing import List, Tuple, Optional
import pickle
import os
import re

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    """BM25 sparse retriever using rank_bm25 library."""

    def __init__(self, save_path: Optional[str] = None):
        self.save_path = save_path
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenizer that handles multiple languages including Bangla."""
        # Convert to lowercase for English, keep original for other scripts
        text = text.lower()
        # Split on whitespace and punctuation, keep words
        tokens = re.findall(r'\b\w+\b', text, re.UNICODE)
        return tokens

    def index(self, documents: List[str]) -> None:
        """Build BM25 index from documents."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed. Run: pip install rank-bm25")
            return

        self.documents = documents
        self.tokenized_docs = [self.tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        logger.info(f"BM25 index built with {len(documents)} documents")

        if self.save_path:
            self.save()

    def search(self, query: str, top_k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for relevant documents using BM25.

        Returns:
            Tuple of (sorted_doc_ids, sorted_scores)
        """
        if self.bm25 is None:
            logger.warning("BM25 index not initialized")
            return np.array([]), np.array([])

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores to 0-1 range
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.zeros_like(scores)

        sorted_ids = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_ids]

        if top_k:
            sorted_ids = sorted_ids[:top_k]
            sorted_scores = sorted_scores[:top_k]

        return sorted_ids, sorted_scores

    def save(self) -> None:
        """Save BM25 index to disk."""
        if self.save_path and self.bm25:
            data = {
                'documents': self.documents,
                'tokenized_docs': self.tokenized_docs,
            }
            with open(self.save_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"BM25 index saved to {self.save_path}")

    def load(self) -> bool:
        """Load BM25 index from disk."""
        if self.save_path and os.path.exists(self.save_path):
            try:
                from rank_bm25 import BM25Okapi

                with open(self.save_path, 'rb') as f:
                    data = pickle.load(f)

                self.documents = data['documents']
                self.tokenized_docs = data['tokenized_docs']
                self.bm25 = BM25Okapi(self.tokenized_docs)

                logger.info(f"BM25 index loaded from {self.save_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}")
                return False
        return False


def hybrid_score_fusion(
    dense_ids: np.ndarray,
    dense_scores: np.ndarray,
    bm25_ids: np.ndarray,
    bm25_scores: np.ndarray,
    alpha: float = 0.7,  # Weight for dense scores
    num_docs: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine dense and BM25 scores using weighted fusion.

    Args:
        dense_ids: Document IDs from dense retrieval (sorted by score)
        dense_scores: Scores from dense retrieval
        bm25_ids: Document IDs from BM25 retrieval (sorted by score)
        bm25_scores: Scores from BM25 retrieval
        alpha: Weight for dense scores (1-alpha for BM25)
        num_docs: Total number of documents

    Returns:
        Tuple of (sorted_doc_ids, fused_scores)
    """
    if num_docs is None:
        num_docs = max(len(dense_ids), len(bm25_ids))

    # Create score arrays for all documents
    dense_score_array = np.zeros(num_docs)
    bm25_score_array = np.zeros(num_docs)

    # Fill in scores
    for idx, score in zip(dense_ids, dense_scores):
        if idx < num_docs:
            dense_score_array[idx] = score

    for idx, score in zip(bm25_ids, bm25_scores):
        if idx < num_docs:
            bm25_score_array[idx] = score

    # Weighted combination
    fused_scores = alpha * dense_score_array + (1 - alpha) * bm25_score_array

    # Sort by fused score
    sorted_ids = np.argsort(fused_scores)[::-1]
    sorted_scores = fused_scores[sorted_ids]

    return sorted_ids, sorted_scores
