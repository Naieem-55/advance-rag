"""
Cross-Encoder Reranker for HippoRAG
Provides precise reranking of retrieved passages
"""

import numpy as np
from typing import List, Tuple, Optional
import os

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Cross-encoder based reranker for improving retrieval precision."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", use_gpu: bool = False):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
            use_gpu: Whether to use GPU for inference
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder

            device = "cuda" if self.use_gpu else "cpu"
            self.model = CrossEncoder(self.model_name, device=device)
            logger.info(f"Loaded cross-encoder: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}")
            self.model = None

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None = all)

        Returns:
            Tuple of (reranked_indices, reranked_scores)
        """
        if self.model is None or len(documents) == 0:
            # Return original order if model not available
            return list(range(len(documents))), [1.0] * len(documents)

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get scores from cross-encoder
        scores = self.model.predict(pairs)

        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]

        if top_k:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]

        return sorted_indices.tolist(), sorted_scores.tolist()


class CohereReranker:
    """Cohere API-based reranker for high-quality reranking."""

    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-v3.5"):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            model: Cohere rerank model name
        """
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            self._init_client()
        else:
            logger.warning("COHERE_API_KEY not set. Cohere reranker disabled.")

    def _init_client(self):
        """Initialize Cohere client."""
        try:
            import cohere
            self.client = cohere.Client(self.api_key)
            logger.info(f"Initialized Cohere reranker with model: {self.model}")
        except ImportError:
            logger.warning("cohere not installed. Run: pip install cohere")
            self.client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Cohere client: {e}")
            self.client = None

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Rerank documents using Cohere API.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            Tuple of (reranked_indices, reranked_scores)
        """
        if self.client is None or len(documents) == 0:
            return list(range(len(documents))), [1.0] * len(documents)

        try:
            response = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_n=top_k or len(documents)
            )

            indices = [r.index for r in response.results]
            scores = [r.relevance_score for r in response.results]

            return indices, scores

        except Exception as e:
            logger.error(f"Cohere rerank error: {e}")
            return list(range(len(documents))), [1.0] * len(documents)


class GeminiReranker:
    """Gemini-based reranker using LiteLLM."""

    def __init__(self, model_name: str = "gemini/gemini-2.5-flash"):
        """
        Initialize Gemini reranker.

        Args:
            model_name: Gemini model name for reranking
        """
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Rerank documents using Gemini for relevance scoring.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            Tuple of (reranked_indices, reranked_scores)
        """
        if len(documents) == 0:
            return [], []

        try:
            import litellm

            # Create prompt for relevance scoring
            prompt = f"""Rate the relevance of each document to the query on a scale of 0-10.
Query: {query}

Documents:
"""
            for i, doc in enumerate(documents[:20]):  # Limit to 20 docs
                doc_preview = doc[:300] + "..." if len(doc) > 300 else doc
                prompt += f"\n[{i}] {doc_preview}\n"

            prompt += """
Return ONLY a JSON array of scores in order, like: [8, 5, 9, 3, ...]
No explanation, just the array."""

            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            # Parse response
            import json
            import re
            content = response.choices[0].message.content
            # Extract JSON array from response
            match = re.search(r'\[[\d,\s\.]+\]', content)
            if match:
                scores = json.loads(match.group())
                # Normalize to 0-1
                scores = [s / 10.0 for s in scores]

                # Pad with zeros if needed
                while len(scores) < len(documents):
                    scores.append(0.0)

                scores = np.array(scores[:len(documents)])
                sorted_indices = np.argsort(scores)[::-1]
                sorted_scores = scores[sorted_indices]

                if top_k:
                    sorted_indices = sorted_indices[:top_k]
                    sorted_scores = sorted_scores[:top_k]

                return sorted_indices.tolist(), sorted_scores.tolist()

        except Exception as e:
            logger.error(f"Gemini rerank error: {e}")

        return list(range(len(documents))), [1.0] * len(documents)


def get_reranker(reranker_type: str = "gemini", **kwargs):
    """
    Factory function to get appropriate reranker.

    Args:
        reranker_type: One of "cross-encoder", "cohere", "gemini"
        **kwargs: Additional arguments for the reranker

    Returns:
        Reranker instance
    """
    if reranker_type == "cross-encoder":
        return CrossEncoderReranker(**kwargs)
    elif reranker_type == "cohere":
        return CohereReranker(**kwargs)
    elif reranker_type == "gemini":
        return GeminiReranker(**kwargs)
    else:
        logger.warning(f"Unknown reranker type: {reranker_type}. Using Gemini.")
        return GeminiReranker(**kwargs)
