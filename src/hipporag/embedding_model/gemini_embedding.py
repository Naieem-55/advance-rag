"""
Gemini Embedding Model for HippoRAG using LiteLLM
"""

import os
import numpy as np
from typing import List, Optional
from tqdm import tqdm

import litellm

from .base import BaseEmbeddingModel, EmbeddingConfig
from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class GeminiEmbeddingModel(BaseEmbeddingModel):
    """Gemini Embedding Model using LiteLLM."""

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        # Override embedding model name if provided
        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        # Verify Gemini API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self._init_embedding_config()

        logger.info(f"Initialized Gemini Embedding: {self.embedding_model_name}")

    def _init_embedding_config(self) -> None:
        """Initialize embedding configuration."""
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
            },
            "embedding_dim": 3072,  # gemini-embedding-001 default dimension
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        self.embedding_dim = 3072  # gemini-embedding-001 default
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encode texts into embeddings using Gemini.

        Args:
            texts: List of texts to encode

        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Handle empty strings - Gemini API rejects empty text
        empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                empty_indices.append(i)
            else:
                non_empty_texts.append(text)

        # If all texts are empty, return zero vectors
        if not non_empty_texts:
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

        try:
            response = litellm.embedding(
                model=self.embedding_model_name,
                input=non_empty_texts,
            )

            # Extract embeddings from response
            non_empty_embeddings = [item['embedding'] for item in response.data]

            # Reconstruct full result with zero vectors for empty strings
            result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
            non_empty_idx = 0
            for i in range(len(texts)):
                if i not in empty_indices:
                    result[i] = non_empty_embeddings[non_empty_idx]
                    non_empty_idx += 1

            # Normalize if required
            if self.embedding_config.norm:
                norms = np.linalg.norm(result, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                result = result / norms

            return result

        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            raise

    def batch_encode(self, texts: List[str], batch_size: int = 100, **kwargs) -> np.ndarray:
        """
        Encode texts in batches.

        Args:
            texts: List of texts to encode
            batch_size: Number of texts per batch

        Returns:
            numpy array of embeddings
        """
        if len(texts) <= batch_size:
            return self.encode(texts)

        results = []
        pbar = tqdm(total=len(texts), desc="Batch Encoding (Gemini)")

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self.encode(batch)
                results.append(batch_embeddings)
            except Exception as e:
                logger.error(f"Error encoding batch: {e}")
                raise e
            pbar.update(len(batch))

        pbar.close()
        return np.concatenate(results, axis=0)
