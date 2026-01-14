"""
BGE-M3 embedding model for Polymath v3.

BGE-M3 produces 1024-dimensional embeddings with:
- Dense retrieval (semantic similarity)
- Sparse retrieval (lexical matching)
- Multi-vector retrieval (ColBERT-style)

We use dense embeddings for pgvector storage.
"""

import logging
from typing import Optional, Union
from functools import lru_cache

import numpy as np

from lib.config import config

logger = logging.getLogger(__name__)

# Global embedder instance
_embedder = None


class Embedder:
    """
    BGE-M3 embedding model wrapper.

    Usage:
        embedder = Embedder()
        embeddings = embedder.encode(["text1", "text2"])
        # embeddings.shape = (2, 1024)
    """

    def __init__(
        self,
        model_name: str = None,
        use_fp16: bool = True,
        device: str = None,
    ):
        """
        Initialize the embedder.

        Args:
            model_name: Model name (default: BAAI/bge-m3)
            use_fp16: Use FP16 for faster inference
            device: Device to use (auto-detected if None)
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.use_fp16 = use_fp16
        self.device = device
        self._model = None

        logger.info(f"Initializing embedder with model: {self.model_name}")

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load the BGE-M3 model."""
        try:
            from FlagEmbedding import BGEM3FlagModel

            logger.info(f"Loading {self.model_name}...")
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device,
            )
            logger.info(f"Model loaded successfully")

        except ImportError:
            logger.warning(
                "FlagEmbedding not installed. Falling back to sentence-transformers."
            )
            self._load_fallback_model()

    def _load_fallback_model(self):
        """Load fallback model using sentence-transformers."""
        from sentence_transformers import SentenceTransformer

        # Use a compatible model
        fallback_model = "BAAI/bge-large-en-v1.5"
        logger.info(f"Loading fallback model: {fallback_model}")
        self._model = SentenceTransformer(fallback_model)
        self._is_fallback = True

    def encode(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 32,
        max_length: int = 8192,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            max_length: Maximum token length
            normalize: Whether to L2-normalize embeddings

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([])

        # Check for fallback model
        if hasattr(self, "_is_fallback") and self._is_fallback:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 100,
            )
            return embeddings

        # BGE-M3 encoding
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        embeddings = output["dense_vecs"]

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-9)

        return embeddings

    def encode_query(
        self,
        query: str,
        max_length: int = 512,
    ) -> np.ndarray:
        """
        Encode a query (optimized for retrieval).

        Args:
            query: Query text
            max_length: Maximum token length

        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        embedding = self.encode([query], max_length=max_length)
        return embedding[0]

    def similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.

        Args:
            query_embedding: Query embedding (1D array)
            doc_embeddings: Document embeddings (2D array)

        Returns:
            Similarity scores (1D array)
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Cosine similarity
        scores = np.dot(doc_embeddings, query_embedding.T).flatten()
        return scores

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return config.EMBEDDING_DIM


@lru_cache(maxsize=1)
def get_embedder(
    model_name: str = None,
    use_fp16: bool = True,
) -> Embedder:
    """
    Get the global embedder instance (singleton).

    Args:
        model_name: Model name (default: from config)
        use_fp16: Use FP16 for faster inference

    Returns:
        Embedder instance
    """
    return Embedder(model_name=model_name, use_fp16=use_fp16)


def compute_similarity(
    text1: str,
    text2: str,
    embedder: Optional[Embedder] = None,
) -> float:
    """
    Compute similarity between two texts.

    Args:
        text1: First text
        text2: Second text
        embedder: Embedder instance (uses global if None)

    Returns:
        Cosine similarity score
    """
    embedder = embedder or get_embedder()
    embeddings = embedder.encode([text1, text2])
    return float(np.dot(embeddings[0], embeddings[1]))
