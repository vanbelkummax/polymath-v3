"""
Neural reranking for Polymath v3.

Uses cross-encoder models for high-quality reranking of search results.
Supports multiple backends: sentence-transformers, Cohere, BGE-reranker.
"""

import logging
from typing import Optional

from lib.config import config
from lib.search.hybrid_search import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """
    Neural reranker for search results.

    Uses cross-encoder models that score (query, passage) pairs
    for more accurate relevance than bi-encoder retrieval alone.

    Usage:
        reranker = Reranker()
        reranked = reranker.rerank(query, results, top_k=10)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize reranker.

        Args:
            model_name: Cross-encoder model name
            device: Device for inference (auto-detected if None)
            batch_size: Batch size for scoring
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

        # Auto-detect device
        if device is None:
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

    @property
    def model(self):
        """Lazy load reranker model."""
        if self._model is None:
            logger.info(f"Loading reranker model: {self.model_name}")

            try:
                # Try FlagEmbedding first (better for BGE models)
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(
                    self.model_name,
                    use_fp16=self.device == "cuda",
                )
                self._model_type = "flag"
                logger.info("Using FlagReranker backend")

            except ImportError:
                # Fall back to sentence-transformers
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                )
                self._model_type = "cross_encoder"
                logger.info("Using sentence-transformers CrossEncoder backend")

        return self._model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Search query
            results: List of SearchResult objects
            top_k: Return top K results (all if None)

        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return []

        # Prepare pairs for scoring
        pairs = [(query, r.passage_text) for r in results]

        # Score pairs
        try:
            if self._model_type == "flag":
                scores = self.model.compute_score(pairs)
                # FlagReranker may return single value for single pair
                if not isinstance(scores, list):
                    scores = [scores]
            else:
                scores = self.model.predict(pairs, batch_size=self.batch_size)

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:top_k] if top_k else results

        # Update scores and sort
        for result, score in zip(results, scores):
            result.score = float(score)

        reranked = sorted(results, key=lambda x: x.score, reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def score_pairs(
        self,
        query: str,
        passages: list[str],
    ) -> list[float]:
        """
        Score query-passage pairs.

        Args:
            query: Search query
            passages: List of passage texts

        Returns:
            List of relevance scores
        """
        if not passages:
            return []

        pairs = [(query, p) for p in passages]

        try:
            if self._model_type == "flag":
                scores = self.model.compute_score(pairs)
                if not isinstance(scores, list):
                    scores = [scores]
            else:
                scores = self.model.predict(pairs, batch_size=self.batch_size)

            return [float(s) for s in scores]

        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return [0.0] * len(passages)


class CohereReranker:
    """
    Cohere API-based reranker.

    Uses Cohere's rerank endpoint for high-quality reranking.
    Requires COHERE_API_KEY environment variable.
    """

    def __init__(self, model: str = "rerank-english-v3.0"):
        """
        Initialize Cohere reranker.

        Args:
            model: Cohere rerank model name
        """
        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy load Cohere client."""
        if self._client is None:
            import cohere

            api_key = config.COHERE_API_KEY
            if not api_key:
                raise ValueError("COHERE_API_KEY not configured")

            self._client = cohere.Client(api_key)

        return self._client

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Rerank using Cohere API.

        Args:
            query: Search query
            results: List of SearchResult objects
            top_k: Return top K results

        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return []

        documents = [r.passage_text for r in results]
        top_n = top_k or len(results)

        try:
            response = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_n=top_n,
            )

            # Map back to results
            reranked = []
            for item in response.results:
                result = results[item.index]
                result.score = item.relevance_score
                reranked.append(result)

            return reranked

        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            return results[:top_k] if top_k else results


def get_reranker(backend: str = "local") -> Reranker:
    """
    Factory function for reranker instances.

    Args:
        backend: "local" for sentence-transformers, "cohere" for API

    Returns:
        Reranker instance
    """
    if backend == "cohere":
        return CohereReranker()
    return Reranker()
