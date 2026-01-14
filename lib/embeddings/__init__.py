"""
Embedding generation for Polymath v3.

Uses BGE-M3 for high-quality 1024-dimensional embeddings.
"""

from .bge_m3 import get_embedder, Embedder

__all__ = ["get_embedder", "Embedder"]
