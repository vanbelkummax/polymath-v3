"""
Concept canonicalization for Polymath v3.

Prevents graph fragmentation by normalizing concept names before storage.
Addresses the "ST" vs "Spatial Transcriptomics" problem.

Two strategies:
1. Rule-based: Lowercase, stem, apply known aliases
2. Embedding-based: Match against existing concepts (optional, more accurate)
"""

import logging
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from lib.config import config
from lib.db.postgres import get_pg_pool

logger = logging.getLogger(__name__)


# =============================================================================
# Concept Centroid Cache (Solves N+1 Query Problem)
# =============================================================================
# Pre-loads top concept embeddings into memory for fast similarity matching.
# Avoids per-concept database queries during bulk ingestion.
# =============================================================================


class ConceptCentroidCache:
    """
    In-memory cache of concept embeddings for fast similarity matching.

    Loads top N concepts by mention count at startup, then uses
    numpy for vectorized cosine similarity (no DB queries).

    Usage:
        cache = get_concept_cache()
        match = cache.find_similar("Cell2Loc", threshold=0.92)
        if match:
            print(f"Matched: {match['name']} (similarity: {match['similarity']:.3f})")
    """

    def __init__(self, max_concepts: int = 2000):
        self.max_concepts = max_concepts
        self._names: list[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._loaded = False
        self._lock = threading.Lock()
        self._load_time = 0.0

    def _load(self) -> None:
        """Load concept embeddings from database."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            start = time.time()
            logger.info(f"Loading concept centroid cache (max {self.max_concepts})...")

            try:
                pool = get_pg_pool()
                with pool.connection() as conn:
                    with conn.cursor() as cur:
                        # Get top concepts with their average passage embedding
                        cur.execute(
                            """
                            WITH top_concepts AS (
                                SELECT concept_name, COUNT(*) as cnt
                                FROM passage_concepts
                                GROUP BY concept_name
                                ORDER BY cnt DESC
                                LIMIT %s
                            )
                            SELECT
                                tc.concept_name,
                                AVG(p.embedding)::vector as centroid
                            FROM top_concepts tc
                            JOIN passage_concepts pc ON tc.concept_name = pc.concept_name
                            JOIN passages p ON pc.passage_id = p.passage_id
                            WHERE p.embedding IS NOT NULL
                            GROUP BY tc.concept_name
                            """,
                            (self.max_concepts,),
                        )

                        names = []
                        embeddings = []

                        for row in cur:
                            if row["centroid"] is not None:
                                names.append(row["concept_name"])
                                embeddings.append(row["centroid"])

                        if embeddings:
                            self._names = names
                            self._embeddings = np.array(embeddings, dtype=np.float32)
                            # Normalize for cosine similarity
                            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
                            self._embeddings = self._embeddings / (norms + 1e-8)

                self._loaded = True
                self._load_time = time.time() - start
                logger.info(
                    f"Loaded {len(self._names)} concept centroids in {self._load_time:.2f}s"
                )

            except Exception as e:
                logger.warning(f"Failed to load concept cache: {e}")
                self._loaded = True  # Don't retry on failure
                self._names = []
                self._embeddings = None

    def find_similar(
        self,
        name: str,
        threshold: float = 0.92,
    ) -> Optional[dict]:
        """
        Find the most similar existing concept using cached embeddings.

        Args:
            name: Concept name to match
            threshold: Minimum cosine similarity

        Returns:
            Dict with name and similarity if match found, else None
        """
        self._load()

        if self._embeddings is None or len(self._names) == 0:
            return None

        try:
            from lib.embeddings.bge_m3 import get_embedder

            embedder = get_embedder()
            query_embedding = embedder.encode([name])[0].astype(np.float32)

            # Normalize query
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm

            # Vectorized cosine similarity (no DB query!)
            similarities = self._embeddings @ query_embedding

            best_idx = np.argmax(similarities)
            best_sim = float(similarities[best_idx])

            if best_sim >= threshold:
                return {
                    "name": self._names[best_idx],
                    "similarity": best_sim,
                }

        except Exception as e:
            logger.debug(f"Cache similarity search failed: {e}")

        return None

    def invalidate(self) -> None:
        """Invalidate cache (call after bulk inserts)."""
        with self._lock:
            self._loaded = False
            self._names = []
            self._embeddings = None
            logger.info("Concept centroid cache invalidated")


# Global cache instance
_concept_cache: Optional[ConceptCentroidCache] = None


def get_concept_cache() -> ConceptCentroidCache:
    """Get or create the global concept cache."""
    global _concept_cache
    if _concept_cache is None:
        _concept_cache = ConceptCentroidCache()
    return _concept_cache


# =============================================================================
# Known Aliases (Rule-Based Canonicalization)
# =============================================================================
# These are hand-curated mappings for common variants.
# Add new entries as you discover fragmentation in the graph.
# =============================================================================

KNOWN_ALIASES = {
    # Spatial transcriptomics variants
    "st": "spatial_transcriptomics",
    "spatial tx": "spatial_transcriptomics",
    "spatial omics": "spatial_transcriptomics",
    "spatialomics": "spatial_transcriptomics",
    "spatial-transcriptomics": "spatial_transcriptomics",

    # Single-cell variants
    "sc": "single_cell",
    "scrna": "single_cell_rna_seq",
    "scrnaseq": "single_cell_rna_seq",
    "sc-rna-seq": "single_cell_rna_seq",
    "single cell rna-seq": "single_cell_rna_seq",
    "single-cell rna-seq": "single_cell_rna_seq",

    # Common method abbreviations
    "ot": "optimal_transport",
    "gnn": "graph_neural_network",
    "gcn": "graph_convolutional_network",
    "vae": "variational_autoencoder",
    "gan": "generative_adversarial_network",
    "bert": "bidirectional_encoder_representations_from_transformers",
    "llm": "large_language_model",
    "cnn": "convolutional_neural_network",
    "rnn": "recurrent_neural_network",
    "lstm": "long_short_term_memory",
    "transformer": "transformer_architecture",

    # Biology terms
    "h&e": "hematoxylin_and_eosin",
    "he": "hematoxylin_and_eosin",
    "h and e": "hematoxylin_and_eosin",
    "ihc": "immunohistochemistry",
    "if": "immunofluorescence",
    "ffpe": "formalin_fixed_paraffin_embedded",
    "wsi": "whole_slide_image",
    "roi": "region_of_interest",
    "deg": "differentially_expressed_gene",
    "degs": "differentially_expressed_genes",
    "go": "gene_ontology",
    "kegg": "kyoto_encyclopedia_of_genes_and_genomes",

    # Platform names
    "visium": "10x_visium",
    "visium hd": "10x_visium_hd",
    "xenium": "10x_xenium",
    "merfish": "multiplexed_error_robust_fish",
    "seqfish": "sequential_fish",
    "slide-seq": "slide_seq",
    "slideseq": "slide_seq",
    "stereo-seq": "stereo_seq",
    "stereoseq": "stereo_seq",

    # Common metrics
    "pcc": "pearson_correlation_coefficient",
    "rmse": "root_mean_squared_error",
    "mse": "mean_squared_error",
    "auc": "area_under_curve",
    "auroc": "area_under_roc_curve",
    "f1": "f1_score",
}


@dataclass
class CanonicalConcept:
    """A canonicalized concept."""

    original: str
    canonical: str
    method: str  # "alias", "embedding", "lowercase", "unchanged"
    confidence: float = 1.0
    matched_existing_id: Optional[str] = None


def normalize_text(text: str) -> str:
    """Basic text normalization."""
    # Lowercase
    text = text.lower().strip()

    # Replace hyphens and underscores with spaces for matching
    text = re.sub(r"[-_]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def to_canonical_form(text: str) -> str:
    """Convert to canonical storage form (snake_case)."""
    # Lowercase and replace spaces/hyphens with underscores
    text = text.lower().strip()
    text = re.sub(r"[\s-]+", "_", text)

    # Remove non-alphanumeric except underscores
    text = re.sub(r"[^a-z0-9_]", "", text)

    # Collapse multiple underscores
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def canonicalize_concept(
    name: str,
    use_embedding_match: bool = False,
    similarity_threshold: float = 0.92,
) -> CanonicalConcept:
    """
    Canonicalize a concept name.

    Strategy:
    1. Check known aliases
    2. Optionally match against existing concepts via embedding
    3. Apply basic normalization

    Args:
        name: Raw concept name from extraction
        use_embedding_match: Whether to use embedding-based matching
        similarity_threshold: Minimum similarity for embedding match

    Returns:
        CanonicalConcept with canonical name and method used
    """
    original = name.strip()
    normalized = normalize_text(original)

    # Strategy 1: Check known aliases
    if normalized in KNOWN_ALIASES:
        return CanonicalConcept(
            original=original,
            canonical=KNOWN_ALIASES[normalized],
            method="alias",
            confidence=1.0,
        )

    # Strategy 2: Embedding-based matching (optional)
    if use_embedding_match:
        match = _find_existing_concept_by_embedding(original, similarity_threshold)
        if match:
            return CanonicalConcept(
                original=original,
                canonical=match["name"],
                method="embedding",
                confidence=match["similarity"],
                matched_existing_id=match.get("passage_id"),
            )

    # Strategy 3: Basic normalization
    canonical = to_canonical_form(original)

    return CanonicalConcept(
        original=original,
        canonical=canonical,
        method="lowercase",
        confidence=0.9,
    )


def _find_existing_concept_by_embedding(
    name: str,
    threshold: float = 0.92,
) -> Optional[dict]:
    """
    Find an existing concept by embedding similarity.

    Uses the in-memory centroid cache to avoid N+1 queries.
    This catches variants like "Cell2Location" vs "cell2location" vs "Cell2Loc".

    Returns:
        Dict with name, similarity if match found
    """
    cache = get_concept_cache()
    return cache.find_similar(name, threshold=threshold)


def canonicalize_concepts(
    concepts: list,
    use_embedding_match: bool = False,
) -> list:
    """
    Canonicalize a list of extracted concepts.

    Args:
        concepts: List of ExtractedConcept objects
        use_embedding_match: Whether to use embedding-based matching

    Returns:
        Same list with names updated to canonical form
    """
    for concept in concepts:
        result = canonicalize_concept(
            concept.name,
            use_embedding_match=use_embedding_match,
        )
        concept.name = result.canonical

    return concepts


@lru_cache(maxsize=1000)
def get_canonical_name(name: str) -> str:
    """
    Get canonical name for a concept (cached).

    Fast path for rule-based canonicalization only.
    Use canonicalize_concept() for full pipeline with embedding matching.

    Args:
        name: Raw concept name

    Returns:
        Canonical name
    """
    normalized = normalize_text(name)

    if normalized in KNOWN_ALIASES:
        return KNOWN_ALIASES[normalized]

    return to_canonical_form(name)
