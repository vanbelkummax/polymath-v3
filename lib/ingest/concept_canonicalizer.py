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
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from lib.config import config
from lib.db.postgres import get_pg_pool

logger = logging.getLogger(__name__)


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

    This is slower but more accurate for catching variants like:
    - "Cell2Location" vs "cell2location" vs "Cell2Loc"

    Returns:
        Dict with name, similarity, passage_id if match found
    """
    try:
        from lib.embeddings.bge_m3 import get_embedder

        embedder = get_embedder()
        name_embedding = embedder.encode([name])[0].tolist()

        pool = get_pg_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Find closest existing concept
                cur.execute(
                    """
                    SELECT DISTINCT concept_name, passage_id,
                           1 - (pc_embedding <=> %s::vector) as similarity
                    FROM passage_concepts pc
                    JOIN (
                        -- Get embedding for each unique concept
                        SELECT concept_name,
                               (SELECT embedding FROM passages
                                WHERE passage_id = pc2.passage_id LIMIT 1) as pc_embedding
                        FROM passage_concepts pc2
                        GROUP BY concept_name, passage_id
                    ) sub ON pc.concept_name = sub.concept_name
                    WHERE sub.pc_embedding IS NOT NULL
                    ORDER BY pc_embedding <=> %s::vector
                    LIMIT 1
                    """,
                    (name_embedding, name_embedding),
                )

                row = cur.fetchone()
                if row and row["similarity"] >= threshold:
                    return {
                        "name": row["concept_name"],
                        "similarity": row["similarity"],
                        "passage_id": row["passage_id"],
                    }

    except Exception as e:
        logger.debug(f"Embedding match failed: {e}")

    return None


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
