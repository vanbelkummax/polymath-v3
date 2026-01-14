"""
Document identity management for Polymath v3.

Provides deterministic document IDs based on content identifiers.
Priority: DOI > PMID > arXiv ID > title hash

This ensures:
1. Same document always gets same ID (idempotent ingestion)
2. Re-ingestion updates rather than duplicates
3. Consistent cross-database references
"""

import hashlib
import re
import unicodedata
import uuid
from typing import Optional

# Namespace UUID for deterministic UUIDv5 generation
POLYMATH_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def normalize_title(title: str) -> str:
    """
    Normalize a title for hashing.

    Transformations:
    - Lowercase
    - Remove accents/diacritics
    - Remove punctuation
    - Collapse whitespace
    - Strip

    Examples:
        "The Transformer Architecture" -> "the transformer architecture"
        "BERT: Pre-training..." -> "bert pretraining"
    """
    if not title:
        return ""

    # Lowercase
    text = title.lower()

    # Remove accents (é -> e)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Remove punctuation except spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse whitespace
    text = " ".join(text.split())

    return text.strip()


def get_title_hash(title: str) -> str:
    """
    Generate a SHA256 hash of the normalized title.

    Args:
        title: Paper title

    Returns:
        64-character hex string
    """
    normalized = normalize_title(title)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_doi(doi: str) -> str:
    """
    Normalize a DOI to lowercase canonical form.

    Handles:
    - Full URLs (https://doi.org/10.1234/...)
    - doi: prefix
    - Whitespace
    """
    if not doi:
        return ""

    doi = doi.strip().lower()

    # Remove URL prefix
    for prefix in ["https://doi.org/", "http://doi.org/", "doi.org/", "doi:"]:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]

    return doi.strip()


def normalize_pmid(pmid: str) -> str:
    """Normalize a PubMed ID to numeric string."""
    if not pmid:
        return ""

    # Extract digits only
    return "".join(c for c in str(pmid) if c.isdigit())


def normalize_arxiv_id(arxiv_id: str) -> str:
    """
    Normalize an arXiv ID.

    Handles:
    - Full URLs (https://arxiv.org/abs/2301.12345)
    - arXiv: prefix
    - Version suffixes (v1, v2)
    """
    if not arxiv_id:
        return ""

    arxiv_id = arxiv_id.strip()

    # Remove URL prefix
    for prefix in ["https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv:"]:
        if arxiv_id.lower().startswith(prefix.lower()):
            arxiv_id = arxiv_id[len(prefix):]

    # Remove version suffix for canonical form
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

    return arxiv_id.strip()


def get_doc_id(
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    title: Optional[str] = None,
) -> uuid.UUID:
    """
    Generate a deterministic document ID.

    Priority order:
    1. DOI (most stable, cross-database)
    2. PMID (stable for PubMed content)
    3. arXiv ID (stable for preprints)
    4. Title hash (fallback)

    Args:
        doi: Digital Object Identifier
        pmid: PubMed ID
        arxiv_id: arXiv identifier
        title: Paper title (required if no other ID)

    Returns:
        UUID (version 5, deterministic)

    Raises:
        ValueError: If no identifier provided
    """
    # Try identifiers in priority order
    if doi:
        normalized = normalize_doi(doi)
        if normalized:
            return uuid.uuid5(POLYMATH_NAMESPACE, f"doi:{normalized}")

    if pmid:
        normalized = normalize_pmid(pmid)
        if normalized:
            return uuid.uuid5(POLYMATH_NAMESPACE, f"pmid:{normalized}")

    if arxiv_id:
        normalized = normalize_arxiv_id(arxiv_id)
        if normalized:
            return uuid.uuid5(POLYMATH_NAMESPACE, f"arxiv:{normalized}")

    if title:
        title_hash = get_title_hash(title)
        return uuid.uuid5(POLYMATH_NAMESPACE, f"title:{title_hash}")

    raise ValueError("At least one identifier (doi, pmid, arxiv_id, or title) required")


def get_passage_id(doc_id: uuid.UUID, char_start: int, char_end: int) -> uuid.UUID:
    """
    Generate a deterministic passage ID.

    Based on document ID and character offsets for reproducibility.

    Args:
        doc_id: Parent document UUID
        char_start: Starting character offset
        char_end: Ending character offset

    Returns:
        UUID (version 5, deterministic)
    """
    key = f"{doc_id}:{char_start}:{char_end}"
    return uuid.uuid5(POLYMATH_NAMESPACE, key)


def extract_doi_from_text(text: str) -> Optional[str]:
    """
    Extract DOI from text using regex.

    Patterns:
    - 10.XXXX/... (standard DOI)
    - doi.org/10.XXXX/...
    - doi:10.XXXX/...
    """
    patterns = [
        r"10\.\d{4,}/[^\s\"\'\]\)>]+",
        r"doi\.org/(10\.\d{4,}/[^\s\"\'\]\)>]+)",
        r"doi:\s*(10\.\d{4,}/[^\s\"\'\]\)>]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            doi = match.group(1) if match.lastindex else match.group(0)
            return normalize_doi(doi)

    return None


def extract_arxiv_from_text(text: str) -> Optional[str]:
    """
    Extract arXiv ID from text.

    Patterns:
    - arxiv:YYMM.NNNNN
    - arxiv.org/abs/YYMM.NNNNN
    - arXiv:hep-th/NNNNNNN (old format)
    """
    patterns = [
        r"arxiv[:\s]+(\d{4}\.\d{4,5})",
        r"arxiv\.org/abs/(\d{4}\.\d{4,5})",
        r"arxiv[:\s]+([a-z-]+/\d{7})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return normalize_arxiv_id(match.group(1))

    return None
