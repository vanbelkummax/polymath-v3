"""
Metadata resolution for Polymath v3.

Priority-ordered metadata lookup:
1. Zotero CSV (fast, high coverage from local library)
2. pdf2doi (extract DOI from PDF binary)
3. CrossRef API (authoritative for DOIs)
4. arXiv API (for preprints)
5. Filename parsing (last resort)
"""

import csv
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from functools import lru_cache

import httpx
from rapidfuzz import fuzz

from lib.config import config

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Resolved paper metadata."""

    title: str
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None

    # Identifiers
    doi: Optional[str] = None
    pmid: Optional[str] = None
    arxiv_id: Optional[str] = None
    openalex_id: Optional[str] = None

    # Content
    abstract: Optional[str] = None
    keywords: list[str] = field(default_factory=list)

    # Source tracking
    source_method: str = "unknown"  # zotero, pdf2doi, crossref, arxiv, filename
    confidence: float = 0.0

    # Zotero
    zotero_key: Optional[str] = None


class ZoteroIndex:
    """In-memory index of Zotero metadata for fast lookup."""

    def __init__(self, csv_path: Optional[Path] = None):
        self.entries: dict[str, dict] = {}  # title_lower -> metadata
        self.doi_index: dict[str, str] = {}  # doi -> title
        self.loaded = False

        if csv_path and csv_path.exists():
            self._load_csv(csv_path)

    def _load_csv(self, csv_path: Path):
        """Load Zotero export CSV into memory."""
        logger.info(f"Loading Zotero index from {csv_path}")

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    title = row.get("title") or row.get("zotero_title") or row.get("pg_title")
                    if not title:
                        continue

                    title_lower = title.lower().strip()
                    doi = row.get("doi") or row.get("new_doi")
                    pmid = row.get("pmid") or row.get("new_pmid")
                    year = row.get("year") or row.get("new_year")

                    entry = {
                        "title": title,
                        "doi": doi,
                        "pmid": pmid,
                        "year": int(year) if year and str(year).isdigit() else None,
                        "zotero_key": row.get("zotero_key") or row.get("doc_id"),
                        "file_path": row.get("file_path"),
                    }

                    self.entries[title_lower] = entry

                    if doi:
                        self.doi_index[doi.lower()] = title_lower

            self.loaded = True
            logger.info(f"Loaded {len(self.entries)} Zotero entries")

        except Exception as e:
            logger.error(f"Failed to load Zotero CSV: {e}")

    def lookup_by_title(
        self, title: str, min_similarity: float = 0.90
    ) -> Optional[tuple[dict, float]]:
        """
        Fuzzy match title against Zotero index.

        Returns:
            Tuple of (metadata dict, similarity score) or None
        """
        if not self.loaded or not title:
            return None

        title_lower = title.lower().strip()

        # Exact match first
        if title_lower in self.entries:
            return self.entries[title_lower], 1.0

        # Fuzzy match
        best_match = None
        best_score = 0.0

        for stored_title, entry in self.entries.items():
            score = fuzz.ratio(title_lower, stored_title) / 100.0
            if score > best_score and score >= min_similarity:
                best_score = score
                best_match = entry

        if best_match:
            return best_match, best_score

        return None

    def lookup_by_doi(self, doi: str) -> Optional[dict]:
        """Look up by DOI."""
        if not self.loaded or not doi:
            return None

        doi_lower = doi.lower().strip()
        title_key = self.doi_index.get(doi_lower)
        if title_key:
            return self.entries.get(title_key)
        return None


# Global Zotero index (loaded once)
_zotero_index: Optional[ZoteroIndex] = None


def get_zotero_index() -> ZoteroIndex:
    """Get or create the global Zotero index."""
    global _zotero_index
    if _zotero_index is None:
        _zotero_index = ZoteroIndex(config.ZOTERO_CSV_PATH)
    return _zotero_index


def lookup_crossref(doi: str) -> Optional[PaperMetadata]:
    """
    Look up metadata from CrossRef by DOI.

    Args:
        doi: Digital Object Identifier

    Returns:
        PaperMetadata or None
    """
    if not doi:
        return None

    url = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": f"Polymath/3.0 (mailto:{config.OPENALEX_EMAIL})"}

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

        work = data.get("message", {})

        # Extract authors
        authors = []
        for author in work.get("author", []):
            name_parts = []
            if author.get("given"):
                name_parts.append(author["given"])
            if author.get("family"):
                name_parts.append(author["family"])
            if name_parts:
                authors.append(" ".join(name_parts))

        # Extract year
        year = None
        for date_field in ["published-print", "published-online", "created"]:
            date_parts = work.get(date_field, {}).get("date-parts", [[]])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
                break

        # Extract title
        titles = work.get("title", [])
        title = titles[0] if titles else None

        if not title:
            return None

        return PaperMetadata(
            title=title,
            authors=authors,
            year=year,
            venue=work.get("container-title", [None])[0],
            doi=doi,
            abstract=work.get("abstract"),
            source_method="crossref",
            confidence=0.95,
        )

    except Exception as e:
        logger.debug(f"CrossRef lookup failed for {doi}: {e}")
        return None


def lookup_arxiv(arxiv_id: str) -> Optional[PaperMetadata]:
    """
    Look up metadata from arXiv API.

    Args:
        arxiv_id: arXiv identifier (e.g., "2301.12345")

    Returns:
        PaperMetadata or None
    """
    if not arxiv_id:
        return None

    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()

        # Parse Atom XML response
        import xml.etree.ElementTree as ET

        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        title = entry.findtext("atom:title", "", ns).strip()
        if not title:
            return None

        # Extract authors
        authors = [
            author.findtext("atom:name", "", ns)
            for author in entry.findall("atom:author", ns)
        ]

        # Extract year from published date
        published = entry.findtext("atom:published", "", ns)
        year = int(published[:4]) if published else None

        return PaperMetadata(
            title=title,
            authors=authors,
            year=year,
            arxiv_id=arxiv_id,
            abstract=entry.findtext("atom:summary", "", ns).strip(),
            source_method="arxiv",
            confidence=0.90,
        )

    except Exception as e:
        logger.debug(f"arXiv lookup failed for {arxiv_id}: {e}")
        return None


def extract_from_pdf(pdf_path: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Extract DOI and arXiv ID from PDF.

    Uses pdf2doi library for reliable extraction.

    Returns:
        Tuple of (doi, arxiv_id)
    """
    doi = None
    arxiv_id = None

    try:
        from pdf2doi import pdf2doi as p2d

        result = p2d.pdf2doi(str(pdf_path))
        if result:
            identifier = result.get("identifier")
            id_type = result.get("identifier_type")

            if id_type == "DOI":
                doi = identifier
            elif id_type == "arxiv":
                arxiv_id = identifier

    except ImportError:
        logger.warning("pdf2doi not installed, skipping PDF identifier extraction")
    except Exception as e:
        logger.debug(f"pdf2doi failed for {pdf_path}: {e}")

    return doi, arxiv_id


def extract_from_filename(filename: str) -> PaperMetadata:
    """
    Extract metadata from filename (last resort).

    Patterns:
    - "2023_AuthorName_Title.pdf"
    - "Author2023_Title.pdf"
    - "Title_2023.pdf"

    Returns:
        PaperMetadata with low confidence
    """
    name = Path(filename).stem

    # Try to extract year
    year_match = re.search(r"(19|20)\d{2}", name)
    year = int(year_match.group(0)) if year_match else None

    # Clean up title
    title = name
    title = re.sub(r"(19|20)\d{2}", "", title)  # Remove year
    title = re.sub(r"[_-]+", " ", title)  # Replace separators
    title = re.sub(r"\s+", " ", title).strip()  # Clean whitespace

    return PaperMetadata(
        title=title or name,
        year=year,
        source_method="filename",
        confidence=0.3,
    )


class MetadataResolver:
    """
    Resolve paper metadata using priority-ordered sources.

    Priority:
    1. Zotero CSV (if match confidence > 0.95)
    2. pdf2doi extraction
    3. CrossRef lookup (if DOI found)
    4. arXiv lookup (if arXiv ID found)
    5. Zotero relaxed match (confidence > 0.80)
    6. Filename parsing
    """

    def __init__(self, zotero_csv_path: Optional[Path] = None):
        """
        Initialize resolver.

        Args:
            zotero_csv_path: Path to Zotero export CSV
        """
        self.zotero = ZoteroIndex(zotero_csv_path or config.ZOTERO_CSV_PATH)

    def resolve(self, pdf_path: Path) -> PaperMetadata:
        """
        Resolve metadata for a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Best available metadata
        """
        filename = pdf_path.name

        # Step 1: Try Zotero with high confidence
        title_guess = self._guess_title_from_filename(filename)
        if title_guess:
            zotero_match = self.zotero.lookup_by_title(title_guess, min_similarity=0.95)
            if zotero_match:
                entry, confidence = zotero_match
                return self._zotero_to_metadata(entry, confidence)

        # Step 2: Extract identifiers from PDF
        doi, arxiv_id = extract_from_pdf(pdf_path)

        # Step 3: Try CrossRef if DOI found
        if doi:
            metadata = lookup_crossref(doi)
            if metadata:
                # Also check Zotero for enrichment
                zotero_entry = self.zotero.lookup_by_doi(doi)
                if zotero_entry:
                    metadata.zotero_key = zotero_entry.get("zotero_key")
                    if not metadata.pmid and zotero_entry.get("pmid"):
                        metadata.pmid = zotero_entry["pmid"]
                return metadata

        # Step 4: Try arXiv if arXiv ID found
        if arxiv_id:
            metadata = lookup_arxiv(arxiv_id)
            if metadata:
                return metadata

        # Step 5: Zotero relaxed match
        if title_guess:
            zotero_match = self.zotero.lookup_by_title(title_guess, min_similarity=0.80)
            if zotero_match:
                entry, confidence = zotero_match
                return self._zotero_to_metadata(entry, confidence)

        # Step 6: Filename fallback
        return extract_from_filename(filename)

    def _guess_title_from_filename(self, filename: str) -> Optional[str]:
        """Extract potential title from filename."""
        name = Path(filename).stem

        # Remove common prefixes/suffixes
        name = re.sub(r"^(\d+_|\d+-|paper_)", "", name)
        name = re.sub(r"(_final|_v\d+|_draft)$", "", name, flags=re.IGNORECASE)

        # Replace separators with spaces
        name = re.sub(r"[_-]+", " ", name)
        name = re.sub(r"\s+", " ", name).strip()

        return name if len(name) > 10 else None

    def _zotero_to_metadata(self, entry: dict, confidence: float) -> PaperMetadata:
        """Convert Zotero entry to PaperMetadata."""
        return PaperMetadata(
            title=entry.get("title", "Unknown"),
            year=entry.get("year"),
            doi=entry.get("doi"),
            pmid=entry.get("pmid"),
            zotero_key=entry.get("zotero_key"),
            source_method="zotero",
            confidence=confidence,
        )
