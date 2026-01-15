"""
Novelty checking for BridgeMine.

Validates that identified gaps are actually novel by checking:
1. Local knowledge base (Postgres, ChromaDB)
2. PubMed (recent publications)
3. Semantic Scholar (broader coverage)

Uses Reciprocal Rank Fusion (RRF) to combine signals from multiple sources.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx

from lib.config import config
from lib.db.postgres import get_pg_pool
from lib.bridgemine.gap_detection import GapCandidate

logger = logging.getLogger(__name__)

# RRF constant
RRF_K = 60


@dataclass
class NoveltyResult:
    """Result of novelty check for a gap candidate."""

    candidate: GapCandidate
    is_novel: bool
    novelty_score: float  # 0-1, higher = more novel

    # Source-specific results
    local_hits: int = 0
    pubmed_hits: int = 0
    semantic_scholar_hits: int = 0
    openalex_hits: int = 0

    # Prior art details
    prior_art: list[dict] = field(default_factory=list)
    rrf_confidence: float = 0.0

    reasoning: Optional[str] = None


class NoveltyChecker:
    """
    Check novelty of gap candidates using multiple sources.

    Uses RRF fusion to combine evidence from:
    - Local Postgres/pgvector search
    - PubMed API
    - Semantic Scholar API

    A gap is considered novel if:
    - Few papers (<10) mention both method and problem together
    - No recent papers (past 2 years) directly address the combination
    - RRF confidence is low

    Usage:
        checker = NoveltyChecker()
        result = checker.check(gap_candidate)
        if result.is_novel:
            print(f"Novel gap found! Score: {result.novelty_score}")
    """

    def __init__(
        self,
        pubmed_limit: int = 50,
        s2_limit: int = 20,
        year_weight: float = 0.3,
    ):
        """
        Initialize novelty checker.

        Args:
            pubmed_limit: Max PubMed results to fetch
            s2_limit: Max Semantic Scholar results
            year_weight: Weight given to recency (0-1)
        """
        self.pubmed_limit = pubmed_limit
        self.s2_limit = s2_limit
        self.year_weight = year_weight

    def check(self, candidate: GapCandidate) -> NoveltyResult:
        """
        Check novelty of a gap candidate.

        Uses 4 sources for comprehensive novelty assessment:
        - Local knowledge base (Postgres FTS)
        - PubMed (biomedical literature)
        - Semantic Scholar (broad coverage)
        - OpenAlex (fast, comprehensive, recent focus)

        Args:
            candidate: GapCandidate to check

        Returns:
            NoveltyResult with novelty assessment
        """
        # Build search query
        query = f"{candidate.method_name} {candidate.target_problem}"
        domain_query = f"{candidate.method_name} {candidate.target_domain}"

        # Check all sources
        local_hits = self._check_local(query)
        pubmed_hits, pubmed_papers = self._check_pubmed(domain_query)
        s2_hits, s2_papers = self._check_semantic_scholar(domain_query)
        openalex_hits, openalex_papers = self._check_openalex(domain_query)

        # Combine prior art from all external sources
        prior_art = self._merge_prior_art(pubmed_papers, s2_papers, openalex_papers)

        # Calculate RRF confidence with 4 sources
        rrf_confidence = self._calculate_rrf_confidence(
            local_hits, pubmed_hits, s2_hits, openalex_hits
        )

        # Calculate novelty score
        novelty_score = self._calculate_novelty_score(
            local_hits, pubmed_hits, s2_hits, openalex_hits, prior_art
        )

        # Determine if novel
        is_novel = (
            novelty_score > 0.5
            and len(prior_art) < 10
            and rrf_confidence < 0.5
        )

        # Update candidate with novelty metrics
        candidate.novelty_score = novelty_score
        candidate.prior_art_count = len(prior_art)
        candidate.rrf_confidence = rrf_confidence

        return NoveltyResult(
            candidate=candidate,
            is_novel=is_novel,
            novelty_score=novelty_score,
            local_hits=local_hits,
            pubmed_hits=pubmed_hits,
            semantic_scholar_hits=s2_hits,
            openalex_hits=openalex_hits,
            prior_art=prior_art[:20],  # Limit stored prior art
            rrf_confidence=rrf_confidence,
            reasoning=self._generate_reasoning(
                is_novel, novelty_score, local_hits, pubmed_hits, s2_hits, openalex_hits
            ),
        )

    def _check_local(self, query: str) -> int:
        """Check local knowledge base for query matches."""
        try:
            pool = get_pg_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COUNT(*) as count
                        FROM passages p
                        WHERE p.search_vector @@ websearch_to_tsquery('english', %s)
                          AND (p.is_superseded = FALSE OR p.is_superseded IS NULL)
                        """,
                        (query,),
                    )
                    row = cur.fetchone()
                    return row["count"] if row else 0

        except Exception as e:
            logger.debug(f"Local check failed: {e}")
            return 0

    def _check_pubmed(self, query: str) -> tuple[int, list[dict]]:
        """Check PubMed for relevant papers."""
        try:
            # Search PubMed
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": self.pubmed_limit,
                "retmode": "json",
                "sort": "relevance",
            }

            with httpx.Client(timeout=15.0) as client:
                response = client.get(search_url, params=params)
                response.raise_for_status()
                data = response.json()

            result = data.get("esearchresult", {})
            count = int(result.get("count", 0))
            ids = result.get("idlist", [])

            if not ids:
                return count, []

            # Fetch summaries for top papers
            summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params = {
                "db": "pubmed",
                "id": ",".join(ids[:10]),
                "retmode": "json",
            }

            time.sleep(0.5)  # Be nice to NCBI
            response = client.get(summary_url, params=params)
            data = response.json()

            papers = []
            result = data.get("result", {})
            for pmid in ids[:10]:
                if pmid in result:
                    paper = result[pmid]
                    papers.append({
                        "pmid": pmid,
                        "title": paper.get("title", ""),
                        "year": self._extract_year(paper.get("pubdate", "")),
                        "source": "pubmed",
                    })

            return count, papers

        except Exception as e:
            logger.debug(f"PubMed check failed: {e}")
            return 0, []

    def _check_semantic_scholar(self, query: str) -> tuple[int, list[dict]]:
        """Check Semantic Scholar for relevant papers."""
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": self.s2_limit,
                "fields": "title,year,paperId,citationCount",
            }

            headers = {}
            if config.S2_API_KEY:
                headers["x-api-key"] = config.S2_API_KEY

            with httpx.Client(timeout=15.0) as client:
                response = client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

            total = data.get("total", 0)
            papers = []

            for paper in data.get("data", []):
                papers.append({
                    "s2_id": paper.get("paperId"),
                    "title": paper.get("title", ""),
                    "year": paper.get("year"),
                    "citations": paper.get("citationCount", 0),
                    "source": "semantic_scholar",
                })

            return total, papers

        except Exception as e:
            logger.debug(f"Semantic Scholar check failed: {e}")
            return 0, []

    def _check_openalex(self, query: str) -> tuple[int, list[dict]]:
        """
        Check OpenAlex for relevant works.

        OpenAlex is faster and has better coverage than Semantic Scholar,
        making it ideal for novelty assessment.
        """
        try:
            url = "https://api.openalex.org/works"
            params = {
                "search": query,
                "per_page": 20,
                "mailto": config.OPENALEX_EMAIL,
                "filter": "publication_year:>2020",  # Focus on recent work
                "select": "id,title,publication_year,cited_by_count,doi",
            }

            with httpx.Client(timeout=15.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            count = data.get("meta", {}).get("count", 0)
            papers = []

            for work in data.get("results", []):
                papers.append({
                    "openalex_id": work.get("id"),
                    "title": work.get("title", ""),
                    "year": work.get("publication_year"),
                    "citations": work.get("cited_by_count", 0),
                    "doi": work.get("doi"),
                    "source": "openalex",
                })

            return count, papers

        except Exception as e:
            logger.debug(f"OpenAlex check failed: {e}")
            return 0, []

    def _extract_year(self, pubdate: str) -> Optional[int]:
        """Extract year from PubMed pubdate string."""
        if not pubdate:
            return None
        try:
            # Format: "2024 Jan 15" or "2024"
            year_str = pubdate.split()[0]
            return int(year_str)
        except (ValueError, IndexError):
            return None

    def _merge_prior_art(
        self,
        pubmed_papers: list[dict],
        s2_papers: list[dict],
        openalex_papers: list[dict] = None,
    ) -> list[dict]:
        """Merge and deduplicate prior art from multiple sources."""
        openalex_papers = openalex_papers or []

        # Simple merge - could be smarter with title matching
        merged = []
        seen_titles = set()

        for paper in pubmed_papers + s2_papers + openalex_papers:
            title_lower = paper.get("title", "").lower()
            if title_lower and title_lower not in seen_titles:
                merged.append(paper)
                seen_titles.add(title_lower)

        # Sort by year (recent first)
        merged.sort(
            key=lambda x: x.get("year") or 0,
            reverse=True,
        )

        return merged

    def _calculate_rrf_confidence(
        self,
        local_hits: int,
        pubmed_hits: int,
        s2_hits: int,
        openalex_hits: int = 0,
    ) -> float:
        """
        Calculate RRF confidence score using 4 sources.

        Higher score = more existing work found = less novel.
        """
        # Normalize hit counts to ranks (more hits = higher rank = lower value)
        scores = []

        if local_hits > 0:
            # Convert count to pseudo-rank
            local_rank = 1 / (1 + local_hits / 100)  # Normalize to ~0-1
            scores.append(1.0 / (RRF_K + local_rank * 100))

        if pubmed_hits > 0:
            pubmed_rank = 1 / (1 + pubmed_hits / 1000)
            scores.append(1.0 / (RRF_K + pubmed_rank * 100))

        if s2_hits > 0:
            s2_rank = 1 / (1 + s2_hits / 1000)
            scores.append(1.0 / (RRF_K + s2_rank * 100))

        if openalex_hits > 0:
            # OpenAlex typically has higher counts, normalize accordingly
            openalex_rank = 1 / (1 + openalex_hits / 2000)
            scores.append(1.0 / (RRF_K + openalex_rank * 100))

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def _calculate_novelty_score(
        self,
        local_hits: int,
        pubmed_hits: int,
        s2_hits: int,
        openalex_hits: int,
        prior_art: list[dict],
    ) -> float:
        """
        Calculate novelty score (0-1) using 4 sources.

        Higher = more novel (less existing work).
        """
        # Base score from hit counts (weight external sources more)
        total_hits = local_hits + pubmed_hits + s2_hits + (openalex_hits * 0.5)  # OpenAlex inflated

        if total_hits == 0:
            base_score = 1.0
        elif total_hits < 10:
            base_score = 0.9
        elif total_hits < 50:
            base_score = 0.7
        elif total_hits < 200:
            base_score = 0.4
        else:
            base_score = 0.1

        # Adjust for recency of prior art
        current_year = datetime.now().year
        recent_papers = sum(
            1 for p in prior_art
            if p.get("year") and p["year"] >= current_year - 2
        )

        recency_penalty = min(0.3, recent_papers * 0.05)
        novelty_score = max(0.0, base_score - recency_penalty)

        return novelty_score

    def _generate_reasoning(
        self,
        is_novel: bool,
        novelty_score: float,
        local_hits: int,
        pubmed_hits: int,
        s2_hits: int,
        openalex_hits: int = 0,
    ) -> str:
        """Generate human-readable reasoning for novelty assessment."""
        sources = f"{local_hits} local, {pubmed_hits} PubMed, {s2_hits} S2, {openalex_hits} OpenAlex"

        if is_novel:
            return (
                f"Novel gap identified (score: {novelty_score:.2f}). "
                f"Found {sources} hits. "
                "Low existing coverage suggests opportunity."
            )
        else:
            return (
                f"Not novel (score: {novelty_score:.2f}). "
                f"Found {sources} hits. "
                "Substantial existing work in this area."
            )

    def check_batch(
        self, candidates: list[GapCandidate], delay: float = 0.5
    ) -> list[NoveltyResult]:
        """
        Check novelty of multiple candidates.

        Args:
            candidates: List of gap candidates
            delay: Delay between API calls (seconds)

        Returns:
            List of NoveltyResult objects
        """
        results = []

        for i, candidate in enumerate(candidates):
            logger.info(f"Checking novelty {i+1}/{len(candidates)}: {candidate.method_name}")

            result = self.check(candidate)
            results.append(result)

            if i < len(candidates) - 1:
                time.sleep(delay)

        # Sort by novelty score
        results.sort(key=lambda x: x.novelty_score, reverse=True)

        return results


def check_novelty(candidate: GapCandidate, **kwargs) -> NoveltyResult:
    """
    Convenience function for novelty checking.

    Args:
        candidate: Gap candidate to check
        **kwargs: Arguments for NoveltyChecker

    Returns:
        NoveltyResult
    """
    checker = NoveltyChecker(**kwargs)
    return checker.check(candidate)
