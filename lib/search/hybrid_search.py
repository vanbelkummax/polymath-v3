"""
Hybrid search for Polymath v3.

Combines:
1. Vector similarity (pgvector HNSW)
2. Full-text search (PostgreSQL tsvector)
3. Graph-based retrieval (Neo4j optional)

Results fused using Reciprocal Rank Fusion (RRF).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from lib.config import config
from lib.db.postgres import get_pg_pool
from lib.embeddings.bge_m3 import get_embedder

logger = logging.getLogger(__name__)

# RRF constant (standard value)
RRF_K = 60


@dataclass
class SearchResult:
    """A single search result."""

    passage_id: str
    doc_id: str
    passage_text: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    section: Optional[str] = None
    doi: Optional[str] = None

    # Scoring
    score: float = 0.0
    vector_rank: Optional[int] = None
    fts_rank: Optional[int] = None
    graph_rank: Optional[int] = None


@dataclass
class SearchResponse:
    """Response from hybrid search."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    total_found: int = 0
    search_time_ms: float = 0.0


class HybridSearcher:
    """
    Hybrid search engine combining vector, FTS, and graph retrieval.

    Usage:
        searcher = HybridSearcher()
        response = searcher.search("spatial transcriptomics deconvolution", n=20)
        for result in response.results:
            print(f"{result.title}: {result.score:.3f}")
    """

    def __init__(
        self,
        vector_weight: float = 0.5,
        fts_weight: float = 0.3,
        graph_weight: float = 0.2,
        use_neo4j: bool = True,
    ):
        """
        Initialize hybrid searcher.

        Args:
            vector_weight: Weight for vector similarity (0-1)
            fts_weight: Weight for full-text search (0-1)
            graph_weight: Weight for graph retrieval (0-1)
            use_neo4j: Whether to include Neo4j graph results
        """
        self.vector_weight = vector_weight
        self.fts_weight = fts_weight
        self.graph_weight = graph_weight
        self.use_neo4j = use_neo4j

        self.embedder = get_embedder()
        self._neo4j_driver = None

    @property
    def neo4j_driver(self):
        """Lazy load Neo4j driver."""
        if self._neo4j_driver is None and self.use_neo4j:
            try:
                from lib.db.neo4j import get_neo4j_driver
                self._neo4j_driver = get_neo4j_driver()
            except Exception as e:
                logger.warning(f"Neo4j not available: {e}")
                self.use_neo4j = False
        return self._neo4j_driver

    def search(
        self,
        query: str,
        n: int = 20,
        vector_k: int = 100,
        fts_k: int = 100,
        rerank: bool = False,
        filters: Optional[dict] = None,
    ) -> SearchResponse:
        """
        Perform hybrid search.

        Args:
            query: Search query
            n: Number of results to return
            vector_k: Number of vector candidates
            fts_k: Number of FTS candidates
            rerank: Whether to apply neural reranking
            filters: Optional filters (year_min, year_max, doi, etc.)

        Returns:
            SearchResponse with ranked results
        """
        import time

        start_time = time.time()

        # Step 1: Get candidates from each source
        vector_results = self._vector_search(query, k=vector_k, filters=filters)
        fts_results = self._fts_search(query, k=fts_k, filters=filters)

        graph_results = []
        if self.use_neo4j and self.graph_weight > 0:
            graph_results = self._graph_search(query, k=vector_k, filters=filters)

        # Step 2: Fuse with RRF
        fused = self._rrf_fusion(
            vector_results=vector_results,
            fts_results=fts_results,
            graph_results=graph_results,
        )

        # Step 3: Rerank if requested
        if rerank:
            from lib.search.reranker import get_reranker_singleton

            reranker = get_reranker_singleton()
            fused = reranker.rerank(query, fused, top_k=n)
        else:
            fused = fused[:n]

        elapsed = (time.time() - start_time) * 1000

        return SearchResponse(
            query=query,
            results=fused,
            total_found=len(fused),
            search_time_ms=elapsed,
        )

    def _vector_search(
        self, query: str, k: int = 100, filters: Optional[dict] = None
    ) -> list[SearchResult]:
        """Vector similarity search using pgvector."""
        # Generate query embedding
        query_embedding = self.embedder.encode([query])[0].tolist()

        pool = get_pg_pool()

        # Build filter clause
        filter_clause = ""
        filter_params = []

        if filters:
            conditions = []
            if filters.get("year_min"):
                conditions.append("d.year >= %s")
                filter_params.append(filters["year_min"])
            if filters.get("year_max"):
                conditions.append("d.year <= %s")
                filter_params.append(filters["year_max"])
            if filters.get("doi"):
                conditions.append("d.doi = %s")
                filter_params.append(filters["doi"])

            if conditions:
                filter_clause = "AND " + " AND ".join(conditions)

        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        p.passage_id,
                        p.doc_id,
                        p.passage_text,
                        p.section,
                        d.title,
                        d.authors,
                        d.year,
                        d.doi,
                        1 - (p.embedding <=> %s::vector) as similarity
                    FROM passages p
                    JOIN documents d ON p.doc_id = d.doc_id
                    WHERE p.embedding IS NOT NULL
                      AND (p.is_superseded = FALSE OR p.is_superseded IS NULL)
                    {filter_clause}
                    ORDER BY p.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    [query_embedding, *filter_params, query_embedding, k],
                )

                results = []
                for i, row in enumerate(cur.fetchall()):
                    results.append(
                        SearchResult(
                            passage_id=row["passage_id"],
                            doc_id=row["doc_id"],
                            passage_text=row["passage_text"],
                            section=row["section"],
                            title=row["title"],
                            authors=row["authors"] or [],
                            year=row["year"],
                            doi=row["doi"],
                            score=row["similarity"],
                            vector_rank=i + 1,
                        )
                    )

        return results

    def _fts_search(
        self, query: str, k: int = 100, filters: Optional[dict] = None
    ) -> list[SearchResult]:
        """Full-text search using PostgreSQL tsvector."""
        pool = get_pg_pool()

        # Build filter clause
        filter_clause = ""
        filter_params = []

        if filters:
            conditions = []
            if filters.get("year_min"):
                conditions.append("d.year >= %s")
                filter_params.append(filters["year_min"])
            if filters.get("year_max"):
                conditions.append("d.year <= %s")
                filter_params.append(filters["year_max"])

            if conditions:
                filter_clause = "AND " + " AND ".join(conditions)

        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Convert query to tsquery format
                cur.execute(
                    f"""
                    SELECT
                        p.passage_id,
                        p.doc_id,
                        p.passage_text,
                        p.section,
                        d.title,
                        d.authors,
                        d.year,
                        d.doi,
                        ts_rank_cd(p.search_vector, websearch_to_tsquery('english', %s)) as rank
                    FROM passages p
                    JOIN documents d ON p.doc_id = d.doc_id
                    WHERE p.search_vector @@ websearch_to_tsquery('english', %s)
                      AND (p.is_superseded = FALSE OR p.is_superseded IS NULL)
                    {filter_clause}
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    [query, query, *filter_params, k],
                )

                results = []
                for i, row in enumerate(cur.fetchall()):
                    results.append(
                        SearchResult(
                            passage_id=row["passage_id"],
                            doc_id=row["doc_id"],
                            passage_text=row["passage_text"],
                            section=row["section"],
                            title=row["title"],
                            authors=row["authors"] or [],
                            year=row["year"],
                            doi=row["doi"],
                            score=row["rank"],
                            fts_rank=i + 1,
                        )
                    )

        return results

    def _graph_search(
        self, query: str, k: int = 100, filters: Optional[dict] = None
    ) -> list[SearchResult]:
        """Graph-based search using Neo4j concept matching."""
        if not self.neo4j_driver:
            return []

        try:
            # Extract key terms from query for concept matching
            from lib.db.neo4j import get_neo4j_driver

            driver = get_neo4j_driver()

            # Use Neo4j fulltext index to find matching concepts
            records, _, _ = driver.execute_query(
                """
                CALL db.index.fulltext.queryNodes("concept_names", $query)
                YIELD node, score
                WITH node, score
                LIMIT 10
                MATCH (node)<-[:MENTIONS]-(p:Passage)
                RETURN DISTINCT
                    p.passage_id as passage_id,
                    p.doc_id as doc_id,
                    p.text as passage_text,
                    p.section as section,
                    sum(score) as total_score
                ORDER BY total_score DESC
                LIMIT $k
                """,
                query=query,
                k=k,
            )

            # Get document metadata from Postgres
            passage_ids = [r["passage_id"] for r in records]
            if not passage_ids:
                return []

            pool = get_pg_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            p.passage_id,
                            p.doc_id,
                            p.passage_text,
                            p.section,
                            d.title,
                            d.authors,
                            d.year,
                            d.doi
                        FROM passages p
                        JOIN documents d ON p.doc_id = d.doc_id
                        WHERE p.passage_id = ANY(%s)
                        """,
                        (passage_ids,),
                    )

                    passage_data = {row["passage_id"]: row for row in cur.fetchall()}

            results = []
            for i, record in enumerate(records):
                pid = record["passage_id"]
                if pid in passage_data:
                    data = passage_data[pid]
                    results.append(
                        SearchResult(
                            passage_id=pid,
                            doc_id=data["doc_id"],
                            passage_text=data["passage_text"],
                            section=data["section"],
                            title=data["title"],
                            authors=data["authors"] or [],
                            year=data["year"],
                            doi=data["doi"],
                            score=record["total_score"],
                            graph_rank=i + 1,
                        )
                    )

            return results

        except Exception as e:
            logger.warning(f"Graph search failed: {e}")
            return []

    def _rrf_fusion(
        self,
        vector_results: list[SearchResult],
        fts_results: list[SearchResult],
        graph_results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Reciprocal Rank Fusion of multiple result lists.

        RRF score = sum(1 / (k + rank)) for each source where result appears.
        """
        # Collect all unique results by passage_id
        all_results: dict[str, SearchResult] = {}

        # Add vector results
        for result in vector_results:
            all_results[result.passage_id] = result

        # Merge FTS results
        for result in fts_results:
            if result.passage_id in all_results:
                all_results[result.passage_id].fts_rank = result.fts_rank
            else:
                all_results[result.passage_id] = result

        # Merge graph results
        for result in graph_results:
            if result.passage_id in all_results:
                all_results[result.passage_id].graph_rank = result.graph_rank
            else:
                all_results[result.passage_id] = result

        # Calculate RRF scores
        for result in all_results.values():
            rrf_score = 0.0

            if result.vector_rank:
                rrf_score += self.vector_weight * (1.0 / (RRF_K + result.vector_rank))

            if result.fts_rank:
                rrf_score += self.fts_weight * (1.0 / (RRF_K + result.fts_rank))

            if result.graph_rank:
                rrf_score += self.graph_weight * (1.0 / (RRF_K + result.graph_rank))

            result.score = rrf_score

        # Sort by RRF score
        fused = sorted(all_results.values(), key=lambda x: x.score, reverse=True)

        return fused

    def search_similar(
        self, passage_id: str, n: int = 10
    ) -> list[SearchResult]:
        """
        Find passages similar to a given passage.

        Args:
            passage_id: Source passage ID
            n: Number of results

        Returns:
            List of similar passages
        """
        pool = get_pg_pool()

        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Get source embedding
                cur.execute(
                    "SELECT embedding FROM passages WHERE passage_id = %s",
                    (passage_id,),
                )
                row = cur.fetchone()
                if not row or not row["embedding"]:
                    return []

                source_embedding = row["embedding"]

                # Find similar passages
                cur.execute(
                    """
                    SELECT
                        p.passage_id,
                        p.doc_id,
                        p.passage_text,
                        p.section,
                        d.title,
                        d.authors,
                        d.year,
                        d.doi,
                        1 - (p.embedding <=> %s::vector) as similarity
                    FROM passages p
                    JOIN documents d ON p.doc_id = d.doc_id
                    WHERE p.passage_id != %s
                      AND p.embedding IS NOT NULL
                      AND (p.is_superseded = FALSE OR p.is_superseded IS NULL)
                    ORDER BY p.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (source_embedding, passage_id, source_embedding, n),
                )

                results = []
                for row in cur.fetchall():
                    results.append(
                        SearchResult(
                            passage_id=row["passage_id"],
                            doc_id=row["doc_id"],
                            passage_text=row["passage_text"],
                            section=row["section"],
                            title=row["title"],
                            authors=row["authors"] or [],
                            year=row["year"],
                            doi=row["doi"],
                            score=row["similarity"],
                        )
                    )

        return results


def search(query: str, n: int = 20, **kwargs) -> SearchResponse:
    """
    Convenience function for hybrid search.

    Args:
        query: Search query
        n: Number of results
        **kwargs: Additional arguments for HybridSearcher.search()

    Returns:
        SearchResponse
    """
    searcher = HybridSearcher()
    return searcher.search(query, n=n, **kwargs)
