"""
Gap detection for BridgeMine.

Identifies under-explored method-problem combinations by analyzing
the concept graph for methods that solve similar problems but haven't
been applied to certain domains.

Key insight: If Method A solves Problem X, and Problem X is similar
to Problem Y in Domain D, but Method A hasn't been applied in Domain D,
this is a potential research opportunity.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from lib.config import config
from lib.db.neo4j import get_neo4j_driver

logger = logging.getLogger(__name__)

# Generic methods to filter out (too broad to be actionable)
GENERIC_METHODS = {
    "deep_learning",
    "machine_learning",
    "neural_network",
    "artificial_intelligence",
    "statistical_analysis",
    "data_analysis",
    "computational_method",
    "algorithm",
    "model",
    "framework",
}


@dataclass
class GapCandidate:
    """A potential research gap identified by BridgeMine."""

    method_name: str
    method_mentions: int
    source_problem: str
    target_problem: str
    target_domain: str

    # Graph metrics
    problem_similarity: float
    method_problem_cooccurrences: int
    domain_penetration: float  # % of domain papers that use this method

    # Novelty metrics (filled by NoveltyChecker)
    novelty_score: float = 0.0
    prior_art_count: int = 0
    rrf_confidence: float = 0.0

    @property
    def is_promising(self) -> bool:
        """Check if this gap is worth investigating."""
        return (
            self.novelty_score > 0.5
            and self.prior_art_count < 10
            and self.domain_penetration < 0.1
        )


@dataclass
class GapDetectionResult:
    """Result of gap detection run."""

    target_domain: str
    candidates: list[GapCandidate] = field(default_factory=list)
    methods_analyzed: int = 0
    problems_analyzed: int = 0
    elapsed_seconds: float = 0.0


class GapDetector:
    """
    Detect research gaps using Neo4j concept graph.

    The core algorithm:
    1. Find methods with many SOLVES relationships
    2. For each method's solved problems, find similar problems
    3. Check if method has been applied to similar problems in target domain
    4. If not, this is a potential gap

    Usage:
        detector = GapDetector()
        result = detector.find_gaps(target_domain="spatial_transcriptomics")
        for gap in result.candidates:
            print(f"{gap.method_name} -> {gap.target_problem}")
    """

    def __init__(
        self,
        min_method_mentions: int = 10,
        min_problem_similarity: float = 0.7,
        max_domain_penetration: float = 0.1,
    ):
        """
        Initialize gap detector.

        Args:
            min_method_mentions: Minimum mentions for a method to consider
            min_problem_similarity: Minimum similarity between problems
            max_domain_penetration: Maximum domain penetration (higher = already explored)
        """
        self.min_method_mentions = min_method_mentions
        self.min_problem_similarity = min_problem_similarity
        self.max_domain_penetration = max_domain_penetration
        self._driver = None

    @property
    def driver(self):
        """Lazy load Neo4j driver."""
        if self._driver is None:
            self._driver = get_neo4j_driver()
        return self._driver

    def find_gaps(
        self,
        target_domain: str,
        limit: int = 100,
        exclude_generic: bool = True,
    ) -> GapDetectionResult:
        """
        Find research gaps for a target domain.

        Args:
            target_domain: Domain to find gaps for (e.g., "spatial_transcriptomics")
            limit: Maximum number of gaps to return
            exclude_generic: Whether to exclude generic methods

        Returns:
            GapDetectionResult with candidate gaps
        """
        import time

        start_time = time.time()

        # Step 1: Get problems in target domain
        domain_problems = self._get_domain_problems(target_domain)
        logger.info(f"Found {len(domain_problems)} problems in {target_domain}")

        if not domain_problems:
            return GapDetectionResult(
                target_domain=target_domain,
                elapsed_seconds=time.time() - start_time,
            )

        # Step 2: Find methods that solve similar problems elsewhere
        candidates = []
        methods_analyzed = 0
        problems_analyzed = len(domain_problems)

        for target_problem in domain_problems[:50]:  # Limit for performance
            # Find similar problems and their methods
            similar_methods = self._find_transfer_candidates(
                target_problem,
                target_domain,
                exclude_generic,
            )

            for method, source_problem, similarity, mentions in similar_methods:
                if exclude_generic and method.lower() in GENERIC_METHODS:
                    continue

                # Calculate domain penetration
                penetration = self._calculate_domain_penetration(
                    method, target_domain
                )

                if penetration > self.max_domain_penetration:
                    continue  # Already well-explored

                candidates.append(
                    GapCandidate(
                        method_name=method,
                        method_mentions=mentions,
                        source_problem=source_problem,
                        target_problem=target_problem,
                        target_domain=target_domain,
                        problem_similarity=similarity,
                        method_problem_cooccurrences=mentions,
                        domain_penetration=penetration,
                    )
                )

                methods_analyzed += 1

        # Sort by potential (high similarity, low penetration)
        candidates.sort(
            key=lambda x: x.problem_similarity * (1 - x.domain_penetration),
            reverse=True,
        )

        elapsed = time.time() - start_time
        logger.info(
            f"Found {len(candidates)} gap candidates in {elapsed:.1f}s"
        )

        return GapDetectionResult(
            target_domain=target_domain,
            candidates=candidates[:limit],
            methods_analyzed=methods_analyzed,
            problems_analyzed=problems_analyzed,
            elapsed_seconds=elapsed,
        )

    def _get_domain_problems(self, domain: str) -> list[str]:
        """Get all problems associated with a domain."""
        try:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (d:DOMAIN {name: $domain})<-[:IN_DOMAIN]-(p:Passage)-[:MENTIONS]->(prob:PROBLEM)
                RETURN DISTINCT prob.name as problem, count(*) as mentions
                ORDER BY mentions DESC
                LIMIT 100
                """,
                domain=domain,
            )
            return [r["problem"] for r in records]

        except Exception as e:
            logger.error(f"Failed to get domain problems: {e}")
            return []

    def _find_transfer_candidates(
        self,
        target_problem: str,
        target_domain: str,
        exclude_generic: bool,
    ) -> list[tuple[str, str, float, int]]:
        """
        Find methods that could transfer to target problem.

        Returns list of (method, source_problem, similarity, mentions).
        """
        try:
            # Find similar problems and their methods
            records, _, _ = self.driver.execute_query(
                """
                // Find the target problem
                MATCH (target:PROBLEM {name: $target_problem})

                // Find similar problems via embedding or co-occurrence
                MATCH (target)-[sim:SIMILAR_TO]-(source:PROBLEM)
                WHERE sim.score >= $min_similarity

                // Get methods that solve the source problem
                MATCH (source)<-[:SOLVES]-(method:METHOD)

                // Count method usage
                MATCH (method)<-[:MENTIONS]-(p:Passage)

                // Exclude methods already heavily used in target domain
                OPTIONAL MATCH (method)<-[:MENTIONS]-(dp:Passage)-[:IN_DOMAIN]->(:DOMAIN {name: $target_domain})

                WITH method, source, sim.score as similarity,
                     count(DISTINCT p) as total_mentions,
                     count(DISTINCT dp) as domain_mentions

                WHERE total_mentions >= $min_mentions
                  AND domain_mentions < total_mentions * 0.1

                RETURN method.name as method,
                       source.name as source_problem,
                       similarity,
                       total_mentions as mentions
                ORDER BY mentions DESC, similarity DESC
                LIMIT 20
                """,
                target_problem=target_problem,
                target_domain=target_domain,
                min_similarity=self.min_problem_similarity,
                min_mentions=self.min_method_mentions,
            )

            return [
                (r["method"], r["source_problem"], r["similarity"], r["mentions"])
                for r in records
            ]

        except Exception as e:
            logger.error(f"Failed to find transfer candidates: {e}")
            return []

    def _calculate_domain_penetration(self, method: str, domain: str) -> float:
        """
        Calculate what fraction of domain papers mention this method.

        Returns value between 0 (never used) and 1 (used everywhere).
        """
        try:
            records, _, _ = self.driver.execute_query(
                """
                // Count total papers in domain
                MATCH (d:DOMAIN {name: $domain})<-[:IN_DOMAIN]-(p:Passage)
                WITH count(DISTINCT p.doc_id) as total_papers

                // Count papers using this method
                MATCH (m:METHOD {name: $method})<-[:MENTIONS]-(p:Passage)-[:IN_DOMAIN]->(:DOMAIN {name: $domain})
                WITH total_papers, count(DISTINCT p.doc_id) as method_papers

                RETURN
                    CASE WHEN total_papers > 0
                         THEN toFloat(method_papers) / total_papers
                         ELSE 0.0
                    END as penetration
                """,
                method=method,
                domain=domain,
            )

            if records:
                return records[0]["penetration"]
            return 0.0

        except Exception as e:
            logger.debug(f"Failed to calculate penetration: {e}")
            return 0.0

    def find_analogies(
        self,
        source_method: str,
        target_domain: str,
        n_results: int = 10,
    ) -> list[GapCandidate]:
        """
        Find problems in target domain that source method could address.

        This is the inverse of find_gaps: given a method, find where it
        might be useful.

        Args:
            source_method: Method to find applications for
            target_domain: Domain to explore
            n_results: Number of results

        Returns:
            List of GapCandidate objects
        """
        try:
            # Get problems the method currently solves
            records, _, _ = self.driver.execute_query(
                """
                // Get problems this method solves
                MATCH (m:METHOD {name: $method})-[:SOLVES]->(source:PROBLEM)

                // Find similar problems in target domain
                MATCH (source)-[sim:SIMILAR_TO]-(target:PROBLEM)
                WHERE sim.score >= $min_similarity

                // Check target is in domain
                MATCH (target)<-[:MENTIONS]-(p:Passage)-[:IN_DOMAIN]->(:DOMAIN {name: $domain})

                // Exclude if method already applied
                WHERE NOT EXISTS {
                    MATCH (m)<-[:MENTIONS]-(p2:Passage)-[:MENTIONS]->(target)
                }

                RETURN DISTINCT
                    source.name as source_problem,
                    target.name as target_problem,
                    sim.score as similarity,
                    count(p) as target_mentions
                ORDER BY similarity DESC, target_mentions DESC
                LIMIT $limit
                """,
                method=source_method,
                domain=target_domain,
                min_similarity=self.min_problem_similarity,
                limit=n_results,
            )

            candidates = []
            for r in records:
                penetration = self._calculate_domain_penetration(
                    source_method, target_domain
                )

                candidates.append(
                    GapCandidate(
                        method_name=source_method,
                        method_mentions=r["target_mentions"],
                        source_problem=r["source_problem"],
                        target_problem=r["target_problem"],
                        target_domain=target_domain,
                        problem_similarity=r["similarity"],
                        method_problem_cooccurrences=r["target_mentions"],
                        domain_penetration=penetration,
                    )
                )

            return candidates

        except Exception as e:
            logger.error(f"Failed to find analogies: {e}")
            return []


def find_gaps(target_domain: str, **kwargs) -> GapDetectionResult:
    """
    Convenience function for gap detection.

    Args:
        target_domain: Domain to find gaps for
        **kwargs: Arguments for GapDetector

    Returns:
        GapDetectionResult
    """
    detector = GapDetector()
    return detector.find_gaps(target_domain, **kwargs)
