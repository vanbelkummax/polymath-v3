"""
Retrieval evaluation metrics for Polymath v3.

Implements standard IR metrics:
- Recall@K: Fraction of relevant docs retrieved in top K
- MRR (Mean Reciprocal Rank): 1/rank of first relevant doc
- nDCG (Normalized Discounted Cumulative Gain): Position-weighted relevance
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalResult:
    """Result of evaluating a single query."""

    query: str
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0

    # Details
    relevant_found: int = 0
    relevant_total: int = 0
    first_relevant_rank: Optional[int] = None
    retrieved_ids: list[str] = field(default_factory=list)

    @property
    def recall_at_k(self) -> dict[int, float]:
        return {5: self.recall_at_5, 10: self.recall_at_10, 20: self.recall_at_20}


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Compute Recall@K.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        k: Cutoff rank

    Returns:
        Fraction of relevant documents retrieved in top K
    """
    if not relevant_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])
    found = len(top_k & relevant_ids)
    return found / len(relevant_ids)


def mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs

    Returns:
        1 / rank of first relevant document (0 if none found)
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(relevances: list[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at K.

    Uses the standard formula: sum(rel_i / log2(i + 1))
    """
    relevances = relevances[:k]
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    relevance_scores: Optional[dict[str, float]] = None,
    k: int = 10,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        relevance_scores: Optional dict mapping doc_id -> relevance (0-1)
                         If None, binary relevance (1 if relevant, 0 otherwise)
        k: Cutoff rank

    Returns:
        nDCG score (0-1)
    """
    if not relevant_ids:
        return 0.0

    # Build relevance list for retrieved docs
    if relevance_scores:
        relevances = [
            relevance_scores.get(doc_id, 0.0) for doc_id in retrieved_ids[:k]
        ]
    else:
        # Binary relevance
        relevances = [
            1.0 if doc_id in relevant_ids else 0.0 for doc_id in retrieved_ids[:k]
        ]

    # Compute DCG
    actual_dcg = dcg_at_k(relevances, k)

    # Compute ideal DCG (perfect ranking)
    if relevance_scores:
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)
    else:
        ideal_relevances = [1.0] * len(relevant_ids)

    ideal_dcg = dcg_at_k(ideal_relevances, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_retrieval(
    query: str,
    retrieved_ids: list[str],
    relevant_ids: set[str],
    relevance_scores: Optional[dict[str, float]] = None,
) -> EvalResult:
    """
    Evaluate retrieval for a single query.

    Args:
        query: The search query
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        relevance_scores: Optional graded relevance scores

    Returns:
        EvalResult with all metrics
    """
    # Find first relevant rank
    first_rank = None
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            first_rank = i + 1
            break

    # Count relevant in retrieved
    relevant_found = len(set(retrieved_ids) & relevant_ids)

    return EvalResult(
        query=query,
        recall_at_5=recall_at_k(retrieved_ids, relevant_ids, 5),
        recall_at_10=recall_at_k(retrieved_ids, relevant_ids, 10),
        recall_at_20=recall_at_k(retrieved_ids, relevant_ids, 20),
        mrr=mrr(retrieved_ids, relevant_ids),
        ndcg_at_10=ndcg(retrieved_ids, relevant_ids, relevance_scores, k=10),
        relevant_found=relevant_found,
        relevant_total=len(relevant_ids),
        first_relevant_rank=first_rank,
        retrieved_ids=retrieved_ids[:20],
    )


def aggregate_metrics(results: list[EvalResult]) -> dict:
    """
    Aggregate metrics across multiple queries.

    Returns:
        Dict with mean metrics and per-query breakdown
    """
    if not results:
        return {"error": "No results to aggregate"}

    n = len(results)
    return {
        "n_queries": n,
        "mean_recall_at_5": sum(r.recall_at_5 for r in results) / n,
        "mean_recall_at_10": sum(r.recall_at_10 for r in results) / n,
        "mean_recall_at_20": sum(r.recall_at_20 for r in results) / n,
        "mean_mrr": sum(r.mrr for r in results) / n,
        "mean_ndcg_at_10": sum(r.ndcg_at_10 for r in results) / n,
        "queries_with_relevant": sum(1 for r in results if r.relevant_found > 0),
        "total_relevant_found": sum(r.relevant_found for r in results),
        "total_relevant_expected": sum(r.relevant_total for r in results),
    }
