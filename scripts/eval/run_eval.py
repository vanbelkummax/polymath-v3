#!/usr/bin/env python3
"""
Evaluation runner for Polymath v3 retrieval.

Computes standard IR metrics:
- Recall@K: Fraction of relevant documents retrieved in top K
- MRR: Mean Reciprocal Rank (position of first relevant doc)
- nDCG@K: Normalized Discounted Cumulative Gain

Usage:
    python scripts/eval/run_eval.py --evalset data/evalsets/core_eval.jsonl
    python scripts/eval/run_eval.py --evalset data/evalsets/core_eval.jsonl --k 5 10 20
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.search.hybrid_search import HybridSearcher

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvalQuery:
    """A single evaluation query with ground truth."""

    query: str
    relevant_passage_ids: list[str]
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class EvalResult:
    """Results for a single query."""

    query: str
    retrieved_ids: list[str]
    relevant_ids: list[str]

    # Metrics
    recall_at_k: dict[int, float] = field(default_factory=dict)
    reciprocal_rank: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)

    # Debugging info
    first_relevant_rank: Optional[int] = None
    relevant_retrieved: int = 0


def load_evalset(path: Path) -> list[EvalQuery]:
    """Load evaluation set from JSONL file."""
    queries = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            queries.append(EvalQuery(
                query=data["query"],
                relevant_passage_ids=data["relevant_passage_ids"],
                tags=data.get("tags", []),
                notes=data.get("notes", ""),
            ))
    return queries


def compute_recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Recall@K."""
    if not relevant:
        return 0.0
    retrieved_at_k = set(retrieved[:k])
    return len(retrieved_at_k & relevant) / len(relevant)


def compute_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Compute Reciprocal Rank (1/rank of first relevant doc)."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute nDCG@K."""
    import math

    def dcg(relevances: list[int], k: int) -> float:
        return sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(relevances[:k])
        )

    # Compute DCG
    relevances = [1 if doc_id in relevant else 0 for doc_id in retrieved[:k]]
    actual_dcg = dcg(relevances, k)

    # Compute ideal DCG (all relevant first)
    ideal_relevances = [1] * min(len(relevant), k) + [0] * max(0, k - len(relevant))
    ideal_dcg = dcg(ideal_relevances, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def evaluate_query(
    query: EvalQuery,
    searcher: HybridSearcher,
    k_values: list[int],
    n_retrieve: int = 50,
    rerank: bool = True,
) -> EvalResult:
    """Evaluate a single query."""
    # Run search
    response = searcher.search(query.query, n=n_retrieve, rerank=rerank)

    # Extract retrieved IDs
    retrieved_ids = [str(r.passage_id) for r in response.results]
    relevant_set = set(query.relevant_passage_ids)

    # Compute metrics
    result = EvalResult(
        query=query.query,
        retrieved_ids=retrieved_ids,
        relevant_ids=query.relevant_passage_ids,
    )

    # Recall@K
    for k in k_values:
        result.recall_at_k[k] = compute_recall_at_k(retrieved_ids, relevant_set, k)

    # MRR
    result.reciprocal_rank = compute_reciprocal_rank(retrieved_ids, relevant_set)

    # nDCG@K
    for k in k_values:
        result.ndcg_at_k[k] = compute_ndcg_at_k(retrieved_ids, relevant_set, k)

    # Debug info
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            if result.first_relevant_rank is None:
                result.first_relevant_rank = i + 1
            result.relevant_retrieved += 1

    return result


def run_evaluation(
    evalset_path: Path,
    k_values: list[int] = [5, 10, 20],
    rerank: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Run full evaluation on an evalset.

    Returns dict with aggregate metrics.
    """
    # Load evalset
    queries = load_evalset(evalset_path)
    logger.info(f"Loaded {len(queries)} queries from {evalset_path}")

    # Initialize searcher
    searcher = HybridSearcher()

    # Run evaluation
    results = []
    for i, query in enumerate(queries):
        if verbose:
            logger.info(f"[{i+1}/{len(queries)}] {query.query[:50]}...")

        result = evaluate_query(query, searcher, k_values, rerank=rerank)
        results.append(result)

    # Aggregate metrics
    aggregate = {
        "n_queries": len(queries),
        "rerank_applied": rerank,
        "k_values": k_values,
    }

    # Average Recall@K
    for k in k_values:
        recalls = [r.recall_at_k.get(k, 0) for r in results]
        aggregate[f"recall@{k}"] = sum(recalls) / len(recalls) if recalls else 0

    # MRR
    mrrs = [r.reciprocal_rank for r in results]
    aggregate["mrr"] = sum(mrrs) / len(mrrs) if mrrs else 0

    # Average nDCG@K
    for k in k_values:
        ndcgs = [r.ndcg_at_k.get(k, 0) for r in results]
        aggregate[f"ndcg@{k}"] = sum(ndcgs) / len(ndcgs) if ndcgs else 0

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Evalset: {evalset_path}")
    logger.info(f"Queries: {len(queries)}")
    logger.info(f"Rerank: {rerank}")
    logger.info("-" * 60)

    for k in k_values:
        logger.info(f"Recall@{k}: {aggregate[f'recall@{k}']:.3f}")

    logger.info(f"MRR: {aggregate['mrr']:.3f}")

    for k in k_values:
        logger.info(f"nDCG@{k}: {aggregate[f'ndcg@{k}']:.3f}")

    logger.info("=" * 60)

    # Show failed queries (no relevant docs retrieved)
    failed = [r for r in results if r.first_relevant_rank is None]
    if failed:
        logger.info(f"\nFailed queries ({len(failed)}):")
        for r in failed[:5]:
            logger.info(f"  - {r.query[:60]}...")

    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument(
        "--evalset",
        type=Path,
        required=True,
        help="Path to evalset JSONL file",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="K values for Recall@K and nDCG@K",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable neural reranking",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    if not args.evalset.exists():
        logger.error(f"Evalset not found: {args.evalset}")
        sys.exit(1)

    results = run_evaluation(
        args.evalset,
        k_values=args.k,
        rerank=not args.no_rerank,
        verbose=args.verbose,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
