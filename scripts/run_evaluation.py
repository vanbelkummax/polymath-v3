#!/usr/bin/env python3
"""
Run retrieval evaluation against an eval set.

Usage:
    python scripts/run_evaluation.py --eval-set data/eval_sets/core.jsonl
    python scripts/run_evaluation.py --eval-set data/eval_sets/core.jsonl --rerank
    python scripts/run_evaluation.py --eval-set data/eval_sets/core.jsonl --output results.json
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.evaluation import (
    load_eval_set,
    evaluate_retrieval,
    EvalResult,
)
from lib.evaluation.metrics import aggregate_metrics
from lib.search.hybrid_search import HybridSearcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_evaluation(
    eval_set_path: Path,
    rerank: bool = False,
    n_results: int = 20,
    output_path: Path = None,
) -> dict:
    """
    Run evaluation against an eval set.

    Args:
        eval_set_path: Path to JSONL eval set
        rerank: Whether to use neural reranking
        n_results: Number of results to retrieve
        output_path: Optional path to save results

    Returns:
        Dict with aggregated metrics
    """
    # Load eval set
    queries = load_eval_set(eval_set_path)
    if not queries:
        logger.error(f"No queries found in {eval_set_path}")
        return {"error": "No queries in eval set"}

    logger.info(f"Loaded {len(queries)} eval queries from {eval_set_path}")

    # Initialize searcher
    searcher = HybridSearcher()
    logger.info(f"Searcher initialized (rerank={rerank})")

    # Run evaluation
    results: list[EvalResult] = []
    start_time = time.time()

    for i, eq in enumerate(queries):
        logger.info(f"[{i+1}/{len(queries)}] Query: {eq.query[:50]}...")

        try:
            # Run search
            response = searcher.search(
                eq.query,
                n=n_results,
                rerank=rerank,
            )

            # Extract passage IDs
            retrieved_ids = [r.passage_id for r in response.results]

            # Evaluate
            result = evaluate_retrieval(
                query=eq.query,
                retrieved_ids=retrieved_ids,
                relevant_ids=eq.relevant_set,
                relevance_scores=eq.relevance_scores or None,
            )
            results.append(result)

            # Log per-query metrics
            logger.info(
                f"  R@10={result.recall_at_10:.3f} "
                f"MRR={result.mrr:.3f} "
                f"nDCG@10={result.ndcg_at_10:.3f}"
            )

        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append(EvalResult(
                query=eq.query,
                relevant_total=len(eq.relevant_ids),
            ))

    elapsed = time.time() - start_time
    logger.info(f"\nEvaluation completed in {elapsed:.1f}s")

    # Aggregate metrics
    aggregated = aggregate_metrics(results)
    aggregated["eval_set"] = str(eval_set_path)
    aggregated["rerank"] = rerank
    aggregated["n_results"] = n_results
    aggregated["timestamp"] = datetime.utcnow().isoformat()
    aggregated["elapsed_seconds"] = elapsed

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Eval set: {eval_set_path}")
    print(f"Queries: {aggregated['n_queries']}")
    print(f"Rerank: {rerank}")
    print("-" * 60)
    print(f"Mean Recall@5:   {aggregated['mean_recall_at_5']:.4f}")
    print(f"Mean Recall@10:  {aggregated['mean_recall_at_10']:.4f}")
    print(f"Mean Recall@20:  {aggregated['mean_recall_at_20']:.4f}")
    print(f"Mean MRR:        {aggregated['mean_mrr']:.4f}")
    print(f"Mean nDCG@10:    {aggregated['mean_ndcg_at_10']:.4f}")
    print("-" * 60)
    print(f"Queries with relevant found: {aggregated['queries_with_relevant']}/{aggregated['n_queries']}")
    print(f"Total relevant found: {aggregated['total_relevant_found']}/{aggregated['total_relevant_expected']}")
    print("=" * 60)

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save aggregated + per-query
        full_results = {
            "aggregated": aggregated,
            "per_query": [
                {
                    "query": r.query,
                    "recall_at_5": r.recall_at_5,
                    "recall_at_10": r.recall_at_10,
                    "recall_at_20": r.recall_at_20,
                    "mrr": r.mrr,
                    "ndcg_at_10": r.ndcg_at_10,
                    "relevant_found": r.relevant_found,
                    "relevant_total": r.relevant_total,
                    "first_relevant_rank": r.first_relevant_rank,
                }
                for r in results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(full_results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument(
        "--eval-set",
        type=Path,
        required=True,
        help="Path to JSONL eval set",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Use neural reranking",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=20,
        help="Number of results to retrieve",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save results JSON",
    )

    args = parser.parse_args()

    run_evaluation(
        eval_set_path=args.eval_set,
        rerank=args.rerank,
        n_results=args.n_results,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
