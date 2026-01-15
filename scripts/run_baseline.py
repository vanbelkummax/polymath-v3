#!/usr/bin/env python3
"""
Run baseline evaluation and store results.

Establishes a baseline for retrieval metrics that regression tests can check against.

Usage:
    # Run baseline and save
    python scripts/run_baseline.py

    # Run with reranking
    python scripts/run_baseline.py --rerank

    # Compare against existing baseline
    python scripts/run_baseline.py --compare
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_evaluation import run_evaluation
from lib.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths
EVAL_SET_PATH = Path(config.PROJECT_ROOT) / "data" / "eval_sets" / "core.jsonl"
BASELINE_PATH = Path(config.PROJECT_ROOT) / "data" / "eval_sets" / "baseline.json"


def run_baseline(
    eval_set_path: Path = EVAL_SET_PATH,
    baseline_path: Path = BASELINE_PATH,
    rerank: bool = False,
    n_results: int = 20,
) -> dict:
    """
    Run evaluation and store as baseline.

    Args:
        eval_set_path: Path to eval set
        baseline_path: Where to save baseline
        rerank: Whether to use reranking
        n_results: Number of results to retrieve

    Returns:
        Baseline metrics dict
    """
    logger.info(f"Running baseline evaluation...")
    logger.info(f"  Eval set: {eval_set_path}")
    logger.info(f"  Rerank: {rerank}")

    # Run evaluation
    metrics = run_evaluation(
        eval_set_path=eval_set_path,
        rerank=rerank,
        n_results=n_results,
    )

    if "error" in metrics:
        logger.error(f"Evaluation failed: {metrics['error']}")
        return metrics

    # Create baseline record
    baseline = {
        "timestamp": datetime.utcnow().isoformat(),
        "eval_set": str(eval_set_path),
        "config": {
            "rerank": rerank,
            "n_results": n_results,
            "vector_weight": 0.5,
            "fts_weight": 0.3,
            "graph_weight": 0.2,
        },
        "metrics": {
            "recall_at_5": metrics.get("mean_recall_at_5", 0),
            "recall_at_10": metrics.get("mean_recall_at_10", 0),
            "recall_at_20": metrics.get("mean_recall_at_20", 0),
            "mrr": metrics.get("mean_mrr", 0),
            "ndcg_at_10": metrics.get("mean_ndcg_at_10", 0),
            "n_queries": metrics.get("n_queries", 0),
            "queries_with_relevant": metrics.get("queries_with_relevant", 0),
        },
    }

    # Save baseline
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)

    logger.info(f"Baseline saved to {baseline_path}")
    return baseline


def load_baseline(baseline_path: Path = BASELINE_PATH) -> dict | None:
    """Load existing baseline."""
    if not baseline_path.exists():
        return None
    with open(baseline_path) as f:
        return json.load(f)


def compare_to_baseline(
    current_metrics: dict,
    baseline: dict,
    regression_margin: float = 0.05,
) -> dict:
    """
    Compare current metrics to baseline.

    Args:
        current_metrics: Current evaluation metrics
        baseline: Baseline metrics
        regression_margin: Allowed regression margin (5% default)

    Returns:
        Comparison results with pass/fail status
    """
    baseline_metrics = baseline["metrics"]

    comparisons = {}
    all_passed = True

    for metric in ["recall_at_10", "mrr", "ndcg_at_10"]:
        baseline_val = baseline_metrics.get(metric, 0)
        current_val = current_metrics.get(f"mean_{metric}", 0)

        # Allow regression_margin below baseline
        threshold = baseline_val * (1 - regression_margin)
        passed = current_val >= threshold

        comparisons[metric] = {
            "baseline": baseline_val,
            "current": current_val,
            "threshold": threshold,
            "passed": passed,
            "delta": current_val - baseline_val,
            "delta_pct": ((current_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0,
        }

        if not passed:
            all_passed = False

    return {
        "passed": all_passed,
        "comparisons": comparisons,
        "baseline_timestamp": baseline.get("timestamp"),
    }


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=EVAL_SET_PATH,
        help="Path to eval set",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_PATH,
        help="Path to save/load baseline",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Use neural reranking",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare against existing baseline instead of creating new one",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=20,
        help="Number of results to retrieve",
    )

    args = parser.parse_args()

    if args.compare:
        # Load baseline
        baseline = load_baseline(args.baseline)
        if not baseline:
            logger.error(f"No baseline found at {args.baseline}")
            logger.info("Run without --compare first to establish baseline")
            sys.exit(1)

        # Run current evaluation
        logger.info("Running evaluation to compare against baseline...")
        current = run_evaluation(
            eval_set_path=args.eval_set,
            rerank=args.rerank,
            n_results=args.n_results,
        )

        if "error" in current:
            logger.error(f"Evaluation failed: {current['error']}")
            sys.exit(1)

        # Compare
        comparison = compare_to_baseline(current, baseline)

        print("\n" + "=" * 60)
        print("BASELINE COMPARISON")
        print("=" * 60)
        print(f"Baseline from: {comparison['baseline_timestamp']}")
        print("-" * 60)

        for metric, data in comparison["comparisons"].items():
            status = "PASS" if data["passed"] else "FAIL"
            print(f"{metric}:")
            print(f"  Baseline: {data['baseline']:.4f}")
            print(f"  Current:  {data['current']:.4f}")
            print(f"  Delta:    {data['delta']:+.4f} ({data['delta_pct']:+.1f}%)")
            print(f"  Status:   [{status}]")
            print()

        print("=" * 60)
        overall = "PASSED" if comparison["passed"] else "FAILED"
        print(f"Overall: {overall}")
        print("=" * 60)

        sys.exit(0 if comparison["passed"] else 1)

    else:
        # Create new baseline
        run_baseline(
            eval_set_path=args.eval_set,
            baseline_path=args.baseline,
            rerank=args.rerank,
            n_results=args.n_results,
        )


if __name__ == "__main__":
    main()
