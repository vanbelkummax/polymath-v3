"""
Evaluation framework for Polymath v3.

Provides:
- JSONL eval set management
- Retrieval metrics (Recall@K, MRR, nDCG)
- Interactive labeling CLI
- Regression testing
"""

from lib.evaluation.metrics import (
    recall_at_k,
    mrr,
    ndcg,
    evaluate_retrieval,
    EvalResult,
)
from lib.evaluation.eval_sets import (
    EvalQuery,
    load_eval_set,
    save_eval_set,
)

__all__ = [
    "recall_at_k",
    "mrr",
    "ndcg",
    "evaluate_retrieval",
    "EvalResult",
    "EvalQuery",
    "load_eval_set",
    "save_eval_set",
]
