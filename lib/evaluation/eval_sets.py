"""
Evaluation set management for Polymath v3.

JSONL format for eval sets:
{
    "query": "spatial transcriptomics deconvolution",
    "relevant_ids": ["uuid1", "uuid2"],
    "relevance_scores": {"uuid1": 1.0, "uuid2": 0.8},
    "domain": "spatial_transcriptomics",
    "difficulty": "medium",
    "notes": "Key papers on cell type deconvolution"
}
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalQuery:
    """A single evaluation query with ground truth."""

    query: str
    relevant_ids: list[str] = field(default_factory=list)
    relevance_scores: dict[str, float] = field(default_factory=dict)

    # Metadata
    domain: Optional[str] = None
    difficulty: Optional[str] = None  # easy, medium, hard
    notes: Optional[str] = None
    created_at: Optional[str] = None
    created_by: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()

    @property
    def relevant_set(self) -> set[str]:
        """Get relevant IDs as a set for efficient lookup."""
        return set(self.relevant_ids)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "EvalQuery":
        """Create from dict."""
        return cls(
            query=data["query"],
            relevant_ids=data.get("relevant_ids", []),
            relevance_scores=data.get("relevance_scores", {}),
            domain=data.get("domain"),
            difficulty=data.get("difficulty"),
            notes=data.get("notes"),
            created_at=data.get("created_at"),
            created_by=data.get("created_by"),
        )


def load_eval_set(path: Path) -> list[EvalQuery]:
    """
    Load evaluation set from JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of EvalQuery objects
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Eval set not found: {path}")
        return []

    queries = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                data = json.loads(line)
                queries.append(EvalQuery.from_dict(data))
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON at line {line_num}: {e}")
            except KeyError as e:
                logger.warning(f"Missing required field at line {line_num}: {e}")

    logger.info(f"Loaded {len(queries)} eval queries from {path}")
    return queries


def save_eval_set(queries: list[EvalQuery], path: Path) -> None:
    """
    Save evaluation set to JSONL file.

    Args:
        queries: List of EvalQuery objects
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for query in queries:
            f.write(json.dumps(query.to_dict()) + "\n")

    logger.info(f"Saved {len(queries)} eval queries to {path}")


def merge_eval_sets(*paths: Path) -> list[EvalQuery]:
    """
    Merge multiple eval sets, deduplicating by query.

    Args:
        *paths: Paths to JSONL files

    Returns:
        Merged list of EvalQuery objects
    """
    seen_queries = {}

    for path in paths:
        for eq in load_eval_set(path):
            # Later entries override earlier ones
            seen_queries[eq.query] = eq

    return list(seen_queries.values())


def filter_by_domain(
    queries: list[EvalQuery],
    domain: str,
) -> list[EvalQuery]:
    """Filter eval queries by domain."""
    return [q for q in queries if q.domain == domain]


def filter_by_difficulty(
    queries: list[EvalQuery],
    difficulty: str,
) -> list[EvalQuery]:
    """Filter eval queries by difficulty."""
    return [q for q in queries if q.difficulty == difficulty]


def create_eval_query_interactive() -> Optional[EvalQuery]:
    """
    Interactive CLI for creating a single eval query.

    Returns:
        EvalQuery or None if cancelled
    """
    print("\n=== Create Evaluation Query ===\n")

    query = input("Query: ").strip()
    if not query:
        print("Cancelled.")
        return None

    print("\nEnter relevant passage IDs (one per line, empty line to finish):")
    relevant_ids = []
    while True:
        pid = input("  ID: ").strip()
        if not pid:
            break
        relevant_ids.append(pid)

    if not relevant_ids:
        print("No relevant IDs provided. Cancelled.")
        return None

    domain = input("\nDomain (optional): ").strip() or None
    difficulty = input("Difficulty (easy/medium/hard, optional): ").strip() or None
    notes = input("Notes (optional): ").strip() or None

    return EvalQuery(
        query=query,
        relevant_ids=relevant_ids,
        domain=domain,
        difficulty=difficulty,
        notes=notes,
    )
