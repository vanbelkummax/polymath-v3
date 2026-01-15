"""
Retrieval telemetry for Polymath v3.

Logs per-query telemetry in JSONL format for debugging, evaluation,
and regression testing.

Enable with: POLYMATH_TELEMETRY=1
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from lib.config import config

logger = logging.getLogger(__name__)


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled via environment variable."""
    return os.environ.get("POLYMATH_TELEMETRY", "0") == "1"


# Default log path
TELEMETRY_LOG_PATH = Path(config.PROJECT_ROOT) / "logs" / "retrieval_runs.jsonl"


@dataclass
class RetrievalTelemetry:
    """Telemetry data for a single retrieval operation."""

    # Identifiers
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Query info
    query: str = ""
    query_hash: str = ""  # For grouping similar queries

    # Result counts
    n_requested: int = 0
    n_returned: int = 0

    # Per-source results
    vector_results: list[dict] = field(default_factory=list)
    fts_results: list[dict] = field(default_factory=list)
    graph_results: list[dict] = field(default_factory=list)

    # Fusion info
    fused_ranks: list[dict] = field(default_factory=list)  # passage_id -> final_rank
    reranker_scores: list[dict] = field(default_factory=list)

    # Timing
    vector_latency_ms: float = 0.0
    fts_latency_ms: float = 0.0
    graph_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Flags
    rerank_applied: bool = False
    filters_applied: Optional[dict] = None

    # Errors
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)


class TelemetryLogger:
    """
    Logger for retrieval telemetry.

    Usage:
        with TelemetryLogger() as tl:
            tl.set_query("spatial transcriptomics")
            with tl.time("vector"):
                vector_results = search_vector(...)
            tl.add_vector_results(vector_results)
            # ... more operations
            tl.finalize()
    """

    def __init__(self, log_path: Path = None):
        """
        Initialize telemetry logger.

        Args:
            log_path: Path to JSONL log file (default: logs/retrieval_runs.jsonl)
        """
        self.log_path = log_path or TELEMETRY_LOG_PATH
        self.enabled = is_telemetry_enabled()
        self.telemetry = RetrievalTelemetry()
        self._start_time = time.perf_counter()
        self._timers: dict[str, float] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.telemetry.errors.append(f"{exc_type.__name__}: {exc_val}")
        if self.enabled:
            self.finalize()
        return False

    def set_query(self, query: str, n_requested: int = 10):
        """Set the query being processed."""
        self.telemetry.query = query
        self.telemetry.query_hash = self._hash_query(query)
        self.telemetry.n_requested = n_requested

    def _hash_query(self, query: str) -> str:
        """Create a simple hash for grouping similar queries."""
        import hashlib
        normalized = " ".join(query.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    class _Timer:
        """Context manager for timing operations."""

        def __init__(self, logger: "TelemetryLogger", name: str):
            self.logger = logger
            self.name = name
            self.start = 0.0

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed_ms = (time.perf_counter() - self.start) * 1000
            self.logger._timers[self.name] = elapsed_ms

    def time(self, operation: str) -> "_Timer":
        """
        Time an operation.

        Usage:
            with tl.time("vector"):
                results = search_vector(...)
        """
        return self._Timer(self, operation)

    def add_vector_results(self, results: list[dict]):
        """Add vector search results."""
        self.telemetry.vector_results = [
            {"passage_id": r.get("passage_id"), "score": r.get("score", 0)}
            for r in results[:20]  # Limit to top 20
        ]

    def add_fts_results(self, results: list[dict]):
        """Add full-text search results."""
        self.telemetry.fts_results = [
            {"passage_id": r.get("passage_id"), "score": r.get("score", 0)}
            for r in results[:20]
        ]

    def add_graph_results(self, results: list[dict]):
        """Add graph traversal results."""
        self.telemetry.graph_results = [
            {"passage_id": r.get("passage_id"), "score": r.get("score", 0)}
            for r in results[:20]
        ]

    def add_fused_results(self, results: list[dict]):
        """Add RRF-fused results with final ranks."""
        self.telemetry.fused_ranks = [
            {
                "passage_id": r.get("passage_id"),
                "rank": i + 1,
                "rrf_score": r.get("score", 0),
            }
            for i, r in enumerate(results[:20])
        ]

    def add_reranked_results(self, results: list[dict]):
        """Add reranker scores."""
        self.telemetry.rerank_applied = True
        self.telemetry.reranker_scores = [
            {
                "passage_id": r.get("passage_id"),
                "rank": i + 1,
                "rerank_score": r.get("rerank_score", 0),
            }
            for i, r in enumerate(results[:20])
        ]

    def set_filters(self, filters: dict):
        """Record applied filters."""
        self.telemetry.filters_applied = filters

    def add_error(self, error: str):
        """Record an error."""
        self.telemetry.errors.append(error)

    def finalize(self):
        """Calculate final metrics and write to log."""
        if not self.enabled:
            return

        # Set timing from recorded timers
        self.telemetry.vector_latency_ms = self._timers.get("vector", 0)
        self.telemetry.fts_latency_ms = self._timers.get("fts", 0)
        self.telemetry.graph_latency_ms = self._timers.get("graph", 0)
        self.telemetry.fusion_latency_ms = self._timers.get("fusion", 0)
        self.telemetry.rerank_latency_ms = self._timers.get("rerank", 0)
        self.telemetry.total_latency_ms = (time.perf_counter() - self._start_time) * 1000

        # Set result count
        if self.telemetry.reranker_scores:
            self.telemetry.n_returned = len(self.telemetry.reranker_scores)
        elif self.telemetry.fused_ranks:
            self.telemetry.n_returned = len(self.telemetry.fused_ranks)

        # Write to log file
        self._write_log()

    def _write_log(self):
        """Append telemetry to JSONL log file."""
        try:
            # Ensure log directory exists
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

            # Append to JSONL
            with open(self.log_path, "a") as f:
                json.dump(self.telemetry.to_dict(), f)
                f.write("\n")

            logger.debug(f"Telemetry logged: {self.telemetry.run_id}")

        except Exception as e:
            logger.warning(f"Failed to write telemetry: {e}")


def log_retrieval(
    query: str,
    results: list[dict],
    latency_ms: float,
    source: str = "hybrid",
    **kwargs: Any,
) -> None:
    """
    Simple function to log a retrieval result.

    For quick logging without the full TelemetryLogger context manager.

    Args:
        query: Search query
        results: List of result dicts with passage_id and score
        latency_ms: Total latency in milliseconds
        source: Search source type (hybrid, vector, fts)
        **kwargs: Additional metadata
    """
    if not is_telemetry_enabled():
        return

    telemetry = RetrievalTelemetry(
        query=query,
        query_hash=TelemetryLogger({})._hash_query(query),
        n_returned=len(results),
        total_latency_ms=latency_ms,
    )

    # Add results based on source
    if source == "vector":
        telemetry.vector_results = results[:20]
    elif source == "fts":
        telemetry.fts_results = results[:20]
    else:
        telemetry.fused_ranks = [
            {"passage_id": r.get("passage_id"), "rank": i + 1, "score": r.get("score", 0)}
            for i, r in enumerate(results[:20])
        ]

    # Add extra metadata
    for key, value in kwargs.items():
        if hasattr(telemetry, key):
            setattr(telemetry, key, value)

    # Write to log
    try:
        TELEMETRY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TELEMETRY_LOG_PATH, "a") as f:
            json.dump(telemetry.to_dict(), f)
            f.write("\n")
    except Exception as e:
        logger.warning(f"Failed to log telemetry: {e}")


def read_telemetry_logs(
    log_path: Path = None,
    query_hash: str = None,
    limit: int = 100,
) -> list[RetrievalTelemetry]:
    """
    Read telemetry logs from JSONL file.

    Args:
        log_path: Path to log file (default: logs/retrieval_runs.jsonl)
        query_hash: Optional filter by query hash
        limit: Maximum number of records to return

    Returns:
        List of RetrievalTelemetry objects
    """
    log_path = log_path or TELEMETRY_LOG_PATH

    if not log_path.exists():
        return []

    results = []
    with open(log_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if query_hash and data.get("query_hash") != query_hash:
                    continue
                results.append(data)
                if len(results) >= limit:
                    break
            except json.JSONDecodeError:
                continue

    return results
