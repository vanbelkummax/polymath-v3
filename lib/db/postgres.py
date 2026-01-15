"""
Postgres connection management with pgvector support.

Uses psycopg3 with connection pooling for efficient database access.
"""

import logging
from contextlib import contextmanager
from typing import Optional, Any, Generator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from lib.config import config

logger = logging.getLogger(__name__)

# Global connection pool
_pool: Optional[ConnectionPool] = None


def get_pg_pool(min_size: int = None, max_size: int = None) -> ConnectionPool:
    """
    Get or create the global connection pool.

    Pool size is configured via PG_POOL_MIN and PG_POOL_MAX environment
    variables, decoupled from worker threads to prevent resource contention
    when multiple clients (MCP server, CLI, batch jobs) run concurrently.

    Args:
        min_size: Override minimum connections (default: PG_POOL_MIN)
        max_size: Override maximum connections (default: PG_POOL_MAX)

    Returns:
        ConnectionPool instance
    """
    global _pool

    if _pool is None:
        # Use config values, decoupled from NUM_WORKERS
        min_size = min_size if min_size is not None else config.PG_POOL_MIN
        max_size = max_size if max_size is not None else config.PG_POOL_MAX

        logger.info(f"Creating Postgres connection pool (min={min_size}, max={max_size})")
        _pool = ConnectionPool(
            config.POSTGRES_DSN,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            configure=_configure_connection,
        )

    return _pool


def _configure_connection(conn: psycopg.Connection) -> None:
    """Configure a new connection (register pgvector, etc.)."""
    register_vector(conn)


@contextmanager
def get_pg_connection() -> Generator[psycopg.Connection, None, None]:
    """
    Get a connection from the pool.

    Usage:
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM documents LIMIT 10")
                results = cur.fetchall()
    """
    pool = get_pg_pool()
    with pool.connection() as conn:
        yield conn


def execute_query(
    query: str,
    params: Optional[tuple] = None,
    fetch: bool = True
) -> Optional[list[dict]]:
    """
    Execute a query and optionally fetch results.

    Args:
        query: SQL query string
        params: Query parameters
        fetch: Whether to fetch and return results

    Returns:
        List of result dicts if fetch=True, else None
    """
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchall()
            conn.commit()
            return None


def execute_many(query: str, params_list: list[tuple]) -> int:
    """
    Execute a query with multiple parameter sets.

    Args:
        query: SQL query string with placeholders
        params_list: List of parameter tuples

    Returns:
        Number of rows affected
    """
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, params_list)
            conn.commit()
            return cur.rowcount


def get_stats() -> dict[str, int]:
    """Get system statistics from the database."""
    stats = {}

    queries = {
        "documents": "SELECT COUNT(*) FROM documents",
        "passages": "SELECT COUNT(*) FROM passages",
        "passages_with_embeddings": "SELECT COUNT(*) FROM passages WHERE embedding IS NOT NULL",
        "concepts": "SELECT COUNT(*) FROM passage_concepts",
        "code_repos": "SELECT COUNT(*) FROM code_repos",
        "code_chunks": "SELECT COUNT(*) FROM code_chunks",
    }

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            for name, query in queries.items():
                try:
                    cur.execute(query)
                    result = cur.fetchone()
                    stats[name] = result["count"] if result else 0
                except Exception as e:
                    logger.warning(f"Failed to get stat {name}: {e}")
                    stats[name] = 0

    return stats


def check_health() -> dict[str, Any]:
    """Check database health and return status."""
    result = {
        "status": "unknown",
        "connection": False,
        "pgvector": False,
        "tables": [],
    }

    try:
        with get_pg_connection() as conn:
            result["connection"] = True

            with conn.cursor() as cur:
                # Check pgvector extension
                cur.execute(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                )
                row = cur.fetchone()
                if row:
                    result["pgvector"] = True
                    result["pgvector_version"] = row["extversion"]

                # Check tables exist
                cur.execute("""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public'
                    AND tablename IN ('documents', 'passages', 'passage_concepts',
                                     'code_repos', 'code_chunks')
                """)
                result["tables"] = [row["tablename"] for row in cur.fetchall()]

        result["status"] = "healthy" if result["pgvector"] else "degraded"

    except Exception as e:
        result["status"] = "unhealthy"
        result["error"] = str(e)

    return result


def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("Postgres connection pool closed")


def configure_for_batch_worker() -> None:
    """
    Configure minimal connection pool for batch worker processes.

    IMPORTANT: Call this at the start of batch scripts to prevent
    connection exhaustion when running multiple workers.

    Each worker gets exactly 1 connection (min=1, max=1), which:
    - Prevents connection pile-up across distributed workers
    - Avoids "too many connections" errors
    - Is sufficient since batch work is serialized per-worker

    MULTIPROCESSING WARNING (fork vs spawn):
    -----------------------------------------
    Python's default "fork" context on Linux copies file descriptors,
    which can cause pool corruption if workers inherit parent connections.

    Safe patterns:
    1. Call configure_for_batch_worker() INSIDE each worker process
    2. Or use spawn context: multiprocessing.set_start_method('spawn')
    3. Or use ThreadPoolExecutor instead (recommended for I/O bound work)

    The IngestPipeline uses ThreadPoolExecutor, which is safe.
    GCP Batch workers are separate processes, so also safe.

    Usage:
        # At start of batch script (inside worker process)
        from lib.db.postgres import configure_for_batch_worker
        configure_for_batch_worker()

        # Then use get_pg_pool() or get_pg_connection() as normal
    """
    global _pool

    if _pool is not None:
        logger.warning("Pool already initialized, skipping batch configuration")
        return

    # Force minimal pool for batch workers
    logger.info("Configuring minimal connection pool for batch worker (1 connection)")
    _pool = ConnectionPool(
        config.POSTGRES_DSN,
        min_size=1,
        max_size=1,
        kwargs={"row_factory": dict_row},
        configure=_configure_connection,
    )
