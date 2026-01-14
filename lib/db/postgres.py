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


def get_pg_pool(min_size: int = 2, max_size: int = 10) -> ConnectionPool:
    """
    Get or create the global connection pool.

    Args:
        min_size: Minimum number of connections to maintain
        max_size: Maximum number of connections allowed

    Returns:
        ConnectionPool instance
    """
    global _pool

    if _pool is None:
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
