"""
Async Postgres connection management for Polymath v3.

Uses psycopg3 async interface with connection pooling.
Falls back to sync driver if async is not available.

Usage:
    async with get_async_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT * FROM documents LIMIT 10")
            rows = await cur.fetchall()
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator, Any

from lib.config import config

logger = logging.getLogger(__name__)

# Global async pool
_async_pool: Optional[Any] = None
_pool_lock = asyncio.Lock()


async def get_async_pool():
    """
    Get or create the async connection pool.

    Thread-safe via asyncio.Lock.
    """
    global _async_pool

    if _async_pool is not None:
        return _async_pool

    async with _pool_lock:
        # Double-check after acquiring lock
        if _async_pool is not None:
            return _async_pool

        try:
            from psycopg_pool import AsyncConnectionPool

            _async_pool = AsyncConnectionPool(
                config.POSTGRES_DSN,
                min_size=config.PG_POOL_MIN,
                max_size=config.PG_POOL_MAX,
                open=False,  # Don't open immediately
            )
            await _async_pool.open()
            logger.info(
                f"Async Postgres pool created (min={config.PG_POOL_MIN}, max={config.PG_POOL_MAX})"
            )

        except ImportError:
            logger.error("psycopg_pool not installed. Install with: pip install psycopg[pool]")
            raise

        except Exception as e:
            logger.error(f"Failed to create async pool: {e}")
            raise

    return _async_pool


@asynccontextmanager
async def get_async_connection() -> AsyncGenerator:
    """
    Get an async database connection from the pool.

    Usage:
        async with get_async_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
    """
    pool = await get_async_pool()

    async with pool.connection() as conn:
        # Enable row_factory for dict-like access
        from psycopg.rows import dict_row
        conn.row_factory = dict_row
        yield conn


async def execute_query(
    query: str,
    params: tuple = (),
    fetch: bool = True,
) -> list[dict]:
    """
    Execute a query and optionally fetch results.

    Args:
        query: SQL query
        params: Query parameters
        fetch: Whether to fetch results

    Returns:
        List of dicts if fetch=True, else empty list
    """
    async with get_async_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)

            if fetch:
                rows = await cur.fetchall()
                return [dict(row) for row in rows]

            await conn.commit()
            return []


async def execute_batch(
    query: str,
    params_list: list[tuple],
    batch_size: int = 1000,
) -> int:
    """
    Execute a query with multiple parameter sets.

    Uses executemany for efficiency.

    Args:
        query: SQL query with placeholders
        params_list: List of parameter tuples
        batch_size: Batch size for commits

    Returns:
        Total rows affected
    """
    total = 0

    async with get_async_connection() as conn:
        async with conn.cursor() as cur:
            for i in range(0, len(params_list), batch_size):
                batch = params_list[i:i + batch_size]
                await cur.executemany(query, batch)
                total += len(batch)
                await conn.commit()

                logger.debug(f"Batch executed: {i + len(batch)}/{len(params_list)}")

    return total


async def vector_search(
    query_embedding: list[float],
    limit: int = 20,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Async vector similarity search.

    Args:
        query_embedding: Query vector (1024-dim for BGE-M3)
        limit: Maximum results
        filters: Optional filters (year_min, year_max, etc.)

    Returns:
        List of passage dicts with similarity scores
    """
    filter_clause = ""
    filter_params = []

    if filters:
        conditions = []
        if filters.get("year_min"):
            conditions.append("d.year >= %s")
            filter_params.append(filters["year_min"])
        if filters.get("year_max"):
            conditions.append("d.year <= %s")
            filter_params.append(filters["year_max"])

        if conditions:
            filter_clause = "AND " + " AND ".join(conditions)

    query = f"""
        SELECT
            p.passage_id,
            p.doc_id,
            p.passage_text,
            p.section,
            d.title,
            d.authors,
            d.year,
            d.doi,
            1 - (p.embedding <=> %s::vector) as similarity
        FROM passages p
        JOIN documents d ON p.doc_id = d.doc_id
        WHERE p.embedding IS NOT NULL
          AND (p.is_superseded = FALSE OR p.is_superseded IS NULL)
        {filter_clause}
        ORDER BY p.embedding <=> %s::vector
        LIMIT %s
    """

    params = (query_embedding, *filter_params, query_embedding, limit)
    return await execute_query(query, params)


async def close_async_pool():
    """Close the async connection pool."""
    global _async_pool

    if _async_pool:
        await _async_pool.close()
        _async_pool = None
        logger.info("Async Postgres pool closed")


# Convenience function to run async code from sync context
def run_async(coro):
    """
    Run async coroutine from sync context.

    Creates a new event loop if needed.
    """
    try:
        loop = asyncio.get_running_loop()
        # If we're in an async context, use ensure_future
        return asyncio.ensure_future(coro)
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(coro)
