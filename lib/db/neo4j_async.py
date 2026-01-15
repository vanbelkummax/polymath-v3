"""
Async Neo4j connection management for Polymath v3.

Uses neo4j async driver for non-blocking graph queries.

Usage:
    async with get_async_session() as session:
        result = await session.run("MATCH (n) RETURN n LIMIT 10")
        records = await result.data()
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator, Any

from lib.config import config

logger = logging.getLogger(__name__)

# Global async driver
_async_driver: Optional[Any] = None
_driver_lock = asyncio.Lock()


async def get_async_driver():
    """
    Get or create the async Neo4j driver.

    Thread-safe via asyncio.Lock.
    """
    global _async_driver

    if _async_driver is not None:
        return _async_driver

    async with _driver_lock:
        # Double-check after acquiring lock
        if _async_driver is not None:
            return _async_driver

        try:
            from neo4j import AsyncGraphDatabase

            _async_driver = AsyncGraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            )

            # Verify connectivity
            await _async_driver.verify_connectivity()
            logger.info(f"Async Neo4j driver connected to {config.NEO4J_URI}")

        except ImportError:
            logger.error("neo4j async driver not available. Install with: pip install neo4j")
            raise

        except Exception as e:
            logger.error(f"Failed to create async Neo4j driver: {e}")
            raise

    return _async_driver


@asynccontextmanager
async def get_async_session() -> AsyncGenerator:
    """
    Get an async Neo4j session.

    Usage:
        async with get_async_session() as session:
            result = await session.run("MATCH (n) RETURN n")
    """
    driver = await get_async_driver()
    async with driver.session() as session:
        yield session


async def execute_cypher(
    query: str,
    params: Optional[dict] = None,
) -> list[dict]:
    """
    Execute a Cypher query and return results.

    Args:
        query: Cypher query
        params: Query parameters

    Returns:
        List of record dicts
    """
    params = params or {}

    async with get_async_session() as session:
        result = await session.run(query, params)
        records = await result.data()
        return records


async def execute_cypher_write(
    query: str,
    params: Optional[dict] = None,
) -> dict:
    """
    Execute a write Cypher query in a transaction.

    Args:
        query: Cypher query
        params: Query parameters

    Returns:
        Summary dict with counters
    """
    params = params or {}

    async def _write_tx(tx):
        result = await tx.run(query, params)
        summary = await result.consume()
        return summary.counters.__dict__

    async with get_async_session() as session:
        return await session.execute_write(_write_tx)


async def batch_create_nodes(
    nodes: list[dict],
    label: str,
    batch_size: int = 1000,
) -> int:
    """
    Batch create nodes with UNWIND.

    Args:
        nodes: List of node property dicts
        label: Node label
        batch_size: Batch size

    Returns:
        Total nodes created
    """
    total = 0

    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]

        query = f"""
        UNWIND $nodes as props
        CREATE (n:{label})
        SET n = props
        """

        result = await execute_cypher_write(query, {"nodes": batch})
        total += result.get("nodes_created", len(batch))

        logger.debug(f"Created {i + len(batch)}/{len(nodes)} {label} nodes")

    return total


async def batch_merge_nodes(
    nodes: list[dict],
    label: str,
    merge_key: str,
    batch_size: int = 1000,
) -> int:
    """
    Batch merge nodes with UNWIND.

    Args:
        nodes: List of node property dicts
        label: Node label
        merge_key: Property to merge on
        batch_size: Batch size

    Returns:
        Total nodes processed
    """
    total = 0

    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]

        query = f"""
        UNWIND $nodes as props
        MERGE (n:{label} {{{merge_key}: props.{merge_key}}})
        SET n = props, n.synced_at = datetime()
        """

        await execute_cypher_write(query, {"nodes": batch})
        total += len(batch)

        logger.debug(f"Merged {i + len(batch)}/{len(nodes)} {label} nodes")

    return total


async def concept_search(
    query: str,
    limit: int = 10,
) -> list[dict]:
    """
    Search concepts by name using fulltext index.

    Args:
        query: Search query
        limit: Maximum results

    Returns:
        List of matching concepts with scores
    """
    cypher = """
    CALL db.index.fulltext.queryNodes("concept_names", $query)
    YIELD node, score
    RETURN node.name as name, labels(node)[0] as type, score
    ORDER BY score DESC
    LIMIT $limit
    """

    return await execute_cypher(cypher, {"query": query, "limit": limit})


async def find_passages_by_concept(
    concept_name: str,
    concept_type: str = "METHOD",
    limit: int = 20,
) -> list[dict]:
    """
    Find passages that mention a concept.

    Args:
        concept_name: Concept name
        concept_type: Concept type (METHOD, PROBLEM, etc.)
        limit: Maximum results

    Returns:
        List of passage dicts
    """
    cypher = f"""
    MATCH (p:Passage)-[r:MENTIONS]->(c:{concept_type} {{name: $name}})
    RETURN p.passage_id as passage_id,
           p.doc_id as doc_id,
           r.confidence as confidence
    ORDER BY r.confidence DESC
    LIMIT $limit
    """

    return await execute_cypher(cypher, {"name": concept_name, "limit": limit})


async def close_async_driver():
    """Close the async Neo4j driver."""
    global _async_driver

    if _async_driver:
        await _async_driver.close()
        _async_driver = None
        logger.info("Async Neo4j driver closed")
