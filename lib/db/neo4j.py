"""
Neo4j connection management for Polymath v3.

Provides driver management and session helpers for graph operations.
"""

import logging
from contextlib import contextmanager
from typing import Optional, Any, Generator

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from lib.config import config

logger = logging.getLogger(__name__)

# Global driver instance
_driver: Optional[Driver] = None


def get_neo4j_driver() -> Driver:
    """
    Get or create the global Neo4j driver.

    Returns:
        Neo4j Driver instance
    """
    global _driver

    if _driver is None:
        logger.info(f"Creating Neo4j driver for {config.NEO4J_URI}")
        _driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
        )

    return _driver


@contextmanager
def get_neo4j_session(database: str = "neo4j") -> Generator[Session, None, None]:
    """
    Get a Neo4j session.

    Usage:
        with get_neo4j_session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
    """
    driver = get_neo4j_driver()
    with driver.session(database=database) as session:
        yield session


def execute_cypher(
    query: str,
    params: Optional[dict] = None,
    database: str = "neo4j"
) -> list[dict]:
    """
    Execute a Cypher query and return results.

    Args:
        query: Cypher query string
        params: Query parameters
        database: Database name

    Returns:
        List of result records as dicts
    """
    with get_neo4j_session(database) as session:
        result = session.run(query, params or {})
        return [dict(record) for record in result]


def execute_write(
    query: str,
    params: Optional[dict] = None,
    database: str = "neo4j"
) -> dict[str, Any]:
    """
    Execute a write query and return summary.

    Args:
        query: Cypher query string
        params: Query parameters
        database: Database name

    Returns:
        Summary dict with counters
    """
    with get_neo4j_session(database) as session:
        result = session.run(query, params or {})
        summary = result.consume()
        return {
            "nodes_created": summary.counters.nodes_created,
            "nodes_deleted": summary.counters.nodes_deleted,
            "relationships_created": summary.counters.relationships_created,
            "relationships_deleted": summary.counters.relationships_deleted,
            "properties_set": summary.counters.properties_set,
        }


def batch_create_nodes(
    label: str,
    nodes: list[dict],
    merge_key: str = "name",
    database: str = "neo4j"
) -> int:
    """
    Batch create or merge nodes.

    Args:
        label: Node label
        nodes: List of node property dicts
        merge_key: Property to use for MERGE
        database: Database name

    Returns:
        Number of nodes created/merged
    """
    if not nodes:
        return 0

    query = f"""
    UNWIND $nodes AS node
    MERGE (n:{label} {{{merge_key}: node.{merge_key}}})
    SET n += node
    """

    with get_neo4j_session(database) as session:
        result = session.run(query, {"nodes": nodes})
        summary = result.consume()
        return summary.counters.nodes_created + summary.counters.properties_set


def create_relationship(
    from_label: str,
    from_key: str,
    from_value: Any,
    rel_type: str,
    to_label: str,
    to_key: str,
    to_value: Any,
    properties: Optional[dict] = None,
    database: str = "neo4j"
) -> bool:
    """
    Create a relationship between two nodes.

    Args:
        from_label: Source node label
        from_key: Source node match key
        from_value: Source node match value
        rel_type: Relationship type
        to_label: Target node label
        to_key: Target node match key
        to_value: Target node match value
        properties: Relationship properties

    Returns:
        True if relationship was created
    """
    query = f"""
    MATCH (a:{from_label} {{{from_key}: $from_value}})
    MATCH (b:{to_label} {{{to_key}: $to_value}})
    MERGE (a)-[r:{rel_type}]->(b)
    SET r += $properties
    """

    with get_neo4j_session(database) as session:
        result = session.run(query, {
            "from_value": from_value,
            "to_value": to_value,
            "properties": properties or {},
        })
        summary = result.consume()
        return summary.counters.relationships_created > 0


def get_node_counts() -> dict[str, int]:
    """Get counts of all node labels."""
    query = """
    CALL db.labels() YIELD label
    CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) as count', {})
    YIELD value
    RETURN label, value.count as count
    """

    try:
        results = execute_cypher(query)
        return {r["label"]: r["count"] for r in results}
    except Exception:
        # Fallback without APOC
        query = "MATCH (n) RETURN labels(n)[0] as label, count(*) as count"
        results = execute_cypher(query)
        counts = {}
        for r in results:
            label = r["label"]
            if label:
                counts[label] = counts.get(label, 0) + r["count"]
        return counts


def check_health() -> dict[str, Any]:
    """Check Neo4j health and return status."""
    result = {
        "status": "unknown",
        "connection": False,
        "node_counts": {},
    }

    try:
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        result["connection"] = True

        # Get node counts
        result["node_counts"] = get_node_counts()
        result["status"] = "healthy"

    except AuthError as e:
        result["status"] = "auth_error"
        result["error"] = str(e)
    except ServiceUnavailable as e:
        result["status"] = "unavailable"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "unhealthy"
        result["error"] = str(e)

    return result


def close_driver() -> None:
    """Close the Neo4j driver."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
        logger.info("Neo4j driver closed")
