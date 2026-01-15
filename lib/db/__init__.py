"""
Database connection management for Polymath v3.

Provides connection pooling for Postgres and Neo4j.

Sync drivers (default):
    from lib.db import get_pg_pool, get_neo4j_driver

Async drivers (for MCP server, high-throughput):
    from lib.db.postgres_async import get_async_connection, vector_search
    from lib.db.neo4j_async import get_async_session, execute_cypher
"""

from .postgres import get_pg_pool, get_pg_connection
from .neo4j import get_neo4j_driver, get_neo4j_session

# Async modules available via direct import:
# from lib.db.postgres_async import get_async_connection
# from lib.db.neo4j_async import get_async_session

__all__ = [
    "get_pg_pool",
    "get_pg_connection",
    "get_neo4j_driver",
    "get_neo4j_session",
]
