#!/usr/bin/env python3
"""
Sync Postgres data to Neo4j graph database.

Creates:
- Paper nodes from documents
- Passage nodes from passages
- Concept nodes (METHOD, PROBLEM, DOMAIN, etc.) from passage_concepts
- MENTIONS relationships between passages and concepts
- SIMILAR_TO relationships between concepts

Usage:
    python scripts/sync_neo4j.py --full          # Full sync
    python scripts/sync_neo4j.py --incremental   # Only new data
    python scripts/sync_neo4j.py --concepts-only # Only sync concepts
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config
from lib.db.postgres import get_pg_pool
from lib.db.neo4j import get_neo4j_driver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Batch size for Neo4j operations
BATCH_SIZE = 1000


def prune_superseded(driver):
    """
    Remove Passage nodes from Neo4j that have been marked as superseded in Postgres.

    This is the "Graph Garbage Collector" - ensures Neo4j reflects Postgres soft deletes.
    Call this before syncing new data to clean up stale nodes.
    """
    logger.info("Pruning superseded passages from graph...")

    pool = get_pg_pool()

    # 1. Get IDs of superseded passages from Postgres
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT passage_id
                FROM passages
                WHERE is_superseded = TRUE
                """
            )
            superseded_ids = [str(row["passage_id"]) for row in cur.fetchall()]

    if not superseded_ids:
        logger.info("  No superseded passages found to prune.")
        return 0

    logger.info(f"  Found {len(superseded_ids)} superseded passages to prune.")
    pruned = 0

    # 2. Delete them from Neo4j in batches
    # DETACH DELETE removes the node AND its relationships (MENTIONS, FROM_PAPER)
    for i in range(0, len(superseded_ids), BATCH_SIZE):
        batch = superseded_ids[i:i + BATCH_SIZE]

        result, _, _ = driver.execute_query(
            """
            UNWIND $ids as pid
            MATCH (p:Passage {passage_id: pid})
            DETACH DELETE p
            RETURN count(*) as deleted
            """,
            ids=batch,
        )

        batch_deleted = result[0]["deleted"] if result else 0
        pruned += batch_deleted
        logger.info(f"  Pruned {i + len(batch)}/{len(superseded_ids)} passages ({pruned} deleted)")

    logger.info(f"✓ Pruning complete: removed {pruned} stale nodes")
    return pruned


def sync_papers(driver, incremental: bool = False):
    """Sync documents to Paper nodes."""
    logger.info("Syncing Paper nodes...")

    pool = get_pg_pool()

    with pool.connection() as conn:
        with conn.cursor() as cur:
            if incremental:
                cur.execute(
                    """
                    SELECT doc_id, title, authors, year, doi, pmid, arxiv_id
                    FROM documents
                    WHERE graph_synced_at IS NULL
                    ORDER BY created_at
                    """
                )
            else:
                cur.execute(
                    """
                    SELECT doc_id, title, authors, year, doi, pmid, arxiv_id
                    FROM documents
                    ORDER BY created_at
                    """
                )

            docs = cur.fetchall()

    logger.info(f"Found {len(docs)} documents to sync")

    # Batch insert
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]

        params = []
        for doc in batch:
            params.append({
                "doc_id": str(doc["doc_id"]),
                "title": doc["title"],
                "authors": doc["authors"] or [],
                "year": doc["year"],
                "doi": doc["doi"],
                "pmid": doc["pmid"],
                "arxiv_id": doc["arxiv_id"],
            })

        driver.execute_query(
            """
            UNWIND $papers as p
            MERGE (paper:Paper {doc_id: p.doc_id})
            SET paper.title = p.title,
                paper.authors = p.authors,
                paper.year = p.year,
                paper.doi = p.doi,
                paper.pmid = p.pmid,
                paper.arxiv_id = p.arxiv_id,
                paper.synced_at = datetime()
            """,
            papers=params,
        )

        logger.info(f"  Synced {i + len(batch)}/{len(docs)} papers")

    # Update sync timestamps
    with pool.connection() as conn:
        with conn.cursor() as cur:
            doc_ids = [str(d["doc_id"]) for d in docs]
            cur.execute(
                """
                UPDATE documents
                SET graph_synced_at = NOW()
                WHERE doc_id = ANY(%s)
                """,
                (doc_ids,),
            )
            conn.commit()

    logger.info(f"✓ Synced {len(docs)} Paper nodes")


def sync_passages(driver, incremental: bool = False):
    """Sync passages to Passage nodes (excludes superseded)."""
    logger.info("Syncing Passage nodes...")

    pool = get_pg_pool()

    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Only sync active passages (not superseded)
            cur.execute(
                """
                SELECT passage_id, doc_id, section, parent_section
                FROM passages
                WHERE is_superseded = FALSE OR is_superseded IS NULL
                ORDER BY created_at
                """
            )
            passages = cur.fetchall()

    logger.info(f"Found {len(passages)} passages to sync")

    # Batch insert
    for i in range(0, len(passages), BATCH_SIZE):
        batch = passages[i:i + BATCH_SIZE]

        params = []
        for p in batch:
            params.append({
                "passage_id": str(p["passage_id"]),
                "doc_id": str(p["doc_id"]),
                "section": p["section"],
                "parent_section": p["parent_section"],
            })

        driver.execute_query(
            """
            UNWIND $passages as p
            MERGE (passage:Passage {passage_id: p.passage_id})
            SET passage.doc_id = p.doc_id,
                passage.section = p.section,
                passage.parent_section = p.parent_section,
                passage.synced_at = datetime()

            WITH passage, p
            MATCH (paper:Paper {doc_id: p.doc_id})
            MERGE (passage)-[:FROM_PAPER]->(paper)
            """,
            passages=params,
        )

        logger.info(f"  Synced {i + len(batch)}/{len(passages)} passages")

    logger.info(f"✓ Synced {len(passages)} Passage nodes")


def sync_concepts(driver):
    """Sync concepts to typed nodes (METHOD, PROBLEM, etc.)."""
    logger.info("Syncing Concept nodes...")

    pool = get_pg_pool()

    # Get distinct concepts by type
    concept_types = ["method", "problem", "domain", "dataset", "metric", "entity"]

    for concept_type in concept_types:
        label = concept_type.upper()

        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT concept_name
                    FROM passage_concepts
                    WHERE concept_type = %s
                    """,
                    (concept_type,),
                )
                concepts = [row["concept_name"] for row in cur.fetchall()]

        if not concepts:
            continue

        logger.info(f"  Syncing {len(concepts)} {label} nodes...")

        # Batch insert
        for i in range(0, len(concepts), BATCH_SIZE):
            batch = concepts[i:i + BATCH_SIZE]

            driver.execute_query(
                f"""
                UNWIND $names as name
                MERGE (c:{label} {{name: name}})
                SET c.synced_at = datetime()
                """,
                names=batch,
            )

        logger.info(f"    ✓ Synced {len(concepts)} {label} nodes")


def sync_mentions(driver):
    """Create MENTIONS relationships between passages and concepts."""
    logger.info("Syncing MENTIONS relationships...")

    pool = get_pg_pool()

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT passage_id, concept_name, concept_type, confidence
                FROM passage_concepts
                """
            )
            mentions = cur.fetchall()

    logger.info(f"Found {len(mentions)} mentions to sync")

    # Group by concept type for efficient querying
    by_type = {}
    for m in mentions:
        t = m["concept_type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append({
            "passage_id": str(m["passage_id"]),
            "concept_name": m["concept_name"],
            "confidence": m["confidence"],
        })

    for concept_type, type_mentions in by_type.items():
        label = concept_type.upper()
        logger.info(f"  Creating {len(type_mentions)} MENTIONS for {label}...")

        for i in range(0, len(type_mentions), BATCH_SIZE):
            batch = type_mentions[i:i + BATCH_SIZE]

            driver.execute_query(
                f"""
                UNWIND $mentions as m
                MATCH (p:Passage {{passage_id: m.passage_id}})
                MATCH (c:{label} {{name: m.concept_name}})
                MERGE (p)-[r:MENTIONS]->(c)
                SET r.confidence = m.confidence
                """,
                mentions=batch,
            )

        logger.info(f"    ✓ Created {len(type_mentions)} MENTIONS for {label}")


def build_similarity_edges(driver, min_cooccurrence: int = 3):
    """Build SIMILAR_TO edges between concepts based on co-occurrence."""
    logger.info("Building SIMILAR_TO edges...")

    # For each concept type, find co-occurring concepts
    concept_types = ["METHOD", "PROBLEM"]

    for label in concept_types:
        logger.info(f"  Computing similarities for {label}...")

        # Find concepts that appear in same passages
        driver.execute_query(
            f"""
            MATCH (c1:{label})<-[:MENTIONS]-(p:Passage)-[:MENTIONS]->(c2:{label})
            WHERE c1 <> c2 AND id(c1) < id(c2)
            WITH c1, c2, count(DISTINCT p) as cooccurrences
            WHERE cooccurrences >= $min_cooccurrence
            MERGE (c1)-[s:SIMILAR_TO]-(c2)
            SET s.cooccurrences = cooccurrences,
                s.score = 1.0 * cooccurrences / (cooccurrences + 10)
            """,
            min_cooccurrence=min_cooccurrence,
        )

        # Count edges
        result, _, _ = driver.execute_query(
            f"""
            MATCH (:{label})-[s:SIMILAR_TO]-(:{label})
            RETURN count(s) / 2 as edge_count
            """
        )
        edge_count = result[0]["edge_count"] if result else 0
        logger.info(f"    ✓ Created {edge_count} SIMILAR_TO edges for {label}")


def main():
    parser = argparse.ArgumentParser(description="Sync Postgres to Neo4j")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full sync (clear and rebuild)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental sync (new data only)",
    )
    parser.add_argument(
        "--papers-only",
        action="store_true",
        help="Only sync papers",
    )
    parser.add_argument(
        "--concepts-only",
        action="store_true",
        help="Only sync concepts and mentions",
    )
    parser.add_argument(
        "--build-similarity",
        action="store_true",
        help="Build SIMILAR_TO edges",
    )

    args = parser.parse_args()

    driver = get_neo4j_driver()
    start_time = time.time()

    try:
        if args.full:
            logger.info("Starting full sync...")
            prune_superseded(driver)  # Clean stale nodes first
            sync_papers(driver, incremental=False)
            sync_passages(driver, incremental=False)
            sync_concepts(driver)
            sync_mentions(driver)
            build_similarity_edges(driver)

        elif args.incremental:
            logger.info("Starting incremental sync...")
            prune_superseded(driver)  # Clean stale nodes first
            sync_papers(driver, incremental=True)
            sync_passages(driver, incremental=True)
            sync_concepts(driver)
            sync_mentions(driver)

        elif args.papers_only:
            sync_papers(driver, incremental=args.incremental)

        elif args.concepts_only:
            sync_concepts(driver)
            sync_mentions(driver)

        elif args.build_similarity:
            build_similarity_edges(driver)

        else:
            # Default: incremental
            logger.info("Starting default sync...")
            prune_superseded(driver)  # Clean stale nodes first
            sync_papers(driver, incremental=True)
            sync_concepts(driver)
            sync_mentions(driver)

        elapsed = time.time() - start_time
        logger.info(f"\n✓ Sync completed in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"Sync failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
