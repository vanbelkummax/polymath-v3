#!/usr/bin/env python3
"""
Sync batch-v1 concepts and their passages to Neo4j.

Creates:
- Passage nodes for passages with batch-v1 concepts
- Concept nodes (typed: METHOD, PROBLEM, DOMAIN, ENTITY)
- MENTIONS relationships
- FROM_PAPER relationships

Usage:
    python scripts/sync_neo4j_batch.py --dry-run
    python scripts/sync_neo4j_batch.py
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from neo4j import GraphDatabase
from lib.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 500


def get_batch_v1_data(conn):
    """Get all batch-v1 passages, concepts, and documents."""
    cur = conn.cursor()

    # Get passages
    cur.execute("""
        SELECT DISTINCT p.passage_id, p.doc_id, p.section,
               LEFT(p.passage_text, 200) as text_preview
        FROM passages p
        JOIN passage_concepts pc ON p.passage_id = pc.passage_id
        WHERE pc.extractor_version = 'batch-v1'
    """)
    passages = cur.fetchall()

    # Get concepts
    cur.execute("""
        SELECT DISTINCT concept_name, concept_type
        FROM passage_concepts
        WHERE extractor_version = 'batch-v1'
    """)
    concepts = cur.fetchall()

    # Get mentions (passage -> concept relationships)
    cur.execute("""
        SELECT passage_id, concept_name, concept_type, confidence
        FROM passage_concepts
        WHERE extractor_version = 'batch-v1'
    """)
    mentions = cur.fetchall()

    # Get documents
    cur.execute("""
        SELECT DISTINCT d.doc_id, d.title, d.year, d.doi
        FROM documents d
        JOIN passages p ON d.doc_id = p.doc_id
        JOIN passage_concepts pc ON p.passage_id = pc.passage_id
        WHERE pc.extractor_version = 'batch-v1'
    """)
    documents = cur.fetchall()

    cur.close()
    return passages, concepts, mentions, documents


def sync_to_neo4j(driver, passages, concepts, mentions, documents, dry_run=False):
    """Sync data to Neo4j."""

    if dry_run:
        logger.info(f"[DRY RUN] Would sync:")
        logger.info(f"  - {len(documents)} Paper nodes")
        logger.info(f"  - {len(passages)} Passage nodes")
        logger.info(f"  - {len(concepts)} Concept nodes")
        logger.info(f"  - {len(mentions)} MENTIONS relationships")
        return

    # 1. Sync Paper nodes
    logger.info(f"Syncing {len(documents)} Paper nodes...")
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        params = [{"doc_id": str(d[0]), "title": d[1], "year": d[2], "doi": d[3]} for d in batch]
        driver.execute_query("""
            UNWIND $docs as d
            MERGE (p:Paper {doc_id: d.doc_id})
            SET p.title = d.title, p.year = d.year, p.doi = d.doi, p.synced_at = datetime()
        """, docs=params)
    logger.info(f"  ✓ Papers synced")

    # 2. Sync Passage nodes with FROM_PAPER
    logger.info(f"Syncing {len(passages)} Passage nodes...")
    for i in range(0, len(passages), BATCH_SIZE):
        batch = passages[i:i + BATCH_SIZE]
        params = [{"passage_id": str(p[0]), "doc_id": str(p[1]), "section": p[2], "text_preview": p[3]} for p in batch]
        driver.execute_query("""
            UNWIND $passages as p
            MERGE (passage:Passage {passage_id: p.passage_id})
            SET passage.doc_id = p.doc_id,
                passage.section = p.section,
                passage.text_preview = p.text_preview,
                passage.synced_at = datetime()
            WITH passage, p
            MATCH (paper:Paper {doc_id: p.doc_id})
            MERGE (passage)-[:FROM_PAPER]->(paper)
        """, passages=params)
    logger.info(f"  ✓ Passages synced")

    # 3. Sync Concept nodes (typed)
    logger.info(f"Syncing {len(concepts)} Concept nodes...")
    by_type = {}
    for name, ctype in concepts:
        if ctype not in by_type:
            by_type[ctype] = []
        by_type[ctype].append(name)

    for ctype, names in by_type.items():
        driver.execute_query("""
            UNWIND $names as name
            MERGE (c:Concept {name: name, type: $ctype})
            SET c.synced_at = datetime()
        """, names=names, ctype=ctype)
        logger.info(f"    ✓ {len(names)} {ctype} concepts")
    logger.info(f"  ✓ Concepts synced")

    # 4. Sync MENTIONS relationships
    logger.info(f"Syncing {len(mentions)} MENTIONS relationships...")
    for i in range(0, len(mentions), BATCH_SIZE):
        batch = mentions[i:i + BATCH_SIZE]
        params = [{"passage_id": str(m[0]), "concept_name": m[1], "concept_type": m[2], "confidence": m[3]} for m in batch]
        driver.execute_query("""
            UNWIND $mentions as m
            MATCH (p:Passage {passage_id: m.passage_id})
            MATCH (c:Concept {name: m.concept_name, type: m.concept_type})
            MERGE (p)-[r:MENTIONS]->(c)
            SET r.confidence = m.confidence, r.synced_at = datetime()
        """, mentions=params)
        if (i + BATCH_SIZE) % 2000 == 0:
            logger.info(f"    Progress: {i + len(batch)}/{len(mentions)}")
    logger.info(f"  ✓ MENTIONS synced")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Connect to Postgres
    conn = psycopg2.connect(config.POSTGRES_DSN)

    # Get data
    logger.info("Fetching batch-v1 data from Postgres...")
    passages, concepts, mentions, documents = get_batch_v1_data(conn)
    conn.close()

    logger.info(f"Found: {len(passages)} passages, {len(concepts)} concepts, {len(mentions)} mentions, {len(documents)} documents")

    # Connect to Neo4j
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=("neo4j", config.NEO4J_PASSWORD)
    )

    # Sync
    sync_to_neo4j(driver, passages, concepts, mentions, documents, dry_run=args.dry_run)

    driver.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
