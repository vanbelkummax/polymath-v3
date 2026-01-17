#!/usr/bin/env python3
"""
Backfill embeddings for passages that have batch-v1 concepts but no embeddings.

Usage:
    python scripts/backfill_embeddings_batch.py --dry-run
    python scripts/backfill_embeddings_batch.py
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import numpy as np
from lib.config import config
from lib.embeddings.bge_m3 import Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 32  # GPU batch size


def get_passages_needing_embeddings(conn):
    """Get passages with batch-v1 concepts but no embeddings."""
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT p.passage_id, p.passage_text
        FROM passages p
        JOIN passage_concepts pc ON p.passage_id = pc.passage_id
        WHERE pc.extractor_version = 'batch-v1'
        AND p.embedding IS NULL
        ORDER BY p.passage_id
    """)
    results = cur.fetchall()
    cur.close()
    return results


def update_embedding(conn, passage_id, embedding):
    """Update passage with embedding."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE passages
        SET embedding = %s, embedding_model = 'bge-m3'
        WHERE passage_id = %s
    """, (embedding.tolist(), passage_id))
    conn.commit()
    cur.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    args = parser.parse_args()

    conn = psycopg2.connect(config.POSTGRES_DSN)

    passages = get_passages_needing_embeddings(conn)
    logger.info(f"Found {len(passages)} passages needing embeddings")

    if args.dry_run:
        logger.info("[DRY RUN] Would generate embeddings for these passages")
        conn.close()
        return

    if not passages:
        logger.info("No passages need embeddings")
        conn.close()
        return

    # Initialize embedder
    embedder = Embedder()

    # Process in batches
    for i in range(0, len(passages), BATCH_SIZE):
        batch = passages[i:i + BATCH_SIZE]
        texts = [p[1] for p in batch]
        ids = [p[0] for p in batch]

        logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(passages) + BATCH_SIZE - 1)//BATCH_SIZE}")

        # Generate embeddings
        embeddings = embedder.encode(texts)

        # Store each embedding
        for j, (passage_id, embedding) in enumerate(zip(ids, embeddings)):
            update_embedding(conn, passage_id, embedding)

        logger.info(f"  Stored {len(batch)} embeddings")

    conn.close()
    logger.info(f"Done! Generated embeddings for {len(passages)} passages")


if __name__ == "__main__":
    main()
