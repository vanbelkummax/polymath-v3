#!/usr/bin/env python3
"""
Backfill concept extraction for passages without concepts.

Uses Gemini API for batch extraction with rate limiting.

Usage:
    python scripts/backfill_concepts.py --workers 8
    python scripts/backfill_concepts.py --worker-id 0 --num-workers 8
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
from lib.ingest.concept_extractor import ConceptExtractor, CONCEPT_TYPE_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_passages_without_concepts(
    limit: int = 1000,
    offset: int = 0,
    worker_id: int = 0,
    num_workers: int = 1,
) -> list[dict]:
    """Get passages that don't have concepts extracted yet."""
    pool = get_pg_pool()

    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Use modulo for worker partitioning
            cur.execute(
                """
                SELECT p.passage_id, p.passage_text, p.doc_id, d.title
                FROM passages p
                JOIN documents d ON p.doc_id = d.doc_id
                WHERE NOT EXISTS (
                    SELECT 1 FROM passage_concepts pc
                    WHERE pc.passage_id = p.passage_id
                )
                AND MOD(ABS(hashtext(p.passage_id::text)), %s) = %s
                ORDER BY p.created_at
                LIMIT %s OFFSET %s
                """,
                (num_workers, worker_id, limit, offset),
            )

            return [
                {
                    "passage_id": row["passage_id"],
                    "passage_text": row["passage_text"],
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                }
                for row in cur.fetchall()
            ]


def save_concepts(passage_id: str, concepts: list, extractor_model: str):
    """Save extracted concepts to database."""
    if not concepts:
        return

    pool = get_pg_pool()

    with pool.connection() as conn:
        with conn.cursor() as cur:
            for concept in concepts:
                cur.execute(
                    """
                    INSERT INTO passage_concepts (
                        passage_id, concept_name, concept_type, confidence,
                        extractor_model, extractor_version
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (passage_id, concept_name) DO NOTHING
                    """,
                    (
                        passage_id,
                        concept.name,
                        concept.type,
                        concept.confidence,
                        extractor_model,
                        "v3.0",
                    ),
                )

            conn.commit()


def update_job_progress(job_name: str, processed: int, failed: int, status: str = "running"):
    """Update job progress in kb_migrations."""
    pool = get_pg_pool()

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO kb_migrations (job_name, status, items_processed, items_failed)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (job_name) DO UPDATE SET
                    status = EXCLUDED.status,
                    items_processed = EXCLUDED.items_processed,
                    items_failed = EXCLUDED.items_failed,
                    updated_at = NOW()
                """,
                (job_name, status, processed, failed),
            )
            conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Backfill concept extraction")
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Worker ID for partitioning",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total number of workers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of passages per batch",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls (seconds)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum passages to process",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Gemini model to use",
    )

    args = parser.parse_args()

    job_name = f"backfill_concepts_w{args.worker_id}"
    logger.info(f"Starting {job_name} (worker {args.worker_id + 1}/{args.num_workers})")

    # Initialize extractor
    extractor = ConceptExtractor(model=args.model)

    # Track stats
    processed = 0
    failed = 0
    total_concepts = 0
    start_time = time.time()

    update_job_progress(job_name, processed, failed, "running")

    try:
        offset = 0
        while True:
            # Get batch of passages
            passages = get_passages_without_concepts(
                limit=args.batch_size,
                offset=offset,
                worker_id=args.worker_id,
                num_workers=args.num_workers,
            )

            if not passages:
                logger.info("No more passages to process")
                break

            logger.info(f"Processing batch of {len(passages)} passages...")

            for passage in passages:
                try:
                    # Extract concepts
                    result = extractor.extract(passage["passage_text"])

                    if result.success and result.concepts:
                        save_concepts(
                            passage["passage_id"],
                            result.concepts,
                            args.model,
                        )
                        total_concepts += len(result.concepts)

                    processed += 1

                except Exception as e:
                    logger.error(f"Failed on {passage['passage_id']}: {e}")
                    failed += 1

                # Rate limiting
                time.sleep(args.delay)

                # Progress logging
                if processed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed * 3600
                    logger.info(
                        f"Progress: {processed} processed, {failed} failed, "
                        f"{total_concepts} concepts, {rate:.0f}/hour"
                    )
                    update_job_progress(job_name, processed, failed)

            # Check limit
            if args.limit and processed >= args.limit:
                logger.info(f"Reached limit of {args.limit}")
                break

            offset += args.batch_size

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        update_job_progress(job_name, processed, failed, "interrupted")
        sys.exit(1)

    # Final update
    update_job_progress(job_name, processed, failed, "completed")

    elapsed = time.time() - start_time
    logger.info(f"\nCompleted in {elapsed/60:.1f} minutes")
    logger.info(f"Processed: {processed}, Failed: {failed}, Concepts: {total_concepts}")


if __name__ == "__main__":
    main()
