#!/usr/bin/env python3
"""
Re-ingest PDF archive into Polymath v3.

Processes all PDFs from the archive directory, resolves metadata
via Zotero CSV, and stores in pgvector database.

Usage:
    python scripts/reingest_archive.py --archive-dir /scratch/polymath_archive
    python scripts/reingest_archive.py --resume  # Continue from last checkpoint
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config
from lib.db.postgres import get_pg_pool
from lib.ingest.pipeline import IngestPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_ARCHIVE_DIR = Path("/scratch/polymath_archive")
CHECKPOINT_FILE = Path("reingest_checkpoint.json")


def get_all_pdfs(archive_dir: Path) -> list[Path]:
    """Get all PDF files from archive directory."""
    return sorted(archive_dir.glob("**/*.pdf"))


def load_checkpoint() -> dict:
    """Load checkpoint from file."""
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {
        "processed": [],
        "failed": [],
        "last_file": None,
        "started_at": None,
    }


def save_checkpoint(checkpoint: dict):
    """Save checkpoint to file."""
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2, default=str))


def update_job_status(batch_name: str, status: str, processed: int, failed: int):
    """Update job status in kb_migrations table."""
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
                (batch_name, status, processed, failed),
            )
            conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Re-ingest PDF archive")
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help="Directory containing PDFs",
    )
    parser.add_argument(
        "--batch-name",
        default=f"reingest_{datetime.now().strftime('%Y%m%d')}",
        help="Batch name for tracking",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (not yet implemented)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of PDFs to process",
    )
    parser.add_argument(
        "--no-concepts",
        action="store_true",
        help="Skip concept extraction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without processing",
    )

    args = parser.parse_args()

    # Get PDF files
    logger.info(f"Scanning {args.archive_dir} for PDFs...")
    all_pdfs = get_all_pdfs(args.archive_dir)
    logger.info(f"Found {len(all_pdfs)} PDF files")

    if args.dry_run:
        for pdf in all_pdfs[:50]:
            print(pdf)
        if len(all_pdfs) > 50:
            print(f"... and {len(all_pdfs) - 50} more")
        return

    # Load checkpoint if resuming
    checkpoint = load_checkpoint() if args.resume else {
        "processed": [],
        "failed": [],
        "last_file": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    # Filter out already processed files
    processed_set = set(checkpoint["processed"])
    pdfs_to_process = [p for p in all_pdfs if str(p) not in processed_set]

    if args.limit:
        pdfs_to_process = pdfs_to_process[:args.limit]

    logger.info(f"Processing {len(pdfs_to_process)} PDFs (skipping {len(processed_set)} already done)")

    # Initialize pipeline
    pipeline = IngestPipeline(
        extract_concepts=not args.no_concepts,
        batch_name=args.batch_name,
    )

    # Track stats
    start_time = time.time()
    succeeded = len(checkpoint["processed"])
    failed = len(checkpoint["failed"])

    # Update job status
    update_job_status(args.batch_name, "running", succeeded, failed)

    try:
        for i, pdf_path in enumerate(pdfs_to_process):
            logger.info(f"[{i+1}/{len(pdfs_to_process)}] Processing: {pdf_path.name}")

            result = pipeline.ingest_pdf(pdf_path)

            if result.success:
                succeeded += 1
                checkpoint["processed"].append(str(pdf_path))
                logger.info(f"  ✓ {result.title} ({result.passage_count} passages)")
            else:
                failed += 1
                checkpoint["failed"].append(str(pdf_path))
                logger.warning(f"  ✗ Failed: {result.error}")

            checkpoint["last_file"] = str(pdf_path)

            # Save checkpoint every 10 files
            if (i + 1) % 10 == 0:
                save_checkpoint(checkpoint)
                update_job_status(args.batch_name, "running", succeeded, failed)

                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 3600
                logger.info(f"  Progress: {succeeded} succeeded, {failed} failed, {rate:.0f}/hour")

    except KeyboardInterrupt:
        logger.info("Interrupted - saving checkpoint...")
        save_checkpoint(checkpoint)
        update_job_status(args.batch_name, "interrupted", succeeded, failed)
        sys.exit(1)

    # Final checkpoint and status
    save_checkpoint(checkpoint)
    update_job_status(args.batch_name, "completed", succeeded, failed)

    elapsed = time.time() - start_time
    logger.info(f"\nCompleted in {elapsed/3600:.1f} hours")
    logger.info(f"Succeeded: {succeeded}, Failed: {failed}")


if __name__ == "__main__":
    main()
