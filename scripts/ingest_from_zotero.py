#!/usr/bin/env python3
"""
Ingest PDFs from Zotero CSV export.

Reads Zotero CSV, extracts PDF paths, and runs through IngestPipeline.
Converts Windows paths to WSL paths automatically.

Usage:
    # Prototype with 200 PDFs
    python scripts/ingest_from_zotero.py --csv /path/to/zotero.csv --limit 200

    # Full ingestion (skip concepts for GCP batch later)
    python scripts/ingest_from_zotero.py --csv /path/to/zotero.csv --skip-concepts

    # Multiple CSVs
    python scripts/ingest_from_zotero.py \
        --csv '/mnt/c/Users/User/Downloads/Polymath_Full_.csv' \
        --csv '/mnt/c/Users/User/Downloads/polymath2_.csv' \
        --limit 200
"""

import argparse
import csv
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ZoteroEntry:
    """Parsed Zotero CSV entry."""
    key: str
    title: str
    authors: Optional[str]
    year: Optional[int]
    doi: Optional[str]
    abstract: Optional[str]
    pdf_path: Optional[Path]


def windows_to_wsl_path(windows_path: str) -> Optional[Path]:
    """
    Convert Windows path to WSL path.

    C:\\Users\\User\\file.pdf -> /mnt/c/Users/User/file.pdf
    """
    if not windows_path:
        return None

    # Handle C:\ style paths
    match = re.match(r'^([A-Za-z]):\\(.+)$', windows_path)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace('\\', '/')
        wsl_path = Path(f"/mnt/{drive}/{rest}")
        return wsl_path

    # Already a Unix path?
    if windows_path.startswith('/'):
        return Path(windows_path)

    return None


def extract_pdf_path(file_attachments: str) -> Optional[Path]:
    """
    Extract PDF path from Zotero File Attachments field.

    The field may contain multiple files; we want the .pdf one.
    """
    if not file_attachments:
        return None

    # Split by semicolon (Zotero separator for multiple files)
    for attachment in file_attachments.split(';'):
        attachment = attachment.strip()
        if attachment.lower().endswith('.pdf'):
            return windows_to_wsl_path(attachment)

    return None


def parse_zotero_csv(csv_path: Path) -> list[ZoteroEntry]:
    """
    Parse Zotero CSV export.

    Expected columns: Key, Title, Author, Publication Year, DOI, Abstract Note, File Attachments
    """
    entries = []

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Extract PDF path
            pdf_path = extract_pdf_path(row.get('File Attachments', ''))

            # Skip entries without PDFs
            if not pdf_path:
                continue

            # Parse year
            year = None
            year_str = row.get('Publication Year', '').strip()
            if year_str and year_str.isdigit():
                year = int(year_str)

            entries.append(ZoteroEntry(
                key=row.get('Key', ''),
                title=row.get('Title', ''),
                authors=row.get('Author'),
                year=year,
                doi=row.get('DOI'),
                abstract=row.get('Abstract Note'),
                pdf_path=pdf_path,
            ))

    return entries


def ingest_entries(
    entries: list[ZoteroEntry],
    limit: Optional[int] = None,
    workers: int = 4,
    skip_concepts: bool = False,
    skip_embeddings: bool = False,
) -> dict:
    """
    Ingest Zotero entries through IngestPipeline.

    Args:
        entries: List of ZoteroEntry objects
        limit: Maximum entries to process
        workers: Number of parallel workers
        skip_concepts: Skip concept extraction (for GCP batch later)
        skip_embeddings: Skip embedding generation

    Returns:
        Summary dict with counts
    """
    # Lazy import to allow dry-run without database
    from lib.ingest.pipeline import IngestPipeline

    if limit:
        entries = entries[:limit]

    logger.info(f"Processing {len(entries)} entries (workers={workers})")

    # Initialize pipeline
    pipeline = IngestPipeline(
        extract_concepts=not skip_concepts,
        compute_embeddings=not skip_embeddings,
        batch_name=f"zotero_ingest_{int(time.time())}",
    )

    # Track results
    succeeded = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    for i, entry in enumerate(entries):
        # Check if PDF exists
        if not entry.pdf_path or not entry.pdf_path.exists():
            logger.warning(f"[{i+1}/{len(entries)}] PDF not found: {entry.pdf_path}")
            skipped += 1
            continue

        logger.info(f"[{i+1}/{len(entries)}] {entry.title[:50]}...")

        try:
            result = pipeline.ingest_pdf(entry.pdf_path)

            if result.success:
                succeeded += 1
                logger.info(f"  ✓ {result.passage_count} passages")
            else:
                failed += 1
                logger.error(f"  ✗ {result.error}")

        except Exception as e:
            failed += 1
            logger.error(f"  ✗ Exception: {e}")

    elapsed = time.time() - start_time

    summary = {
        "total": len(entries),
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "elapsed_seconds": elapsed,
        "rate_per_hour": (succeeded / elapsed * 3600) if elapsed > 0 else 0,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total entries:  {summary['total']}")
    print(f"Succeeded:      {summary['succeeded']}")
    print(f"Failed:         {summary['failed']}")
    print(f"Skipped (no PDF): {summary['skipped']}")
    print(f"Time:           {elapsed:.1f}s")
    print(f"Rate:           {summary['rate_per_hour']:.0f} docs/hour")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs from Zotero CSV")
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        required=True,
        help="Path to Zotero CSV (can specify multiple)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of entries to process",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--skip-concepts",
        action="store_true",
        help="Skip concept extraction (use GCP batch later)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse CSV and show stats without ingesting",
    )

    args = parser.parse_args()

    # Parse all CSVs
    all_entries = []
    for csv_path in args.csv:
        logger.info(f"Parsing {csv_path}...")
        entries = parse_zotero_csv(csv_path)
        logger.info(f"  Found {len(entries)} entries with PDFs")
        all_entries.extend(entries)

    logger.info(f"Total entries: {len(all_entries)}")

    # Check how many PDFs exist
    existing = sum(1 for e in all_entries if e.pdf_path and e.pdf_path.exists())
    logger.info(f"PDFs found on disk: {existing}/{len(all_entries)}")

    if args.dry_run:
        print("\nDry run - not ingesting")
        print(f"Would process {min(args.limit or len(all_entries), len(all_entries))} entries")
        return

    # Ingest
    ingest_entries(
        all_entries,
        limit=args.limit,
        workers=args.workers,
        skip_concepts=args.skip_concepts,
        skip_embeddings=args.skip_embeddings,
    )


if __name__ == "__main__":
    main()
