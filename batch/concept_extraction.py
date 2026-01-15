#!/usr/bin/env python3
"""
GCP Batch job for concept extraction at scale.

Processes passages from GCS input file and writes extracted concepts to output.

Usage (within GCP Batch):
    python -m batch.concept_extraction \
        --input-uri gs://bucket/passages.jsonl \
        --output-uri gs://bucket/concepts.jsonl
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add parent to path when running as module
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config
from lib.ingest.concept_extractor import ConceptExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ConceptExtractionJob:
    """
    Batch job for concept extraction.

    Input format (JSONL):
        {"passage_id": "uuid", "text": "passage text..."}

    Output format (JSONL):
        {"passage_id": "uuid", "concepts": [{"name": "...", "type": "...", "confidence": 0.8}]}
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        batch_size: int = 100,
    ):
        """
        Initialize job.

        Args:
            model: Gemini model to use
            batch_size: Number of passages to process at once
        """
        self.model = model
        self.batch_size = batch_size
        self.extractor = ConceptExtractor(model=model)

    def run(self, input_uri: str, output_uri: str):
        """
        Run the extraction job.

        Implements proper sharding using BATCH_TASK_INDEX to ensure each worker
        processes only its slice of the data (avoids 10x cost overrun).

        Args:
            input_uri: GCS URI for input JSONL
            output_uri: GCS URI for output JSONL
        """
        from google.cloud import storage

        # =================================================================
        # GCP Batch Sharding: Each worker processes only its slice
        # =================================================================
        task_index = int(os.environ.get("BATCH_TASK_INDEX", 0))
        task_count = int(os.environ.get("BATCH_TASK_COUNT", 1))
        logger.info(f"Worker {task_index + 1}/{task_count} starting")

        # Download input
        logger.info(f"Downloading input from {input_uri}")
        input_path = self._download_gcs(input_uri)

        # Read all lines and calculate shard slice
        with open(input_path, "r") as f:
            all_lines = f.readlines()

        total_records = len(all_lines)
        chunk_size = total_records // task_count
        start_idx = task_index * chunk_size
        # Last worker gets any remainder
        end_idx = start_idx + chunk_size if task_index < task_count - 1 else total_records

        my_lines = all_lines[start_idx:end_idx]
        logger.info(
            f"Worker {task_index}: processing records {start_idx}-{end_idx} "
            f"({len(my_lines)} of {total_records} total)"
        )

        # Process only our shard
        output_path = tempfile.mktemp(suffix=".jsonl")
        processed = 0
        failed = 0

        with open(output_path, "w") as outfile:
            batch = []

            for line in my_lines:
                record = json.loads(line)
                batch.append(record)

                if len(batch) >= self.batch_size:
                    results = self._process_batch(batch)
                    for result in results:
                        outfile.write(json.dumps(result) + "\n")
                        if result.get("concepts"):
                            processed += 1
                        else:
                            failed += 1

                    batch = []
                    logger.info(f"Worker {task_index}: processed {processed + failed} passages")

            # Process remaining
            if batch:
                results = self._process_batch(batch)
                for result in results:
                    outfile.write(json.dumps(result) + "\n")
                    if result.get("concepts"):
                        processed += 1
                    else:
                        failed += 1

        # Upload output with worker-specific suffix for parallel writes
        # Each worker writes to a separate file to avoid overwrites
        if task_count > 1:
            # Split output into per-worker files: concepts_0.jsonl, concepts_1.jsonl, etc.
            base_uri = output_uri.rsplit(".", 1)[0]
            worker_output_uri = f"{base_uri}_{task_index}.jsonl"
        else:
            worker_output_uri = output_uri

        logger.info(f"Uploading output to {worker_output_uri}")
        self._upload_gcs(output_path, worker_output_uri)

        logger.info(f"Worker {task_index} completed: {processed} processed, {failed} failed")

        # Cleanup
        os.remove(input_path)
        os.remove(output_path)

    def _process_batch(self, batch: list[dict]) -> list[dict]:
        """Process a batch of passages."""
        results = []

        for record in batch:
            passage_id = record.get("passage_id")
            text = record.get("text", "")

            try:
                extraction = self.extractor.extract(text)

                concepts = []
                if extraction.success:
                    for concept in extraction.concepts:
                        concepts.append({
                            "name": concept.name,
                            "type": concept.type,
                            "confidence": concept.confidence,
                        })

                results.append({
                    "passage_id": passage_id,
                    "concepts": concepts,
                    "model": self.model,
                })

            except Exception as e:
                logger.error(f"Failed on {passage_id}: {e}")
                results.append({
                    "passage_id": passage_id,
                    "concepts": [],
                    "error": str(e),
                })

        return results

    def _download_gcs(self, uri: str) -> str:
        """Download file from GCS."""
        from google.cloud import storage

        client = storage.Client()
        bucket_name, blob_name = self._parse_gcs_uri(uri)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        local_path = tempfile.mktemp(suffix=".jsonl")
        blob.download_to_filename(local_path)

        return local_path

    def _upload_gcs(self, local_path: str, uri: str):
        """Upload file to GCS."""
        from google.cloud import storage

        client = storage.Client()
        bucket_name, blob_name = self._parse_gcs_uri(uri)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.upload_from_filename(local_path)

    def _parse_gcs_uri(self, uri: str) -> tuple[str, str]:
        """Parse gs:// URI into bucket and blob."""
        # gs://bucket/path/to/file
        uri = uri.replace("gs://", "")
        parts = uri.split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""


def prepare_input_file(output_path: str, limit: int = None):
    """
    Prepare input file from database passages.

    Args:
        output_path: Path to write JSONL file
        limit: Maximum passages to export
    """
    from lib.db.postgres import get_pg_pool

    pool = get_pg_pool()

    with pool.connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT p.passage_id, p.passage_text
                FROM passages p
                WHERE NOT EXISTS (
                    SELECT 1 FROM passage_concepts pc
                    WHERE pc.passage_id = p.passage_id
                )
                ORDER BY p.created_at
            """
            if limit:
                query += f" LIMIT {limit}"

            cur.execute(query)

            with open(output_path, "w") as f:
                count = 0
                for row in cur:
                    f.write(
                        json.dumps({
                            "passage_id": str(row["passage_id"]),
                            "text": row["passage_text"],
                        })
                        + "\n"
                    )
                    count += 1

    logger.info(f"Exported {count} passages to {output_path}")


def import_output_file(input_path: str):
    """
    Import concept extraction output back to database.

    Args:
        input_path: Path to output JSONL file
    """
    from lib.db.postgres import get_pg_pool

    pool = get_pg_pool()
    imported = 0

    with open(input_path, "r") as f:
        for line in f:
            record = json.loads(line)
            passage_id = record.get("passage_id")
            concepts = record.get("concepts", [])
            model = record.get("model", "unknown")

            if not concepts:
                continue

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
                                concept["name"],
                                concept["type"],
                                concept["confidence"],
                                model,
                                "v3.0-batch",
                            ),
                        )

                    conn.commit()
                    imported += 1

    logger.info(f"Imported concepts for {imported} passages")


def main():
    parser = argparse.ArgumentParser(description="Concept extraction batch job")
    parser.add_argument("--input-uri", required=True, help="GCS input URI")
    parser.add_argument("--output-uri", required=True, help="GCS output URI")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model to use")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")

    args = parser.parse_args()

    job = ConceptExtractionJob(model=args.model, batch_size=args.batch_size)
    job.run(args.input_uri, args.output_uri)


if __name__ == "__main__":
    main()
