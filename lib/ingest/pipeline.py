"""
Main ingestion pipeline for Polymath v3.

Orchestrates: PDF parsing → metadata resolution → chunking → embedding → storage.
Handles both single-file and batch ingestion with progress tracking.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from lib.config import config
from lib.db.postgres import get_pg_pool
from lib.embeddings.bge_m3 import get_embedder
from lib.ingest.chunking import Chunk, chunk_markdown_by_headers, chunk_plain_text
from lib.ingest.concept_extractor import ConceptExtractor, extract_concepts_from_passage
from lib.ingest.doc_identity import get_doc_id, get_passage_id
from lib.ingest.metadata import MetadataResolver, PaperMetadata
from lib.ingest.pdf_parser import PDFParser, ParseResult

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of ingesting a single document."""

    doc_id: Optional[uuid.UUID] = None
    title: str = ""
    passage_count: int = 0
    concept_count: int = 0
    success: bool = True
    error: Optional[str] = None
    metadata_source: str = "unknown"
    elapsed_seconds: float = 0.0


@dataclass
class BatchIngestResult:
    """Result of batch ingestion."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    results: list[IngestResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class IngestPipeline:
    """
    Main ingestion pipeline for Polymath v3.

    Usage:
        pipeline = IngestPipeline()

        # Single file
        result = pipeline.ingest_pdf("/path/to/paper.pdf")

        # Batch
        results = pipeline.ingest_batch(["/path/to/paper1.pdf", "/path/to/paper2.pdf"])
    """

    def __init__(
        self,
        extract_concepts: bool = True,
        compute_embeddings: bool = True,
        batch_name: Optional[str] = None,
        zotero_csv_path: Optional[Path] = None,
        soft_delete: bool = False,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            extract_concepts: Whether to extract concepts using Gemini
            compute_embeddings: Whether to compute embeddings
            batch_name: Name for tracking this batch
            zotero_csv_path: Path to Zotero CSV for metadata lookup
            soft_delete: If True, mark old passages as superseded instead of deleting.
                        This preserves manual annotations on passages.
                        Use for re-ingestion when you have user annotations to keep.
        """
        self.extract_concepts = extract_concepts
        self.compute_embeddings = compute_embeddings
        self.soft_delete = soft_delete
        self.batch_name = batch_name or f"ingest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Initialize components
        self.parser = PDFParser(strip_nul=True, detect_tables=True)
        self.metadata_resolver = MetadataResolver(zotero_csv_path or config.ZOTERO_CSV_PATH)
        self.embedder = get_embedder() if compute_embeddings else None
        self.concept_extractor = ConceptExtractor() if extract_concepts else None

    def ingest_pdf(self, pdf_path: Path) -> IngestResult:
        """
        Ingest a single PDF file.

        Steps:
        1. Parse PDF text
        2. Resolve metadata (Zotero → pdf2doi → CrossRef)
        3. Generate deterministic doc_id
        4. Chunk text
        5. Generate embeddings
        6. Extract concepts
        7. Store in database

        Args:
            pdf_path: Path to PDF file

        Returns:
            IngestResult with status and counts
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            return IngestResult(
                success=False,
                error=f"File not found: {pdf_path}",
            )

        try:
            # Step 1: Parse PDF
            logger.info(f"Parsing PDF: {pdf_path.name}")
            parse_result = self.parser.parse(pdf_path)

            if not parse_result.has_text:
                return IngestResult(
                    success=False,
                    error=f"No text extracted from PDF: {pdf_path.name}",
                )

            # Step 2: Resolve metadata
            logger.info("Resolving metadata...")
            metadata = self.metadata_resolver.resolve(pdf_path)

            # Step 3: Generate doc_id
            doc_id = get_doc_id(
                doi=metadata.doi,
                pmid=metadata.pmid,
                arxiv_id=metadata.arxiv_id,
                title=metadata.title,
            )

            # Step 4: Chunk text
            logger.info("Chunking text...")
            chunks = self._chunk_text(parse_result)

            if not chunks:
                return IngestResult(
                    doc_id=doc_id,
                    title=metadata.title,
                    success=False,
                    error="No chunks generated from text",
                    metadata_source=metadata.source_method,
                )

            # Step 5: Generate embeddings
            embeddings = []
            if self.compute_embeddings and self.embedder:
                logger.info(f"Computing embeddings for {len(chunks)} chunks...")
                texts = [chunk.content for chunk in chunks]
                embeddings = self.embedder.encode(texts)

            # Step 6: Extract concepts (if enabled)
            all_concepts = []
            if self.extract_concepts and self.concept_extractor:
                logger.info("Extracting concepts...")
                # Only extract from first N chunks to limit API calls
                for chunk in chunks[:20]:
                    concepts = extract_concepts_from_passage(
                        chunk.content,
                        self.concept_extractor,
                    )
                    all_concepts.extend(concepts)

            # Step 7: Store in database
            logger.info("Storing in database...")
            passage_count = self._store_document(
                doc_id=doc_id,
                metadata=metadata,
                pdf_path=pdf_path,
                chunks=chunks,
                embeddings=embeddings,
                concepts=all_concepts,
            )

            elapsed = time.time() - start_time
            logger.info(
                f"Ingested {pdf_path.name}: {passage_count} passages, "
                f"{len(all_concepts)} concepts in {elapsed:.1f}s"
            )

            return IngestResult(
                doc_id=doc_id,
                title=metadata.title,
                passage_count=passage_count,
                concept_count=len(all_concepts),
                success=True,
                metadata_source=metadata.source_method,
                elapsed_seconds=elapsed,
            )

        except (SystemExit, KeyboardInterrupt):
            # Always re-raise these - user wants to stop
            raise

        except Exception as e:
            # Check for critical database errors that should halt processing
            error_str = str(e).lower()
            if any(term in error_str for term in [
                "connection refused",
                "database is starting up",
                "too many connections",
                "out of memory",
                "disk full",
            ]):
                logger.critical(f"Critical database error, halting: {e}")
                raise

            logger.error(f"Ingestion failed for {pdf_path}: {e}")
            return IngestResult(
                success=False,
                error=str(e),
                elapsed_seconds=time.time() - start_time,
            )

    def _chunk_text(self, parse_result: ParseResult) -> list[Chunk]:
        """Chunk parsed text using appropriate strategy."""
        text = parse_result.text

        # Try markdown chunking first (works if text has ## headers)
        if "##" in text or "# " in text:
            chunks = chunk_markdown_by_headers(text)
            if chunks:
                return chunks

        # Fall back to plain text chunking
        return chunk_plain_text(text)

    def _store_document(
        self,
        doc_id: uuid.UUID,
        metadata: PaperMetadata,
        pdf_path: Path,
        chunks: list[Chunk],
        embeddings: list,
        concepts: list,
    ) -> int:
        """
        Store document and passages in database.

        Returns:
            Number of passages stored
        """
        pool = get_pg_pool()

        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Check if document exists (upsert)
                cur.execute(
                    """
                    INSERT INTO documents (
                        doc_id, title, title_hash, authors, year, venue,
                        doi, pmid, arxiv_id, zotero_key, abstract,
                        pdf_path, source_method, metadata_confidence, ingest_batch
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (doc_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        authors = EXCLUDED.authors,
                        year = EXCLUDED.year,
                        doi = COALESCE(EXCLUDED.doi, documents.doi),
                        pmid = COALESCE(EXCLUDED.pmid, documents.pmid),
                        arxiv_id = COALESCE(EXCLUDED.arxiv_id, documents.arxiv_id),
                        updated_at = NOW()
                    """,
                    (
                        str(doc_id),
                        metadata.title,
                        self._get_title_hash(metadata.title),
                        metadata.authors,
                        metadata.year,
                        metadata.venue,
                        metadata.doi,
                        metadata.pmid,
                        metadata.arxiv_id,
                        metadata.zotero_key,
                        metadata.abstract,
                        str(pdf_path),
                        metadata.source_method,
                        metadata.confidence,
                        self.batch_name,
                    ),
                )

                # Handle existing passages (for re-ingestion)
                if self.soft_delete:
                    # Soft delete: mark as superseded to preserve annotations
                    cur.execute(
                        """
                        UPDATE passages
                        SET is_superseded = TRUE,
                            superseded_at = NOW(),
                            superseded_by_batch = %s
                        WHERE doc_id = %s AND is_superseded = FALSE
                        """,
                        (self.batch_name, str(doc_id)),
                    )
                    logger.debug(f"Soft-deleted {cur.rowcount} existing passages")
                else:
                    # Hard delete: remove old passages (faster, but loses annotations)
                    cur.execute(
                        "DELETE FROM passages WHERE doc_id = %s",
                        (str(doc_id),),
                    )

                # Insert passages with embeddings
                passage_count = 0
                for i, chunk in enumerate(chunks):
                    passage_id = get_passage_id(doc_id, chunk.char_start, chunk.char_end)
                    embedding = embeddings[i] if (embeddings is not None and len(embeddings) > i) else None

                    cur.execute(
                        """
                        INSERT INTO passages (
                            passage_id, doc_id, passage_text, section, parent_section,
                            page_num, char_start, char_end, passage_index, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(passage_id),
                            str(doc_id),
                            chunk.content,
                            chunk.header,
                            chunk.parent_header,
                            chunk.page_num,
                            chunk.char_start,
                            chunk.char_end,
                            i,
                            embedding.tolist() if embedding is not None else None,
                        ),
                    )
                    passage_count += 1

                # Insert concepts
                if concepts:
                    # Get first passage ID for concept assignment
                    first_passage_id = get_passage_id(
                        doc_id, chunks[0].char_start, chunks[0].char_end
                    )

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
                                str(first_passage_id),
                                concept.name,
                                concept.type,
                                concept.confidence,
                                config.GEMINI_MODEL,
                                "v3.0",
                            ),
                        )

                conn.commit()

        return passage_count

    def _get_title_hash(self, title: str) -> str:
        """Generate title hash for deduplication."""
        from lib.ingest.doc_identity import get_title_hash
        return get_title_hash(title)

    def ingest_batch(
        self,
        pdf_paths: list[Path],
        max_workers: int = 4,
        progress_callback=None,
    ) -> BatchIngestResult:
        """
        Ingest multiple PDFs in parallel.

        Args:
            pdf_paths: List of PDF file paths
            max_workers: Number of parallel workers
            progress_callback: Optional callback(current, total, result)

        Returns:
            BatchIngestResult with aggregated stats
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        start_time = time.time()
        results = []
        succeeded = 0
        failed = 0
        processed = 0

        # Thread-safe counter for progress tracking
        lock = threading.Lock()

        # Track batch in database
        self._start_batch_tracking(len(pdf_paths))

        def process_pdf(pdf_path: Path) -> IngestResult:
            """Process a single PDF (called from thread pool)."""
            return self.ingest_pdf(pdf_path)

        # Use ThreadPoolExecutor for parallel processing
        # ThreadPool is appropriate here because work is I/O bound
        # (PDF parsing, API calls, database writes)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_path = {
                executor.submit(process_pdf, path): path
                for path in pdf_paths
            }

            # Process results as they complete
            for future in as_completed(future_to_path):
                pdf_path = future_to_path[future]

                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Worker exception for {pdf_path}: {e}")
                    result = IngestResult(
                        success=False,
                        error=f"Worker exception: {e}",
                    )

                # Thread-safe updates
                with lock:
                    results.append(result)
                    processed += 1

                    if result.success:
                        succeeded += 1
                    else:
                        failed += 1

                    # Update batch progress
                    self._update_batch_progress(processed, succeeded, failed)

                    if progress_callback:
                        progress_callback(processed, len(pdf_paths), result)

        elapsed = time.time() - start_time
        self._complete_batch_tracking(succeeded, failed)

        return BatchIngestResult(
            total=len(pdf_paths),
            succeeded=succeeded,
            failed=failed,
            results=results,
            elapsed_seconds=elapsed,
        )

    def _start_batch_tracking(self, total: int):
        """Record batch start in database."""
        pool = get_pg_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ingest_batches (batch_name, total_files, status)
                    VALUES (%s, %s, 'running')
                    ON CONFLICT (batch_name) DO UPDATE SET
                        total_files = EXCLUDED.total_files,
                        status = 'running',
                        started_at = NOW()
                    """,
                    (self.batch_name, total),
                )
                conn.commit()

    def _update_batch_progress(self, processed: int, succeeded: int, failed: int):
        """Update batch progress in database."""
        pool = get_pg_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE ingest_batches
                    SET processed_files = %s,
                        succeeded_files = %s,
                        failed_files = %s
                    WHERE batch_name = %s
                    """,
                    (processed, succeeded, failed, self.batch_name),
                )
                conn.commit()

    def _complete_batch_tracking(self, succeeded: int, failed: int):
        """Mark batch as complete in database."""
        pool = get_pg_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE ingest_batches
                    SET status = 'completed',
                        succeeded_files = %s,
                        failed_files = %s,
                        completed_at = NOW()
                    WHERE batch_name = %s
                    """,
                    (succeeded, failed, self.batch_name),
                )
                conn.commit()


def ingest_single_pdf(pdf_path: Path, **kwargs) -> IngestResult:
    """
    Convenience function to ingest a single PDF.

    Args:
        pdf_path: Path to PDF file
        **kwargs: Arguments passed to IngestPipeline

    Returns:
        IngestResult
    """
    pipeline = IngestPipeline(**kwargs)
    return pipeline.ingest_pdf(pdf_path)
