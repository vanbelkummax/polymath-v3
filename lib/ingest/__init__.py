"""
Ingestion pipeline for Polymath v3.

Handles PDF parsing, chunking, metadata resolution, and database storage.
"""

from .pipeline import IngestPipeline, IngestResult
from .pdf_parser import PDFParser, ParseResult
from .chunking import chunk_markdown_by_headers, chunk_plain_text, Chunk
from .metadata import MetadataResolver, PaperMetadata
from .doc_identity import get_doc_id, get_title_hash
from .concept_extractor import ConceptExtractor

__all__ = [
    "IngestPipeline",
    "IngestResult",
    "PDFParser",
    "ParseResult",
    "chunk_markdown_by_headers",
    "chunk_plain_text",
    "Chunk",
    "MetadataResolver",
    "PaperMetadata",
    "get_doc_id",
    "get_title_hash",
    "ConceptExtractor",
]
