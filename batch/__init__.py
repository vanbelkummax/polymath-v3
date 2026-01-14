"""
GCP Batch API modules for Polymath v3.

Provides distributed processing for:
- Concept extraction at scale
- Embedding generation
- Large-scale ingestion
"""

from .job_manager import BatchJobManager, JobStatus
from .concept_extraction import ConceptExtractionJob

__all__ = [
    "BatchJobManager",
    "JobStatus",
    "ConceptExtractionJob",
]
