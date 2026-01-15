"""
Pytest configuration and fixtures for Polymath v3 tests.
"""

import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Session-scoped fixtures (shared across all tests)
# =============================================================================


@pytest.fixture(scope="session")
def searcher():
    """Shared HybridSearcher instance."""
    from lib.search.hybrid_search import HybridSearcher
    return HybridSearcher()


@pytest.fixture(scope="session")
def gap_detector():
    """Shared GapDetector instance."""
    from lib.bridgemine.gap_detection import GapDetector
    return GapDetector()


# =============================================================================
# Sample data fixtures
# =============================================================================


@pytest.fixture
def sample_document():
    """Create a sample document dict for testing."""
    import uuid
    return {
        "doc_id": str(uuid.uuid4()),
        "title": "Spatial Transcriptomics Methods for Tissue Analysis",
        "authors": ["Smith, J.", "Doe, A."],
        "year": 2024,
        "doi": "10.1234/test.2024.001",
        "pmid": "12345678",
        "arxiv_id": None,
        "venue": "Nature Methods",
        "abstract": "We present novel methods for spatial transcriptomics analysis.",
    }


@pytest.fixture
def sample_passage_text():
    """Sample passage text for testing."""
    return """
    Spatial transcriptomics has emerged as a powerful technique for studying
    gene expression in tissue context. Unlike single-cell RNA sequencing,
    which loses spatial information during dissociation, spatial methods
    preserve the tissue architecture. Here we describe a novel approach
    using transformer-based deep learning models to predict gene expression
    from H&E stained histology images. Our method achieves state-of-the-art
    performance on the Visium HD benchmark dataset.
    """


@pytest.fixture
def sample_markdown_text():
    """Sample markdown text for chunking tests."""
    return """# Abstract

This paper presents a novel approach to spatial transcriptomics analysis.
We develop new computational methods that leverage deep learning to predict
gene expression patterns from histology images. Our approach achieves
state-of-the-art performance on multiple benchmark datasets.

# Introduction

Spatial transcriptomics has revolutionized our understanding of tissue biology.
Traditional bulk RNA sequencing loses spatial context, while single-cell methods
require tissue dissociation. Spatial methods preserve tissue architecture while
measuring gene expression. Here we describe a novel deep learning approach.

## Background

The field of spatial transcriptomics has grown rapidly since the introduction
of methods like 10x Visium and Slide-seq. These technologies enable researchers
to measure gene expression while preserving spatial information about cell
locations within tissue sections.

# Methods

We developed a transformer-based model for predicting gene expression from
H&E stained histology images. The model was trained on paired Visium and
histology data from multiple tissue types.

## Data Collection

We collected data from 50 tissue samples spanning 5 tissue types.
Each sample included matched Visium spatial transcriptomics and H&E imaging.
Quality control was performed using standard metrics.

## Model Architecture

The model uses a Vision Transformer backbone with specialized attention
mechanisms for capturing spatial relationships between tissue regions.
We implement a multi-scale approach to capture both local and global patterns.
"""


@pytest.fixture
def sample_chunks():
    """Create sample Chunk objects for testing."""
    from lib.ingest.chunking import Chunk
    return [
        Chunk(
            content="This is the introduction section with background context.",
            header="Introduction",
            parent_header=None,
            level=1,
            char_start=0,
            char_end=100,
        ),
        Chunk(
            content="This section describes the methods used in our analysis.",
            header="Methods",
            parent_header=None,
            level=1,
            char_start=101,
            char_end=200,
        ),
        Chunk(
            content="Detailed data collection procedures are described here.",
            header="Data Collection",
            parent_header="Methods",
            level=2,
            char_start=201,
            char_end=300,
        ),
    ]


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a temporary PDF file for testing."""
    pdf_path = tmp_path / "test_paper.pdf"
    # Create minimal valid PDF content
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000206 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
300
%%EOF
"""
    pdf_path.write_bytes(pdf_content)
    return pdf_path


# =============================================================================
# Mock fixtures
# =============================================================================


@pytest.fixture
def mock_pg_pool():
    """Mock PostgreSQL connection pool."""
    with patch("lib.db.postgres.get_pg_pool") as mock:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock.return_value = mock_pool
        yield mock_cursor


@pytest.fixture
def mock_embedder():
    """Mock embedding model."""
    with patch("lib.embeddings.bge_m3.get_embedder") as mock:
        mock_embedder = MagicMock()
        import numpy as np
        mock_embedder.encode.return_value = np.random.randn(1, 1024).astype(np.float32)
        mock.return_value = mock_embedder
        yield mock_embedder


# =============================================================================
# Pytest markers
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require DB)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )
