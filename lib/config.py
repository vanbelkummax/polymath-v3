"""
Centralized configuration for Polymath v3.

All configuration values should be imported from this module.
Supports environment variable overrides for containerization.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml or .git."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current.parent


@dataclass
class Config:
    """Polymath v3 configuration."""

    # ==========================================================================
    # Paths
    # ==========================================================================
    PROJECT_ROOT: Path = field(default_factory=_find_project_root)

    @property
    def STAGING_DIR(self) -> Path:
        return Path(os.environ.get("STAGING_DIR", str(self.PROJECT_ROOT / "ingest_staging")))

    @property
    def PDF_ARCHIVE_PATH(self) -> Path:
        return Path(os.environ.get("PDF_ARCHIVE_PATH", "/scratch/polymath_archive/"))

    @property
    def ZOTERO_CSV_PATH(self) -> Optional[Path]:
        path = os.environ.get("ZOTERO_CSV_PATH")
        return Path(path) if path else None

    # ==========================================================================
    # Database Connections
    # ==========================================================================
    @property
    def POSTGRES_DSN(self) -> str:
        return os.environ.get(
            "POSTGRES_DSN",
            "dbname=polymath_v3 user=polymath host=/var/run/postgresql"
        )

    @property
    def NEO4J_URI(self) -> str:
        return os.environ.get("NEO4J_URI", "bolt://localhost:7687")

    @property
    def NEO4J_USER(self) -> str:
        return os.environ.get("NEO4J_USER", "neo4j")

    @property
    def NEO4J_PASSWORD(self) -> str:
        return os.environ.get("NEO4J_PASSWORD", "")

    # ==========================================================================
    # Embedding Models
    # ==========================================================================
    @property
    def EMBEDDING_MODEL(self) -> str:
        return os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

    @property
    def EMBEDDING_DIM(self) -> int:
        return int(os.environ.get("EMBEDDING_DIM", "1024"))

    @property
    def RERANKER_MODEL(self) -> str:
        return os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    # ==========================================================================
    # Google Cloud Platform
    # ==========================================================================
    @property
    def GCP_PROJECT_ID(self) -> Optional[str]:
        return os.environ.get("GCP_PROJECT_ID")

    @property
    def GCP_SERVICE_ACCOUNT(self) -> Optional[str]:
        return os.environ.get("GCP_SERVICE_ACCOUNT")

    @property
    def GCP_BUCKET(self) -> str:
        return os.environ.get("GCP_BUCKET", "polymath-batch-jobs")

    @property
    def GEMINI_API_KEY(self) -> Optional[str]:
        return os.environ.get("GEMINI_API_KEY")

    # ==========================================================================
    # External APIs
    # ==========================================================================
    @property
    def OPENALEX_EMAIL(self) -> str:
        return os.environ.get("OPENALEX_EMAIL", "polymath@example.com")

    @property
    def BRAVE_API_KEY(self) -> Optional[str]:
        return os.environ.get("BRAVE_API_KEY")

    # ==========================================================================
    # Processing
    # ==========================================================================
    @property
    def BATCH_SIZE(self) -> int:
        return int(os.environ.get("BATCH_SIZE", "100"))

    @property
    def NUM_WORKERS(self) -> int:
        return int(os.environ.get("NUM_WORKERS", "8"))

    @property
    def LOG_LEVEL(self) -> str:
        return os.environ.get("LOG_LEVEL", "INFO")

    # ==========================================================================
    # Rate Limits (requests per second)
    # ==========================================================================
    RATE_LIMITS = {
        "crossref": 10.0,
        "openalex": 10.0,
        "arxiv": 0.33,
        "semanticscholar": 1.0,
        "pubmed": 3.0,
        "github": 0.5,
    }

    # ==========================================================================
    # Concept Extraction
    # ==========================================================================
    CONCEPT_TYPES = (
        "method",
        "problem",
        "domain",
        "dataset",
        "metric",
        "entity",
        "mechanism",
        "data_structure",
    )

    # ==========================================================================
    # Validation
    # ==========================================================================
    def validate(self) -> list[str]:
        """Return list of configuration errors."""
        errors = []

        # Check required paths
        if not self.PROJECT_ROOT.exists():
            errors.append(f"PROJECT_ROOT does not exist: {self.PROJECT_ROOT}")

        # Check database credentials
        if not self.NEO4J_PASSWORD:
            errors.append("NEO4J_PASSWORD not set")

        # Check embedding model
        if self.EMBEDDING_MODEL != "BAAI/bge-m3":
            errors.append(
                f"Non-standard embedding model: {self.EMBEDDING_MODEL}. "
                "Ensure dimension compatibility."
            )

        return errors

    def ensure_dirs(self):
        """Create required directories if they don't exist."""
        self.STAGING_DIR.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  PROJECT_ROOT={self.PROJECT_ROOT}\n"
            f"  POSTGRES_DSN={self.POSTGRES_DSN[:30]}...\n"
            f"  NEO4J_URI={self.NEO4J_URI}\n"
            f"  EMBEDDING_MODEL={self.EMBEDDING_MODEL}\n"
            f"  GCP_PROJECT_ID={self.GCP_PROJECT_ID}\n"
            f")"
        )


# Global config instance
config = Config()


# Convenience exports
POSTGRES_DSN = config.POSTGRES_DSN
NEO4J_URI = config.NEO4J_URI
NEO4J_USER = config.NEO4J_USER
NEO4J_PASSWORD = config.NEO4J_PASSWORD
EMBEDDING_MODEL = config.EMBEDDING_MODEL
EMBEDDING_DIM = config.EMBEDDING_DIM
RERANKER_MODEL = config.RERANKER_MODEL
GEMINI_API_KEY = config.GEMINI_API_KEY
GCP_PROJECT_ID = config.GCP_PROJECT_ID
GCP_BUCKET = config.GCP_BUCKET
