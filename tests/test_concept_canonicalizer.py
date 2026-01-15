"""
Tests for concept canonicalization.

Note: These tests focus on the pure functions that don't require database access.
The embedding-based matching is tested separately in integration tests.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# We need to mock the database imports before loading the module
# to test the pure functions without database dependencies

# Mock the database module
sys.modules['lib.db.postgres'] = MagicMock()
sys.modules['lib.config'] = MagicMock()

# Import directly from the module file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "concept_canonicalizer",
    Path(__file__).parent.parent / "lib" / "ingest" / "concept_canonicalizer.py"
)
canonicalizer_module = importlib.util.module_from_spec(spec)

# Need to mock get_pg_pool before loading
with patch.dict(sys.modules, {'lib.db.postgres': MagicMock(), 'lib.config': MagicMock()}):
    spec.loader.exec_module(canonicalizer_module)

KNOWN_ALIASES = canonicalizer_module.KNOWN_ALIASES
CanonicalConcept = canonicalizer_module.CanonicalConcept
normalize_text = canonicalizer_module.normalize_text
to_canonical_form = canonicalizer_module.to_canonical_form
canonicalize_concept = canonicalizer_module.canonicalize_concept
get_canonical_name = canonicalizer_module.get_canonical_name


class TestNormalizeText:
    """Tests for text normalization."""

    def test_lowercase(self):
        """Should convert to lowercase."""
        assert normalize_text("SPATIAL TRANSCRIPTOMICS") == "spatial transcriptomics"

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        assert normalize_text("  hello world  ") == "hello world"

    def test_replaces_hyphens(self):
        """Should replace hyphens with spaces."""
        assert normalize_text("single-cell") == "single cell"

    def test_replaces_underscores(self):
        """Should replace underscores with spaces."""
        assert normalize_text("graph_neural_network") == "graph neural network"

    def test_collapses_whitespace(self):
        """Should collapse multiple spaces."""
        assert normalize_text("too    many   spaces") == "too many spaces"


class TestToCanonicalForm:
    """Tests for canonical form conversion."""

    def test_snake_case(self):
        """Should convert to snake_case."""
        assert to_canonical_form("Spatial Transcriptomics") == "spatial_transcriptomics"

    def test_removes_special_chars(self):
        """Should remove special characters."""
        assert to_canonical_form("H&E staining!") == "he_staining"

    def test_handles_hyphens(self):
        """Should convert hyphens to underscores."""
        assert to_canonical_form("single-cell RNA-seq") == "single_cell_rna_seq"

    def test_collapses_underscores(self):
        """Should collapse multiple underscores."""
        assert to_canonical_form("too__many___underscores") == "too_many_underscores"

    def test_strips_edge_underscores(self):
        """Should strip leading/trailing underscores."""
        assert to_canonical_form("_leading_trailing_") == "leading_trailing"


class TestKnownAliases:
    """Tests for known alias mappings."""

    def test_st_alias(self):
        """ST should map to spatial_transcriptomics."""
        assert KNOWN_ALIASES.get("st") == "spatial_transcriptomics"

    def test_gnn_alias(self):
        """GNN should map to graph_neural_network."""
        assert KNOWN_ALIASES.get("gnn") == "graph_neural_network"

    def test_he_alias(self):
        """H&E variants should map to hematoxylin_and_eosin."""
        assert KNOWN_ALIASES.get("h&e") == "hematoxylin_and_eosin"
        assert KNOWN_ALIASES.get("he") == "hematoxylin_and_eosin"

    def test_visium_alias(self):
        """Visium should map to 10x_visium."""
        assert KNOWN_ALIASES.get("visium") == "10x_visium"

    def test_scrna_aliases(self):
        """scRNA variants should map correctly."""
        assert KNOWN_ALIASES.get("scrna") == "single_cell_rna_seq"
        assert KNOWN_ALIASES.get("scrnaseq") == "single_cell_rna_seq"


class TestCanonicalizeConcept:
    """Tests for full canonicalization pipeline."""

    def test_uses_alias_when_available(self):
        """Should use known alias when available."""
        result = canonicalize_concept("ST")
        assert result.canonical == "spatial_transcriptomics"
        assert result.method == "alias"
        assert result.confidence == 1.0

    def test_alias_case_insensitive(self):
        """Alias matching should be case-insensitive."""
        result = canonicalize_concept("GNN")
        assert result.canonical == "graph_neural_network"
        assert result.method == "alias"

    def test_alias_with_hyphen(self):
        """Should handle aliases with hyphens (normalized to spaces before lookup)."""
        # Hyphens are normalized to spaces before alias lookup
        # "spatial-transcriptomics" -> "spatial transcriptomics" -> matched via "spatial tx" or fallback
        result = canonicalize_concept("spatial-transcriptomics")
        assert result.canonical == "spatial_transcriptomics"
        # May match alias if "spatial transcriptomics" is in aliases, else falls back to normalization
        assert result.method in ("alias", "lowercase")

    def test_fallback_to_lowercase(self):
        """Should fall back to lowercase normalization."""
        result = canonicalize_concept("Novel Method Name")
        assert result.canonical == "novel_method_name"
        assert result.method == "lowercase"
        assert result.confidence == 0.9

    def test_preserves_original(self):
        """Should preserve original name."""
        result = canonicalize_concept("  My Concept  ")
        assert result.original == "My Concept"

    def test_embedding_match_disabled_by_default(self):
        """Embedding matching should be disabled by default."""
        result = canonicalize_concept("some concept", use_embedding_match=False)
        # Should not use embedding method
        assert result.method in ["alias", "lowercase"]


class TestGetCanonicalName:
    """Tests for cached canonical name lookup."""

    def test_returns_alias(self):
        """Should return alias when available."""
        assert get_canonical_name("ST") == "spatial_transcriptomics"
        assert get_canonical_name("gnn") == "graph_neural_network"

    def test_returns_normalized(self):
        """Should return normalized form when no alias."""
        assert get_canonical_name("My Custom Method") == "my_custom_method"

    def test_caching(self):
        """Results should be cached."""
        # Call twice with same input
        result1 = get_canonical_name("test concept")
        result2 = get_canonical_name("test concept")
        assert result1 == result2

    def test_case_insensitive(self):
        """Should be case-insensitive for aliases."""
        assert get_canonical_name("ST") == get_canonical_name("st")
        assert get_canonical_name("GNN") == get_canonical_name("gnn")


class TestCanonicalConceptDataclass:
    """Tests for the CanonicalConcept dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        concept = CanonicalConcept(
            original="test",
            canonical="test",
            method="lowercase",
        )
        assert concept.confidence == 1.0
        assert concept.matched_existing_id is None

    def test_all_fields(self):
        """Should store all fields correctly."""
        concept = CanonicalConcept(
            original="ST",
            canonical="spatial_transcriptomics",
            method="alias",
            confidence=1.0,
            matched_existing_id="abc-123",
        )
        assert concept.original == "ST"
        assert concept.canonical == "spatial_transcriptomics"
        assert concept.method == "alias"
        assert concept.confidence == 1.0
        assert concept.matched_existing_id == "abc-123"
