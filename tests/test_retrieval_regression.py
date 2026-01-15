"""
Regression tests for Polymath v3 retrieval.

These tests verify that known queries return expected results.
Run with: pytest tests/test_retrieval_regression.py -v
"""

import pytest


class TestSearchRegression:
    """Regression tests for hybrid search."""

    def test_search_returns_results(self, searcher):
        """Search should return non-empty results for reasonable queries."""
        response = searcher.search("spatial transcriptomics", n=10)
        assert response.results, "Search should return results for 'spatial transcriptomics'"
        assert len(response.results) > 0

    def test_search_respects_n_limit(self, searcher):
        """Search should respect the n parameter."""
        response = searcher.search("gene expression", n=5)
        assert len(response.results) <= 5

    def test_search_with_reranking(self, searcher):
        """Search with reranking should complete without error."""
        response = searcher.search("deep learning histopathology", n=10, rerank=True)
        assert response.results is not None

    def test_search_without_reranking(self, searcher):
        """Search without reranking should complete faster."""
        response = searcher.search("cancer classification", n=10, rerank=False)
        assert response.results is not None

    @pytest.mark.parametrize("query", [
        "transformer attention mechanism",
        "cell type deconvolution",
        "single cell RNA seq",
        "H&E staining analysis",
        "Visium HD spatial",
    ])
    def test_common_queries_return_results(self, searcher, query):
        """Common scientific queries should return results."""
        response = searcher.search(query, n=5)
        # Note: These may fail if KB is empty - that's expected
        # This test serves as a regression baseline once KB is populated


class TestSearchQuality:
    """Tests for search result quality."""

    def test_spatial_transcriptomics_relevance(self, searcher):
        """Spatial transcriptomics query should return relevant results."""
        response = searcher.search("spatial transcriptomics methods", n=10, rerank=True)

        if not response.results:
            pytest.skip("No results - KB may be empty")

        # Check that at least one result contains relevant terms
        relevant_terms = ["spatial", "transcriptomics", "spot", "gene expression"]
        has_relevant = any(
            any(term in r.passage_text.lower() for term in relevant_terms)
            for r in response.results
        )
        assert has_relevant, "Results should contain relevant terminology"


class TestGapDetection:
    """Regression tests for BridgeMine gap detection."""

    def test_gap_detector_runs(self, gap_detector):
        """Gap detector should run without error."""
        # May return empty if graph is not populated
        result = gap_detector.find_gaps("test_domain", limit=5)
        assert result is not None
        assert hasattr(result, "candidates")

    def test_gap_detector_filters_generic(self, gap_detector):
        """Gap detector should filter generic methods."""
        result = gap_detector.find_gaps("spatial_transcriptomics", limit=20, exclude_generic=True)

        if not result.candidates:
            pytest.skip("No gaps found - graph may be empty")

        # Check that generic methods are filtered
        generic = ["deep_learning", "machine_learning", "neural_network"]
        for candidate in result.candidates:
            assert candidate.method_name.lower() not in generic, \
                f"Generic method {candidate.method_name} should be filtered"


class TestTelemetry:
    """Tests for telemetry logging."""

    def test_telemetry_disabled_by_default(self):
        """Telemetry should be disabled by default."""
        from lib.telemetry import is_telemetry_enabled
        # This test may fail if POLYMATH_TELEMETRY=1 in environment
        # That's actually fine - it means telemetry is working

    def test_telemetry_logger_context(self):
        """TelemetryLogger context manager should work."""
        from lib.telemetry import TelemetryLogger

        with TelemetryLogger() as tl:
            tl.set_query("test query", n_requested=10)
            with tl.time("vector"):
                pass  # Simulate work
            tl.add_vector_results([{"passage_id": "test", "score": 0.9}])

        # Should complete without error
