"""
Pytest configuration and fixtures for Polymath v3 tests.
"""

import pytest
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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
