"""
BridgeMine: Cross-domain research gap detection for Polymath v3.

Identifies opportunities to transfer methods from one domain to another
by analyzing the concept graph for under-explored method-problem pairs.
"""

from .gap_detection import GapDetector, GapCandidate
from .novelty_check import NoveltyChecker, NoveltyResult

__all__ = [
    "GapDetector",
    "GapCandidate",
    "NoveltyChecker",
    "NoveltyResult",
]
