"""
Validation modules for Polymath v3.

Provides hallucination detection using the 3-stage approach from HaluMem paper:
1. Extraction: Extract verifiable claims from generated text
2. Update: Contextualize claims with retrieved evidence
3. QA: Verify each claim against evidence
"""

from .hallucination import (
    HallucinationDetector,
    Claim,
    VerificationResult,
    detect_hallucinations,
)

__all__ = [
    "HallucinationDetector",
    "Claim",
    "VerificationResult",
    "detect_hallucinations",
]
