"""
Hallucination detection for Polymath v3.

Based on the HaluMem paper's 3-stage approach:
1. EXTRACTION: Extract atomic, verifiable claims from text
2. UPDATE: Contextualize claims with retrieved evidence
3. QA: Verify each claim against evidence (SUPPORTED/CONTRADICTED/UNVERIFIABLE)

This catches both intrinsic hallucinations (self-contradiction)
and extrinsic hallucinations (factual errors vs. source material).
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from lib.config import config
from lib.prompts import (
    HALLUCINATION_CLAIM_EXTRACTION_PROMPT,
    HALLUCINATION_VERIFICATION_PROMPT,
    format_prompt,
)
from lib.search.hybrid_search import HybridSearcher, SearchResult

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of claim verification."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"
    ERROR = "error"


@dataclass
class Claim:
    """An atomic, verifiable claim extracted from text."""

    text: str
    source_sentence: str
    claim_index: int


@dataclass
class VerificationResult:
    """Result of verifying a single claim."""

    claim: Claim
    status: VerificationStatus
    confidence: float = 0.0
    evidence: list[SearchResult] = field(default_factory=list)
    reasoning: Optional[str] = None


@dataclass
class HallucinationReport:
    """Full hallucination detection report."""

    original_text: str
    claims: list[Claim] = field(default_factory=list)
    results: list[VerificationResult] = field(default_factory=list)

    # Aggregate stats
    supported_count: int = 0
    contradicted_count: int = 0
    unverifiable_count: int = 0

    # Citation density metrics (for "soft" evidence gating)
    citation_density: float = 0.0  # Citations per sentence
    uncited_sentences: list[str] = field(default_factory=list)

    @property
    def hallucination_score(self) -> float:
        """
        Calculate hallucination score (0-1).

        Higher score = more hallucinations detected.
        """
        total = len(self.results)
        if total == 0:
            return 0.0

        # Weight contradicted claims heavily
        score = (
            self.contradicted_count * 1.0 + self.unverifiable_count * 0.3
        ) / total

        return min(1.0, score)

    @property
    def evidence_coverage(self) -> float:
        """
        Calculate evidence coverage (0-1).

        Higher = more claims supported by evidence.
        This is the "soft" evidence gate metric.
        """
        total = len(self.results)
        if total == 0:
            return 1.0  # No claims = nothing to verify = OK

        return self.supported_count / total

    @property
    def is_reliable(self) -> bool:
        """Check if text is reliable (low hallucination)."""
        return self.hallucination_score < 0.3

    @property
    def needs_more_citations(self) -> bool:
        """Check if text needs more citations (soft flag, not hard failure)."""
        return self.citation_density < 0.3 or self.evidence_coverage < 0.5


class HallucinationDetector:
    """
    Detect hallucinations using 3-stage verification.

    Usage:
        detector = HallucinationDetector()
        report = detector.detect("Claude says X about Y...")
        print(f"Hallucination score: {report.hallucination_score:.2f}")
        for result in report.results:
            if result.status == VerificationStatus.CONTRADICTED:
                print(f"HALLUCINATION: {result.claim.text}")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        searcher: Optional[HybridSearcher] = None,
        n_evidence: int = 5,
    ):
        """
        Initialize detector.

        Args:
            model: LLM model for extraction and verification (defaults to config.GEMINI_MODEL)
            searcher: Search engine for evidence retrieval
            n_evidence: Number of evidence passages to retrieve per claim
        """
        self.model = model or config.GEMINI_MODEL
        self.searcher = searcher or HybridSearcher()
        self.n_evidence = n_evidence
        self._client = None

    @property
    def client(self):
        """Lazy load Gemini client."""
        if self._client is None:
            try:
                from google import genai

                api_key = config.GEMINI_API_KEY
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not configured")

                self._client = genai.Client(api_key=api_key)

            except ImportError:
                raise ImportError("google-genai package not installed")

        return self._client

    def detect(self, text: str) -> HallucinationReport:
        """
        Detect hallucinations in text using 3-stage approach.

        Args:
            text: Text to check for hallucinations

        Returns:
            HallucinationReport with detailed results
        """
        # Stage 1: Extract claims
        logger.info("Stage 1: Extracting claims...")
        claims = self._extract_claims(text)

        if not claims:
            return HallucinationReport(original_text=text)

        logger.info(f"Extracted {len(claims)} claims")

        # Stage 2 & 3: Update and verify each claim
        results = []
        supported = 0
        contradicted = 0
        unverifiable = 0

        for claim in claims:
            logger.info(f"Verifying claim {claim.claim_index + 1}/{len(claims)}")

            # Stage 2: Retrieve evidence
            evidence = self._retrieve_evidence(claim)

            # Stage 3: Verify claim against evidence
            result = self._verify_claim(claim, evidence)
            results.append(result)

            # Update counts
            if result.status == VerificationStatus.SUPPORTED:
                supported += 1
            elif result.status == VerificationStatus.CONTRADICTED:
                contradicted += 1
            else:
                unverifiable += 1

        # Calculate citation density
        citation_density, uncited = self._calculate_citation_density(text)

        return HallucinationReport(
            original_text=text,
            claims=claims,
            results=results,
            supported_count=supported,
            contradicted_count=contradicted,
            unverifiable_count=unverifiable,
            citation_density=citation_density,
            uncited_sentences=uncited,
        )

    def _extract_claims(self, text: str) -> list[Claim]:
        """
        Stage 1: Extract atomic, verifiable claims from text.

        Returns list of Claim objects.
        """
        prompt = format_prompt(HALLUCINATION_CLAIM_EXTRACTION_PROMPT, text=text)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 2000,
                },
            )

            return self._parse_claims(response.text)

        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return []

    def _parse_claims(self, response: str) -> list[Claim]:
        """Parse extracted claims from LLM response."""
        claims = []
        blocks = response.split("---")

        for i, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue

            # Extract claim text
            claim_match = re.search(r"CLAIM \d+:\s*(.+?)(?:\n|SOURCE:|$)", block, re.DOTALL)
            source_match = re.search(r"SOURCE:\s*(.+?)$", block, re.DOTALL)

            if claim_match:
                claim_text = claim_match.group(1).strip()
                source_text = source_match.group(1).strip() if source_match else ""

                if claim_text:
                    claims.append(
                        Claim(
                            text=claim_text,
                            source_sentence=source_text,
                            claim_index=len(claims),
                        )
                    )

        return claims

    def _retrieve_evidence(self, claim: Claim) -> list[SearchResult]:
        """
        Stage 2: Retrieve evidence for claim verification.

        Returns list of relevant passages.
        """
        try:
            response = self.searcher.search(
                claim.text,
                n=self.n_evidence,
                rerank=True,
            )
            return response.results

        except Exception as e:
            logger.error(f"Evidence retrieval failed: {e}")
            return []

    def _verify_claim(
        self, claim: Claim, evidence: list[SearchResult]
    ) -> VerificationResult:
        """
        Stage 3: Verify claim against retrieved evidence.

        Returns VerificationResult with status and reasoning.
        """
        if not evidence:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.0,
                evidence=[],
                reasoning="No evidence found in knowledge base.",
            )

        # Build evidence context
        evidence_text = ""
        for i, e in enumerate(evidence):
            evidence_text += f"\n[{i+1}] {e.title}"
            if e.year:
                evidence_text += f" ({e.year})"
            evidence_text += f"\n{e.passage_text}\n"

        prompt = format_prompt(
            HALLUCINATION_VERIFICATION_PROMPT,
            claim=claim.text,
            evidence=evidence_text,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 500,
                },
            )

            return self._parse_verification(response.text, claim, evidence)

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.ERROR,
                confidence=0.0,
                evidence=evidence,
                reasoning=f"Verification error: {e}",
            )

    def _parse_verification(
        self, response: str, claim: Claim, evidence: list[SearchResult]
    ) -> VerificationResult:
        """Parse verification response from LLM."""
        # Extract verdict
        status = VerificationStatus.UNVERIFIABLE
        if "VERDICT:" in response:
            verdict_line = [l for l in response.split("\n") if "VERDICT:" in l][0]
            verdict = verdict_line.split(":")[-1].strip().upper()

            if "SUPPORTED" in verdict:
                status = VerificationStatus.SUPPORTED
            elif "CONTRADICTED" in verdict:
                status = VerificationStatus.CONTRADICTED

        # Extract confidence
        confidence = 0.5
        if "CONFIDENCE:" in response:
            try:
                conf_line = [l for l in response.split("\n") if "CONFIDENCE:" in l][0]
                conf_value = conf_line.split(":")[-1].strip()
                confidence = float(conf_value)
            except (ValueError, IndexError):
                pass

        # Extract reasoning
        reasoning = ""
        if "REASONING:" in response:
            reasoning_start = response.find("REASONING:")
            reasoning = response[reasoning_start + len("REASONING:"):].strip()

        return VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=evidence,
            reasoning=reasoning,
        )

    def _calculate_citation_density(self, text: str) -> tuple[float, list[str]]:
        """
        Calculate citation density for soft evidence gating.

        Returns:
            Tuple of (citations_per_sentence, list_of_uncited_sentences)
        """
        # Split into sentences (simple heuristic)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0, []

        # Count citations per sentence
        # Look for patterns like [1], [2,3], (Author, 2024), etc.
        citation_patterns = [
            r'\[\d+\]',           # [1], [2]
            r'\[\d+(?:,\s*\d+)*\]',  # [1,2,3]
            r'\[\d+-\d+\]',       # [1-5]
            r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)',  # (Smith, 2024)
            r'\([A-Z][a-z]+\s+&\s+[A-Z][a-z]+,?\s*\d{4}\)',  # (Smith & Jones, 2024)
        ]

        combined_pattern = '|'.join(citation_patterns)
        cited_count = 0
        uncited = []

        for sentence in sentences:
            if re.search(combined_pattern, sentence):
                cited_count += 1
            else:
                # Only track uncited sentences that are substantive
                if len(sentence) > 30:  # Skip short sentences
                    uncited.append(sentence[:100] + "..." if len(sentence) > 100 else sentence)

        citation_density = cited_count / len(sentences) if sentences else 0.0

        return citation_density, uncited[:10]  # Limit uncited list

    def quick_check(self, text: str, threshold: float = 0.3) -> tuple[bool, float]:
        """
        Quick hallucination check without full report.

        Args:
            text: Text to check
            threshold: Hallucination score threshold

        Returns:
            Tuple of (is_reliable, hallucination_score)
        """
        report = self.detect(text)
        return report.is_reliable, report.hallucination_score


def detect_hallucinations(text: str, **kwargs) -> HallucinationReport:
    """
    Convenience function for hallucination detection.

    Args:
        text: Text to check
        **kwargs: Arguments for HallucinationDetector

    Returns:
        HallucinationReport
    """
    detector = HallucinationDetector(**kwargs)
    return detector.detect(text)


def verify_claim(claim: str, **kwargs) -> VerificationResult:
    """
    Verify a single claim against the knowledge base.

    Args:
        claim: Claim text to verify
        **kwargs: Arguments for HallucinationDetector

    Returns:
        VerificationResult
    """
    detector = HallucinationDetector(**kwargs)
    claim_obj = Claim(text=claim, source_sentence=claim, claim_index=0)
    evidence = detector._retrieve_evidence(claim_obj)
    return detector._verify_claim(claim_obj, evidence)


def extract_claims(text: str, **kwargs) -> list[Claim]:
    """
    Extract verifiable claims from text.

    Convenience function for use in research agent and other modules.

    Args:
        text: Text to extract claims from
        **kwargs: Arguments for HallucinationDetector

    Returns:
        List of Claim objects
    """
    detector = HallucinationDetector(**kwargs)
    return detector._extract_claims(text)
