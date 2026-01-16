"""
Concept extraction for Polymath v3.

Uses Gemini API for high-quality concept extraction from passages.
Supports both real-time extraction and batch processing via GCP.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

from typing import TypedDict

from lib.config import config
from lib.prompts import CONCEPT_EXTRACTION_PROMPT, format_prompt

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions (prevents key errors in dict handling)
# =============================================================================


class ConceptDict(TypedDict, total=False):
    """Type definition for concept dictionaries from JSON parsing."""

    name: str
    concept: str  # Alternative key
    confidence: float
    evidence: str


class ExtractionResponseDict(TypedDict, total=False):
    """Type definition for extraction API response."""

    methods: list[str | ConceptDict]
    problems: list[str | ConceptDict]
    domains: list[str | ConceptDict]
    datasets: list[str | ConceptDict]
    metrics: list[str | ConceptDict]
    entities: list[str | ConceptDict]


@dataclass
class ExtractedConcept:
    """A single extracted concept."""

    name: str
    type: str  # method, problem, domain, dataset, metric, entity
    confidence: float = 0.8
    evidence: Optional[str] = None  # Quote from source


@dataclass
class ExtractionResult:
    """Result of concept extraction."""

    concepts: list[ExtractedConcept] = field(default_factory=list)
    raw_response: Optional[str] = None
    model: str = "unknown"
    success: bool = True
    error: Optional[str] = None


class ConceptExtractor:
    """
    Extract scientific concepts from text using Gemini API.

    Usage:
        extractor = ConceptExtractor()
        result = extractor.extract("We used optimal transport for...")
        for concept in result.concepts:
            print(f"{concept.type}: {concept.name}")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize extractor.

        Args:
            model: Gemini model to use (defaults to config.GEMINI_MODEL)
            api_key: API key (uses config if not provided)
        """
        self.model = model or config.GEMINI_MODEL
        self.api_key = api_key or config.GEMINI_API_KEY
        self._client = None

    @property
    def client(self):
        """Lazy load Gemini client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not configured")

            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError("google-genai package not installed")

        return self._client

    def extract(
        self,
        text: str,
        max_length: int = 4000,
        canonicalize: bool = True,
    ) -> ExtractionResult:
        """
        Extract concepts from text.

        Args:
            text: Text to extract concepts from
            max_length: Maximum text length to process
            canonicalize: Whether to canonicalize concept names

        Returns:
            ExtractionResult with extracted concepts
        """
        if not text or len(text.strip()) < 50:
            return ExtractionResult(
                success=False,
                error="Text too short for concept extraction",
            )

        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + "..."

        prompt = format_prompt(CONCEPT_EXTRACTION_PROMPT, text=text)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 8192,  # Model supports up to 8192
                },
            )

            raw_response = response.text
            concepts = self._parse_response(raw_response)

            # Canonicalize concept names to prevent graph fragmentation
            if canonicalize and concepts:
                from lib.ingest.concept_canonicalizer import canonicalize_concepts
                concepts = canonicalize_concepts(concepts)

            return ExtractionResult(
                concepts=concepts,
                raw_response=raw_response,
                model=self.model,
                success=True,
            )

        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return ExtractionResult(
                success=False,
                error=str(e),
                model=self.model,
            )

    def _parse_response(self, response: str) -> list[ExtractedConcept]:
        """Parse JSON response into concept list."""
        concepts = []

        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            logger.warning("No JSON found in response")
            return concepts

        try:
            data = json.loads(json_match.group(0))

            type_mapping = {
                "methods": "method",
                "problems": "problem",
                "domains": "domain",
                "datasets": "dataset",
                "metrics": "metric",
                "entities": "entity",
            }

            for key, concept_type in type_mapping.items():
                items = data.get(key, [])
                for item in items:
                    if isinstance(item, str) and item.strip():
                        concepts.append(
                            ExtractedConcept(
                                name=item.strip(),
                                type=concept_type,
                                confidence=0.8,
                            )
                        )
                    elif isinstance(item, dict):
                        name = item.get("name") or item.get("concept")
                        if name:
                            concepts.append(
                                ExtractedConcept(
                                    name=name.strip(),
                                    type=concept_type,
                                    confidence=item.get("confidence", 0.8),
                                    evidence=item.get("evidence"),
                                )
                            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")

        return concepts

    def extract_batch(
        self,
        texts: list[str],
        batch_size: int = 10,
    ) -> list[ExtractionResult]:
        """
        Extract concepts from multiple texts.

        Args:
            texts: List of texts to process
            batch_size: Number of texts to process in parallel

        Returns:
            List of ExtractionResult objects
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            for text in batch:
                result = self.extract(text)
                results.append(result)

        return results


def extract_concepts_from_passage(
    passage_text: str,
    extractor: Optional[ConceptExtractor] = None,
) -> list[ExtractedConcept]:
    """
    Convenience function to extract concepts from a single passage.

    Args:
        passage_text: Text to extract from
        extractor: Extractor instance (creates new if None)

    Returns:
        List of extracted concepts
    """
    extractor = extractor or ConceptExtractor()
    result = extractor.extract(passage_text)
    return result.concepts if result.success else []


# Concept type mapping for database storage
CONCEPT_TYPE_LABELS = {
    "method": "METHOD",
    "problem": "PROBLEM",
    "domain": "DOMAIN",
    "dataset": "DATASET",
    "metric": "METRIC",
    "entity": "ENTITY",
    "mechanism": "MECHANISM",
    "data_structure": "DATA_STRUCTURE",
}
