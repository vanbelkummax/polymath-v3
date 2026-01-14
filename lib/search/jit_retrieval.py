"""
Just-In-Time (JIT) Retrieval for Polymath v3.

Based on the GAM (Generate-Aggregate-Match) paper insight:
Query-time synthesis beats pre-computed summaries.

Key idea: Instead of storing embeddings of entire documents,
retrieve relevant passages and synthesize answers on-demand.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from lib.config import config
from lib.prompts import (
    JIT_SYNTHESIS_PROMPT,
    JIT_CLAIM_VERIFICATION_PROMPT,
    JIT_FOLLOWUP_QUERY_PROMPT,
    format_prompt,
)
from lib.search.hybrid_search import HybridSearcher, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of JIT retrieval."""

    query: str
    passages: list[SearchResult] = field(default_factory=list)
    synthesis: Optional[str] = None
    confidence: float = 0.0
    sources_used: int = 0


class JITRetriever:
    """
    Just-In-Time retrieval with optional synthesis.

    The key insight from GAM: Don't pre-compute document summaries.
    Instead, retrieve at query time and synthesize as needed.

    Usage:
        retriever = JITRetriever()
        result = retriever.retrieve("What methods exist for spatial deconvolution?")
        print(result.synthesis)  # LLM-generated answer
        for p in result.passages:
            print(f"- {p.title}: {p.passage_text[:100]}...")
    """

    def __init__(
        self,
        searcher: Optional[HybridSearcher] = None,
        synthesize: bool = True,
        model: Optional[str] = None,
    ):
        """
        Initialize JIT retriever.

        Args:
            searcher: HybridSearcher instance (creates new if None)
            synthesize: Whether to generate synthesis
            model: LLM model for synthesis (defaults to config.GEMINI_MODEL)
        """
        self.searcher = searcher or HybridSearcher()
        self.synthesize = synthesize
        self.model = model or config.GEMINI_MODEL
        self._client = None

    @property
    def client(self):
        """Lazy load Gemini client for synthesis."""
        if self._client is None and self.synthesize:
            try:
                from google import genai

                api_key = config.GEMINI_API_KEY
                if not api_key:
                    logger.warning("GEMINI_API_KEY not set, synthesis disabled")
                    self.synthesize = False
                    return None

                self._client = genai.Client(api_key=api_key)
            except ImportError:
                logger.warning("google-genai not installed, synthesis disabled")
                self.synthesize = False

        return self._client

    def retrieve(
        self,
        query: str,
        n_passages: int = 10,
        max_context_length: int = 8000,
        rerank: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant passages and optionally synthesize an answer.

        Args:
            query: User query
            n_passages: Number of passages to retrieve
            max_context_length: Max chars for synthesis context
            rerank: Whether to rerank results

        Returns:
            RetrievalResult with passages and optional synthesis
        """
        # Step 1: Retrieve passages
        response = self.searcher.search(query, n=n_passages, rerank=rerank)
        passages = response.results

        if not passages:
            return RetrievalResult(
                query=query,
                passages=[],
                synthesis="No relevant passages found.",
                confidence=0.0,
            )

        # Step 2: Synthesize if enabled
        synthesis = None
        confidence = 0.0

        if self.synthesize and self.client:
            synthesis, confidence = self._synthesize(query, passages, max_context_length)

        return RetrievalResult(
            query=query,
            passages=passages,
            synthesis=synthesis,
            confidence=confidence,
            sources_used=len(passages),
        )

    def _synthesize(
        self,
        query: str,
        passages: list[SearchResult],
        max_context_length: int,
    ) -> tuple[Optional[str], float]:
        """
        Synthesize an answer from retrieved passages.

        Returns:
            Tuple of (synthesis text, confidence score)
        """
        # Build context from passages
        context_parts = []
        total_length = 0

        for i, p in enumerate(passages):
            source_info = f"[{i+1}] {p.title}"
            if p.year:
                source_info += f" ({p.year})"

            passage_text = f"{source_info}\n{p.passage_text}\n"

            if total_length + len(passage_text) > max_context_length:
                break

            context_parts.append(passage_text)
            total_length += len(passage_text)

        context = "\n".join(context_parts)

        # Use centralized prompt
        prompt = format_prompt(JIT_SYNTHESIS_PROMPT, query=query, context=context)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.3,
                    "max_output_tokens": 1000,
                },
            )

            synthesis = response.text

            # Estimate confidence based on source usage
            # Count how many citations appear in the response
            citation_count = sum(
                1 for i in range(len(passages)) if f"[{i+1}]" in synthesis
            )
            confidence = min(1.0, citation_count / max(3, len(passages) * 0.5))

            return synthesis, confidence

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return None, 0.0

    def retrieve_for_claim(
        self,
        claim: str,
        n_passages: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve evidence for or against a specific claim.

        Useful for fact-checking and hallucination detection.

        Args:
            claim: The claim to verify
            n_passages: Number of passages to retrieve

        Returns:
            RetrievalResult with supporting/contradicting evidence
        """
        # Search for the claim
        response = self.searcher.search(claim, n=n_passages, rerank=True)
        passages = response.results

        if not passages or not self.synthesize or not self.client:
            return RetrievalResult(
                query=claim,
                passages=passages,
                synthesis=None,
                confidence=0.0,
                sources_used=len(passages),
            )

        # Build context
        context_parts = []
        for i, p in enumerate(passages):
            context_parts.append(f"[{i+1}] {p.title}: {p.passage_text}")

        context = "\n\n".join(context_parts)

        # Use centralized prompt
        prompt = format_prompt(JIT_CLAIM_VERIFICATION_PROMPT, claim=claim, context=context)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 500,
                },
            )

            synthesis = response.text

            # Extract confidence from response
            confidence = 0.5
            if "CONFIDENCE:" in synthesis:
                try:
                    conf_line = [
                        l for l in synthesis.split("\n") if "CONFIDENCE:" in l
                    ][0]
                    conf_value = conf_line.split(":")[-1].strip()
                    confidence = float(conf_value)
                except (ValueError, IndexError):
                    pass

            return RetrievalResult(
                query=claim,
                passages=passages,
                synthesis=synthesis,
                confidence=confidence,
                sources_used=len(passages),
            )

        except Exception as e:
            logger.error(f"Claim verification failed: {e}")
            return RetrievalResult(
                query=claim,
                passages=passages,
                synthesis=None,
                confidence=0.0,
                sources_used=len(passages),
            )

    def multi_hop_retrieve(
        self,
        query: str,
        n_hops: int = 2,
        n_passages_per_hop: int = 5,
    ) -> RetrievalResult:
        """
        Multi-hop retrieval for complex questions.

        Retrieves initial passages, extracts sub-questions,
        and retrieves additional context iteratively.

        Args:
            query: Complex query
            n_hops: Number of retrieval iterations
            n_passages_per_hop: Passages per iteration

        Returns:
            RetrievalResult with aggregated passages and synthesis
        """
        all_passages = []
        seen_ids = set()

        current_query = query

        for hop in range(n_hops):
            # Retrieve for current query
            response = self.searcher.search(
                current_query, n=n_passages_per_hop, rerank=True
            )

            # Add new passages
            for p in response.results:
                if p.passage_id not in seen_ids:
                    all_passages.append(p)
                    seen_ids.add(p.passage_id)

            if hop < n_hops - 1 and self.client:
                # Generate follow-up query for next hop
                current_query = self._generate_followup_query(
                    query, current_query, response.results
                )

        # Final synthesis with all passages
        synthesis = None
        confidence = 0.0

        if self.synthesize and self.client and all_passages:
            synthesis, confidence = self._synthesize(query, all_passages, 12000)

        return RetrievalResult(
            query=query,
            passages=all_passages,
            synthesis=synthesis,
            confidence=confidence,
            sources_used=len(all_passages),
        )

    def _generate_followup_query(
        self,
        original_query: str,
        current_query: str,
        passages: list[SearchResult],
    ) -> str:
        """Generate a follow-up query based on retrieved passages."""
        if not self.client:
            return current_query

        context = "\n".join([p.passage_text[:500] for p in passages[:3]])

        # Use centralized prompt
        prompt = format_prompt(
            JIT_FOLLOWUP_QUERY_PROMPT,
            original_query=original_query,
            current_query=current_query,
            context=context,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.5,
                    "max_output_tokens": 100,
                },
            )
            return response.text.strip()
        except Exception:
            return current_query


def retrieve(query: str, **kwargs) -> RetrievalResult:
    """
    Convenience function for JIT retrieval.

    Args:
        query: User query
        **kwargs: Arguments for JITRetriever.retrieve()

    Returns:
        RetrievalResult
    """
    retriever = JITRetriever()
    return retriever.retrieve(query, **kwargs)
