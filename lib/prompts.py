"""
Centralized prompt templates for Polymath v3.

All LLM prompts are defined here to make prompt engineering easier
and to ensure consistency across the codebase.
"""

# =============================================================================
# Concept Extraction Prompts
# =============================================================================

CONCEPT_EXTRACTION_PROMPT = """Extract key scientific concepts from this research text. Return ONLY valid JSON:
{{"methods": [], "problems": [], "domains": [], "datasets": [], "metrics": [], "entities": []}}

Rules:
- methods: techniques, algorithms, tools (e.g., "spatial transcriptomics", "gradient descent")
- problems: challenges being addressed (e.g., "cell type deconvolution", "batch effects")
- domains: research fields (e.g., "computational pathology", "drug discovery")
- datasets: specific datasets mentioned (e.g., "Visium HD", "10x Xenium", "TCGA")
- metrics: evaluation metrics (e.g., "PCC", "SSIM", "AUC", "R²")
- entities: genes, proteins, diseases, cell types (e.g., "TP53", "EGFR", "T-cell")
- Be specific, not generic (e.g., "transformer attention" not just "neural network")
- Only include explicitly mentioned concepts
- Return 5-15 concepts maximum

Text:
{text}"""


# =============================================================================
# JIT Retrieval Prompts
# =============================================================================

JIT_SYNTHESIS_PROMPT = """Based on the following research passages, answer this question:

QUESTION: {query}

CONTEXT:
{context}

INSTRUCTIONS:
- Answer based ONLY on the provided context
- Cite sources using [1], [2], etc.
- If the context doesn't contain enough information, say so
- Be concise but thorough
- Include specific details, numbers, and methods when available

ANSWER:"""


JIT_CLAIM_VERIFICATION_PROMPT = """Evaluate this claim against the provided evidence:

CLAIM: {claim}

EVIDENCE:
{context}

INSTRUCTIONS:
- Determine if the claim is SUPPORTED, CONTRADICTED, or INSUFFICIENT EVIDENCE
- Cite specific passages using [1], [2], etc.
- Explain your reasoning briefly

OUTPUT FORMAT:
VERDICT: [SUPPORTED/CONTRADICTED/INSUFFICIENT]
CONFIDENCE: [0.0-1.0]
REASONING: [Your explanation with citations]"""


JIT_FOLLOWUP_QUERY_PROMPT = """Original question: {original_query}
Current search: {current_query}

Retrieved context:
{context}

Based on this context, what follow-up question would help answer the original question better?
Return ONLY the follow-up question, nothing else."""


# =============================================================================
# Hallucination Detection Prompts
# =============================================================================

HALLUCINATION_CLAIM_EXTRACTION_PROMPT = """Extract all verifiable factual claims from this text.
Each claim should be:
- Atomic (one fact per claim)
- Self-contained (understandable without context)
- Verifiable (could be checked against sources)

TEXT:
{text}

OUTPUT FORMAT (one claim per line):
CLAIM 1: [claim text]
SOURCE: [original sentence containing this claim]
---
CLAIM 2: [claim text]
SOURCE: [original sentence containing this claim]
---
...

Extract all claims now:"""


HALLUCINATION_VERIFICATION_PROMPT = """Verify this claim against the provided evidence.

CLAIM: {claim}

EVIDENCE:
{evidence}

INSTRUCTIONS:
- Determine if the claim is SUPPORTED, CONTRADICTED, or UNVERIFIABLE by the evidence
- SUPPORTED: Evidence clearly confirms the claim
- CONTRADICTED: Evidence clearly refutes the claim
- UNVERIFIABLE: Evidence is insufficient or irrelevant

OUTPUT FORMAT:
VERDICT: [SUPPORTED/CONTRADICTED/UNVERIFIABLE]
CONFIDENCE: [0.0-1.0]
REASONING: [Brief explanation citing evidence numbers]

Verify now:"""


# =============================================================================
# Helper Functions
# =============================================================================

def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with the given arguments.

    Args:
        template: Prompt template string
        **kwargs: Values to substitute

    Returns:
        Formatted prompt string
    """
    return template.format(**kwargs)
