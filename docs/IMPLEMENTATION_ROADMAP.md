# Polymath v3 Implementation Roadmap

## Executive Summary

After analyzing the codebase, I recommend implementing **all three tracks in parallel** with different priority levels:

| Track | Priority | Impact | Effort | ROI |
|-------|----------|--------|--------|-----|
| **B: BridgeMine** | P0 | High | Medium | **Highest** - Unique differentiator for thesis/grants |
| **A: Infrastructure** | P1 | Medium | Low | High - Enables scale |
| **C: MCP Agent** | P2 | Medium | Medium | Good - Daily workflow |

**Rationale for MD-PhD use case:**
- BridgeMine is the killer feature - no other system finds novel cross-domain research gaps
- Infrastructure fixes are quick wins that prevent production bottlenecks
- MCP agent improvements enhance daily usability but aren't blocking

---

## Track A: Infrastructure (P1)

### Problem 1: Connection Pool / Worker Mismatch

**Current State:**
```python
# lib/db/postgres.py - hardcoded
def get_pg_pool(min_size: int = 2, max_size: int = 10) -> ConnectionPool:

# lib/config.py - configurable
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
```

If `NUM_WORKERS > 10`, threads block waiting for connections.

**Fix:**
```python
def get_pg_pool(min_size: int = None, max_size: int = None) -> ConnectionPool:
    from lib.config import config
    min_size = min_size or min(2, config.NUM_WORKERS)
    max_size = max_size or config.NUM_WORKERS + 2  # +2 for overhead
```

**Effort:** 15 minutes

---

### Problem 2: Neo4j Bulk Load Performance

**Current State:** `scripts/sync_neo4j.py` uses `MERGE` for all operations.

**Issue:** `MERGE` on relationships acquires write locks, causing bottlenecks during full re-sync.

**Fix for initial bulk load:**
```bash
# Export to CSV
python scripts/export_for_neo4j.py --format csv

# Use neo4j-admin import (10-100x faster)
neo4j-admin database import full \
  --nodes=Passage=passages.csv \
  --nodes=METHOD=methods.csv \
  --relationships=MENTIONS=mentions.csv
```

**For incremental sync:** Keep current `MERGE` approach (correct for updates).

**Effort:** 2 hours (script to export CSVs)

---

### Problem 3: Idempotent Batch Ingestion

**Current State:** No crash recovery for batch jobs.

**Fix:** Add `ingest_jobs` table:
```sql
CREATE TABLE ingest_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pdf_path TEXT NOT NULL UNIQUE,
    status TEXT CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    doc_id UUID REFERENCES documents(doc_id),
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    retry_count INT DEFAULT 0
);
```

**Effort:** 1 hour

---

## Track B: BridgeMine Intelligence (P0)

### Enhancement 1: OpenAlex Integration for Novelty Check

**Why:** OpenAlex is faster, free, and has better coverage than Semantic Scholar.

**Add to `novelty_check.py`:**
```python
def _check_openalex(self, query: str) -> tuple[int, list[dict]]:
    """Check OpenAlex for relevant works."""
    import httpx

    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": 20,
        "mailto": config.OPENALEX_EMAIL,
        "filter": "publication_year:>2020",
        "select": "id,title,publication_year,cited_by_count,doi",
    }

    response = httpx.get(url, params=params, timeout=15)
    data = response.json()

    count = data.get("meta", {}).get("count", 0)
    papers = [
        {
            "openalex_id": w["id"],
            "title": w.get("title", ""),
            "year": w.get("publication_year"),
            "citations": w.get("cited_by_count", 0),
            "doi": w.get("doi"),
            "source": "openalex",
        }
        for w in data.get("results", [])
    ]

    return count, papers
```

**Modify RRF fusion to include OpenAlex as 4th signal.**

**Effort:** 1 hour

---

### Enhancement 2: Trivial Gap Filtering

**Current State:** Generic methods filtered by name (`GENERIC_METHODS` set).

**Problem:** Some gaps are trivially obvious (e.g., "use machine learning" suggestions).

**Add semantic filtering:**
```python
class GapQualityFilter:
    """Filter out low-quality gaps using heuristics."""

    TRIVIAL_PATTERNS = [
        r"apply .* to .*",  # Too generic
        r"use .* for .*",   # Not specific
    ]

    MIN_METHOD_SPECIFICITY = 0.7  # Embedding distance from "method"

    def filter(self, candidates: list[GapCandidate]) -> list[GapCandidate]:
        return [c for c in candidates if self._is_quality(c)]

    def _is_quality(self, c: GapCandidate) -> bool:
        # Check method specificity (not just "deep learning")
        if c.method_mentions < 20:  # Too obscure
            return False
        if c.method_mentions > 5000:  # Too generic
            return False
        if c.problem_similarity < 0.75:  # Weak analogy
            return False
        return True
```

**Effort:** 30 minutes

---

### Enhancement 3: Conditional Context in Claim Extraction

**Current Issue:** "X causes Y" may be true in mice but false in humans.

**Update `lib/prompts.py`:**
```python
HALLUCINATION_CLAIM_EXTRACTION_PROMPT = """Extract all verifiable factual claims from this text.
Each claim should be:
- Atomic (one fact per claim)
- Self-contained (understandable without context)
- Verifiable (could be checked against sources)
- INCLUDE CONDITIONAL CONTEXT where present (organism, cell type, experimental conditions)

TEXT:
{text}

OUTPUT FORMAT (one claim per line):
CLAIM 1: [claim text with context, e.g., "In murine models, X causes Y"]
CONTEXT: [organism/cell type/conditions if specified]
SOURCE: [original sentence containing this claim]
---
...
"""
```

**Effort:** 15 minutes

---

### Enhancement 4: Domain-Specific Concept Hierarchy

For spatial transcriptomics, add domain-aware concept relationships:

```python
SPATIAL_TRANSCRIPTOMICS_HIERARCHY = {
    "methods": {
        "spot_deconvolution": ["Cell2location", "RCTD", "SPOTlight", "Tangram"],
        "image_prediction": ["Img2ST", "HisToGene", "iStar", "GHIST"],
        "spatial_clustering": ["SpaGCN", "BayesSpace", "STAGATE"],
    },
    "problems": {
        "resolution": ["spot size", "subcellular", "single-cell"],
        "coverage": ["gene dropout", "sparsity", "capture efficiency"],
    },
}
```

This helps BridgeMine find gaps like:
> "RCTD (spot deconvolution) hasn't been combined with morphology features from HisToGene"

**Effort:** 2 hours (build hierarchy from your papers)

---

## Track C: MCP Agent Loop (P2)

### Enhancement 1: Research Agent Workflow

Add a new MCP tool that loops autonomously:

```python
Tool(
    name="research_deep_dive",
    description="""Autonomously investigate a research question.

    Performs multiple rounds of:
    1. Search knowledge base
    2. Identify gaps in findings
    3. Verify claims
    4. Refine search based on gaps

    Returns a comprehensive research brief with citations.

    Args:
        question: Research question to investigate
        max_rounds: Maximum search iterations (default: 3)
        domain: Optional domain focus (e.g., "spatial_transcriptomics")""",
    inputSchema={
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "max_rounds": {"type": "integer", "default": 3},
            "domain": {"type": "string"},
        },
        "required": ["question"],
    },
)
```

**Implementation:**
```python
async def handle_research_deep_dive(args: dict) -> list[TextContent]:
    from lib.search.hybrid_search import HybridSearcher
    from lib.search.jit_retrieval import JITRetriever
    from lib.bridgemine.gap_detection import GapDetector

    question = args["question"]
    max_rounds = args.get("max_rounds", 3)
    domain = args.get("domain")

    findings = []
    all_passages = []

    for round_num in range(max_rounds):
        # Search
        retriever = JITRetriever()
        result = retriever.retrieve(question, n_passages=15)
        all_passages.extend(result.passages)

        # Identify what's missing
        if domain:
            detector = GapDetector()
            gaps = detector.find_gaps(domain, limit=5)

            # Refine question based on gaps
            if gaps.candidates:
                gap = gaps.candidates[0]
                question = f"{question} AND {gap.method_name}"

        findings.append({
            "round": round_num + 1,
            "query": question,
            "synthesis": result.synthesis,
            "sources": len(result.passages),
        })

    # Final synthesis across all rounds
    return format_research_brief(findings, all_passages)
```

**Effort:** 3 hours

---

### Enhancement 2: Progress Streaming

For long operations, add SSE-style progress updates:

```python
# In MCP handler
async def handle_ingest_pdf_with_progress(args: dict):
    pdf_path = Path(args["pdf_path"])

    # Stream progress
    yield TextContent(type="text", text="[1/5] Parsing PDF...")
    parsed = await parse_pdf(pdf_path)

    yield TextContent(type="text", text=f"[2/5] Chunking {len(parsed.pages)} pages...")
    chunks = await chunk_text(parsed)

    yield TextContent(type="text", text=f"[3/5] Embedding {len(chunks)} passages...")
    # ... etc
```

**Effort:** 2 hours (requires MCP streaming support)

---

## Implementation Order

### Week 1: Quick Wins + BridgeMine Core

| Day | Task | Track | Effort |
|-----|------|-------|--------|
| 1 | Fix connection pool | A | 15 min |
| 1 | Add OpenAlex to NoveltyChecker | B | 1 hour |
| 1 | Update claim extraction prompt | B | 15 min |
| 2 | Add trivial gap filtering | B | 30 min |
| 2 | Add telemetry module (UPGRADE_ASSESSMENT P0) | A | 2 hours |
| 3 | Add spatial transcriptomics hierarchy | B | 2 hours |

### Week 2: Scale + Agent

| Day | Task | Track | Effort |
|-----|------|-------|--------|
| 1 | Add ingest_jobs table for crash recovery | A | 1 hour |
| 2 | Create Neo4j CSV export script | A | 2 hours |
| 3 | Implement research_deep_dive tool | C | 3 hours |
| 4 | Add evalset framework | A | 2 hours |

---

## Metrics for Success

After implementation, verify:

1. **BridgeMine Quality:**
   ```bash
   polymath gaps "spatial_transcriptomics" --limit 10
   # Should return <3 trivial suggestions
   # Should include 1+ novel method transfer
   ```

2. **Infrastructure:**
   ```bash
   time python scripts/reingest_archive.py --workers 12 --limit 100
   # Should complete without connection pool blocking
   ```

3. **MCP Agent:**
   ```
   research_deep_dive("What methods predict spatial gene expression from H&E?")
   # Should perform 3 search rounds
   # Should cite 10+ papers
   # Should identify 2+ research gaps
   ```

---

## Bottom Line

**For your MD-PhD work:**

1. **BridgeMine is your competitive advantage** - No other tool finds cross-domain research gaps. Invest here first.

2. **Infrastructure is table stakes** - The pool fix takes 15 minutes and prevents production headaches.

3. **MCP agent is quality-of-life** - Nice to have for daily research, but not blocking.

**Start with:**
```bash
# 1. Pool fix (15 min)
# 2. OpenAlex integration (1 hour)
# 3. Spatial hierarchy (2 hours)
```

These 3 changes will make BridgeMine 10x more useful for finding Img2ST research opportunities.
