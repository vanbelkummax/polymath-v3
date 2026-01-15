# Polymath v3 Upgrade Assessment

## Current Architecture Summary

**Polymath v3 already has:**
- **Hybrid Search**: Vector (pgvector HNSW) + FTS (tsvector) + Neo4j graph, fused via RRF
- **Neural Reranking**: BAAI/bge-reranker-v2-m3
- **Structure-Aware Chunking**: Markdown header splitting, stores `section`, `parent_section`, `section_level`, `char_start`, `char_end`
- **JIT Retrieval**: GAM-style query-time synthesis with citation markers `[1]`, `[2]`, etc.
- **Hallucination Detection**: 3-stage claim extraction → evidence retrieval → verification
- **Batch Tracking**: `ingest_batches`, `gcp_batch_jobs`, `audit_log` tables
- **Tests Directory**: Exists but empty (no test framework)

---

## Proposal Assessment

### P0 - Evaluation + Observability

| Item | Worth It? | Rationale |
|------|-----------|-----------|
| **Retrieval Telemetry (JSONL)** | **YES** | No per-query logging exists. Critical for debugging. Quick win. |
| **Postgres retrieval_runs table** | SKIP | JSONL is sufficient initially. Add DB storage later if needed. |
| **Evalset Format (JSONL)** | **YES** | No evaluation framework exists. Essential for measuring improvements. |
| **Interactive Labeling CLI** | **YES** | Enables building gold-standard evalset from real queries. |
| **Eval Runner + Metrics** | **YES** | Recall@K, MRR, nDCG@10 are standard. Must have. |
| **Regression Tests** | **YES** | Tests directory is empty. Basic quality assurance. |

**P0 Verdict: Implement all telemetry + eval infrastructure.**

---

### P1 - Hierarchical Chunking + Multi-Granularity Retrieval

| Item | Worth It? | Rationale |
|------|-----------|-----------|
| **Sentence-level chunking** | **NO** | Current paragraph-level chunks with header context work well for scientific text. Sentence-level fragments lose coherence. Would require re-ingesting 750K+ passages. |
| **passage_units table** | **NO** | Current `passages` table already stores `char_start`, `char_end`, `section`, `parent_section`. Adding a parallel table creates sync complexity. |
| **Multi-granularity retrieval** | **NO** | The proposal's "retrieve sentences, expand to paragraphs" essentially recreates what we already have. The context assembly in JIT retrieval already builds coherent context from paragraph-level hits. |
| **Context expansion around hits** | **MAYBE LATER** | Could add ±1 adjacent passage retrieval, but current approach is working. |

**P1 Verdict: Do NOT implement hierarchical multi-granularity retrieval.** The ROI is unclear and requires massive re-ingestion. Current paragraph-level chunking with header context is appropriate for scientific papers.

---

### P1 - Evidence Gate

| Item | Worth It? | Rationale |
|------|-----------|-----------|
| **Hard citation requirement** | **YES (modified)** | Valuable for reducing hallucination. However, "delete uncited sentences" is too aggressive. Better approach: **flag uncited claims** rather than delete. |
| **"Insufficient evidence" fallback** | **YES** | Important for intellectual honesty when KB lacks coverage. |

**Evidence Gate Verdict: Implement a softer version that flags rather than deletes.**

---

### P1 - Concept Canonicalization

| Item | Worth It? | Rationale |
|------|-----------|-----------|
| **Alias normalization table** | **MAYBE LATER** | Nice to have, but not urgent unless graph quality is demonstrably poor. Current approach stores concepts with provenance. |
| **Similarity-based merging** | **NO** | Risky. Can merge distinct concepts. Better to leave separate and query both. |

**Canonicalization Verdict: Low priority. Revisit after eval framework reveals problems.**

---

## Recommended Implementation Plan

### Sprint 1: Observability + Eval Foundation (P0) ✅ IMPLEMENT

1. **Retrieval Telemetry Module** (`lib/telemetry.py`)
   - Emit JSONL per query with: run_id, query, vector/fts/graph results, fused ranks, reranker scores, latency
   - Environment flag: `POLYMATH_TELEMETRY=1`

2. **Evalset Format + Runner**
   - `data/evalsets/core_eval.jsonl` - JSONL with query, relevant_passage_ids
   - `scripts/eval/run_eval.py` - Compute Recall@K, MRR, nDCG@10
   - `scripts/eval/label_evalset.py` - Interactive labeling CLI

3. **Regression Tests**
   - `tests/test_retrieval_regression.py` - pytest-based sanity checks
   - 5-10 queries that MUST retrieve known passages

### Sprint 2: Evidence Gating (P1, modified) ⚠️ IMPLEMENT CAUTIOUSLY

4. **Evidence Gate** (`lib/validation/evidence_gate.py`)
   - Flag uncited sentences (don't delete)
   - Return `evidence_coverage` score
   - "Insufficient evidence" fallback when coverage < 30%

### NOT Implementing

- ❌ Hierarchical multi-granularity chunking
- ❌ passage_units table
- ❌ Sentence-level retrieval + expansion
- ❌ Concept alias tables (for now)

---

## Schema Changes Required

**For Sprint 1 (minimal):**
```sql
-- Optional: Store telemetry in DB (skip initially, JSONL is fine)
CREATE TABLE IF NOT EXISTS retrieval_telemetry (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON retrieval_telemetry (created_at);
CREATE INDEX ON retrieval_telemetry (query_hash);
```

No schema changes required if using JSONL only.

---

## Files to Create

```
polymath-v3/
├── lib/
│   └── telemetry.py           # NEW: Retrieval logging
├── data/
│   └── evalsets/
│       └── core_eval.jsonl    # NEW: Starter evalset
├── scripts/
│   └── eval/
│       ├── run_eval.py        # NEW: Metrics runner
│       └── label_evalset.py   # NEW: Interactive labeler
├── tests/
│   ├── conftest.py            # NEW: pytest fixtures
│   └── test_retrieval.py      # NEW: Regression tests
└── logs/
    └── retrieval_runs.jsonl   # AUTO-CREATED: Telemetry output
```

---

## Success Criteria

1. ✅ `python scripts/eval/run_eval.py --evalset data/evalsets/core_eval.jsonl` runs and outputs metrics
2. ✅ `pytest tests/` passes with 5+ regression queries
3. ✅ `POLYMATH_TELEMETRY=1 polymath search "query"` writes to `logs/retrieval_runs.jsonl`
4. ✅ JIT retrieval returns `evidence_coverage` score

---

## Bottom Line

**The proposal is well-intentioned but over-scoped.** The hierarchical multi-granularity retrieval section (50% of the work) solves a problem we don't clearly have. Current paragraph-level chunking with header context is appropriate for scientific papers.

**Implement P0 (telemetry + eval) immediately.** This gives us the measurement infrastructure to determine if P1 features are actually needed.

**Implement evidence gating cautiously** with flagging rather than deletion.

**Skip hierarchical chunking** unless eval metrics reveal retrieval quality problems that can't be solved by prompt engineering or reranker improvements.
