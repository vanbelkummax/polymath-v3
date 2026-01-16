# Polymath v3 - Production Pipeline Guide

## Status (2026-01-16)

Pipeline validated and in production. Using main `polymath` database with migrated schema.

**IMPORTANT:** This is the canonical source of truth for Polymath v3 code. Do NOT use `/home/user/polymath-repo/` for v3 features.

---

## Key Directories

| Path | Purpose | Status |
|------|---------|--------|
| `/home/user/polymath-v3/` | **CANONICAL v3 CODE** | Active |
| `/home/user/polymath-repo/` | Legacy v2 code | Reference only |
| `/home/user/work/polymax/` | Prototyping/scratch | May be outdated |

---

## Data Sources

| Source | Path | Count |
|--------|------|-------|
| Zotero CSV 1 | `/mnt/c/Users/User/Downloads/Polymath_Full_.csv` | ~3,000 |
| Zotero CSV 2 | `/mnt/c/Users/User/Downloads/polymath2_.csv` | ~1,100 |
| PDF Storage | `/mnt/c/Users/User/Zotero/storage/` | WSL mount |
| PDFs on disk | ~1,250 available | |

---

## AI Models

| Stage | Model | Location | Cost |
|-------|-------|----------|------|
| Embeddings | BGE-M3 (1024-dim) | Local GPU | $0 |
| Concept Extraction | gemini-2.0-flash | Gemini API | ~$0.00001/passage |
| Reranking | bge-reranker-v2-m3 | Local GPU | $0 |

---

## Quick Start: Ingest PDFs

```bash
cd /home/user/polymath-v3
export POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql"
export NEO4J_PASSWORD="polymathic2026"

# Dry run first
python scripts/ingest_from_zotero.py \
  --csv '/mnt/c/Users/User/Downloads/Polymath_Full_.csv' \
  --csv '/mnt/c/Users/User/Downloads/polymath2_.csv' \
  --limit 200 --dry-run

# Ingest (skip concepts for batch later)
python scripts/ingest_from_zotero.py \
  --csv '/mnt/c/Users/User/Downloads/Polymath_Full_.csv' \
  --csv '/mnt/c/Users/User/Downloads/polymath2_.csv' \
  --limit 200 --skip-concepts --workers 4

# Verify
psql -U polymath -d polymath -c "
  SELECT ingest_batch, count(*) FROM documents
  WHERE ingest_batch IS NOT NULL
  GROUP BY ingest_batch ORDER BY ingest_batch DESC LIMIT 5;
"
```

**Performance:** ~9-15 seconds per PDF, ~30 passages average

---

## Concept Extraction

```bash
cd /home/user/polymath-v3
export POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql"

# Targeted extraction for a specific batch
python scripts/extract_new_batch.py

# Or use backfill for all passages without concepts
python scripts/backfill_concepts.py --limit 500 --batch-size 30

# Verify
psql -U polymath -d polymath -c "
  SELECT concept_type, count(*)
  FROM passage_concepts
  WHERE extractor_version = 'v3.0'
  GROUP BY concept_type;
"
```

---

## Neo4j Sync

```bash
docker start neo4j
python scripts/sync_neo4j.py --incremental  # or --full

cypher-shell -u neo4j -p polymathic2026 "
  MATCH (p:Passage) RETURN 'passages', count(p)
  UNION ALL MATCH (c:Concept) RETURN 'concepts', count(c);
"
```

---

## Search

```bash
python -c "
from lib.search.hybrid_search import HybridSearcher
s = HybridSearcher()
r = s.search('spatial transcriptomics', n=5, rerank=True)
for x in r.results: print(f'{x.score:.3f} {x.title[:50]}')
"
```

---

## Architecture

```
Zotero CSV → IngestPipeline → Postgres (pgvector) → Neo4j
                  ↓                    ↓
            BGE-M3 (local)      Gemini API (concepts)
```

| Component | File | Purpose |
|-----------|------|---------|
| Ingestion | `scripts/ingest_from_zotero.py` | Zotero CSV → PDFs → DB |
| Pipeline | `lib/ingest/pipeline.py` | PDF → chunks → embeddings |
| Concept Extraction | `lib/ingest/concept_extractor.py` | Text → concepts via Gemini |
| Embeddings | `lib/embeddings/bge_m3.py` | BGE-M3 1024-dim |
| Search | `lib/search/hybrid_search.py` | Vector + FTS + Graph |
| Neo4j Sync | `scripts/sync_neo4j.py` | Postgres → Neo4j |

---

## Environment

```bash
# Required
export POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_PASSWORD="polymathic2026"
export GCP_PROJECT_ID="fifth-branch-483806-m1"
```

---

## Known Issues & Fixes (2026-01-16)

### Fixed Issues

| Issue | Root Cause | Fix |
|-------|------------|-----|
| Numpy array truth value | `if embeddings` fails for numpy arrays | Use `if (embeddings is not None and len(embeddings) > i)` |
| Schema mismatch | Main DB uses `page_char_start` vs v3 uses `char_start` | Added new columns to passages table |
| Missing pgvector column | No `embedding vector(1024)` column | Added via ALTER TABLE |
| ON CONFLICT mismatch | Wrong columns in conflict clause | Match PK `(passage_id, concept_name, extractor_version)` |
| API truncation | `max_output_tokens: 1000` too low | **Set to 4096** in concept_extractor.py |

### Key Learnings

1. **Test with production data early** - Truncation only appeared with real passages
2. **Check API finish reasons** - `FinishReason.MAX_TOKENS` immediately reveals truncation
3. **Schema migrations need testing** - Multiple column mismatches between v2 and v3
4. **Backfill scripts should filter by batch** - Avoid processing old non-scientific passages

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| PDF not found | Check `/mnt/c/Users/User/Zotero/storage/` mount |
| OOM embeddings | Set `--workers 2` or reduce batch size |
| GCP auth fail | `gcloud auth application-default login` |
| Neo4j down | `docker start neo4j` |
| Slow /mnt/ | Copy PDFs to `/home/user/work/` first |
| Concept extraction fails | Check `max_output_tokens` is 4096 |
| "No JSON found" | API response truncated - increase token limit |

---

## Database Schema (Main polymath DB)

```sql
-- Key tables
documents (doc_id, title, authors, year, doi, ingest_batch, ...)
passages (passage_id, doc_id, passage_text, char_start, char_end, embedding vector(1024), ...)
passage_concepts (passage_id, concept_name, concept_type, confidence, extractor_version, ...)
```

---

## Validation Checklist

| Check | Command | Target |
|-------|---------|--------|
| New docs | `SELECT count(*) FROM documents WHERE ingest_batch LIKE 'zotero_ingest_%'` | 38+ |
| Passages | `SELECT count(*) FROM passages WHERE embedding IS NOT NULL` | 1260+ |
| Concepts | `SELECT count(*) FROM passage_concepts WHERE extractor_version = 'v3.0'` | 300+ |
| Search | Test query with rerank | <2s latency |
