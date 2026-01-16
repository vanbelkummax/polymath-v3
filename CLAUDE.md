# Polymath v3 - Production Pipeline Guide

## Active Jobs (2026-01-16)

**GCP Batch Concept Extraction:**
- Job ID: `7102301956688838656`
- Started: 2026-01-16 15:06 UTC
- Passages: 1,000
- Model: gemini-2.5-flash-lite
- Status check: `python /home/user/polymath-repo/scripts/batch_concept_extraction_async.py --status`
- Process results: `python /home/user/polymath-repo/scripts/process_batch_results.py --latest`

---

## Key Directories

| Path | Purpose |
|------|---------|
| `/home/user/polymath-v3/` | **CANONICAL v3 CODE** |
| `/home/user/polymath-repo/` | Batch API scripts, legacy v2 |

---

## AI Models & Costs

| Stage | Model | Cost |
|-------|-------|------|
| Embeddings | BGE-M3 (1024-dim, local) | $0 |
| Concepts (real-time) | gemini-2.5-flash | ~$0.0003/passage |
| Concepts (batch API) | gemini-2.5-flash-lite | ~$0.00015/passage |
| Reranking | bge-reranker-v2-m3 (local) | $0 |

**Use batch API for >100 passages** (50% cheaper, ~30min latency).

---

## Quick Commands

```bash
# Environment
export POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql"
export NEO4J_PASSWORD="polymathic2026"
export GOOGLE_APPLICATION_CREDENTIALS="/home/user/.gcp/service-account.json"

# Ingest PDFs
cd /home/user/polymath-v3
python scripts/ingest_from_zotero.py \
  --csv '/mnt/c/Users/User/Downloads/Polymath_Full_.csv' \
  --csv '/mnt/c/Users/User/Downloads/polymath2_.csv' \
  --limit 200 --skip-concepts --workers 4

# Batch concept extraction (recommended)
cd /home/user/polymath-repo
python scripts/batch_concept_extraction_async.py --limit 1000
python scripts/batch_concept_extraction_async.py --status
python scripts/process_batch_results.py --latest

# Real-time concept extraction (if <100 passages)
cd /home/user/polymath-v3
python scripts/extract_new_batch.py

# Neo4j sync
docker start neo4j
python scripts/sync_neo4j.py --incremental

# Verify counts
psql -U polymath -d polymath -c "
  SELECT 'documents' as t, count(*) FROM documents
  UNION ALL SELECT 'passages', count(*) FROM passages WHERE embedding IS NOT NULL
  UNION ALL SELECT 'concepts', count(*) FROM passage_concepts;
"
```

---

## Architecture

```
Zotero CSV → IngestPipeline → Postgres (pgvector) → Neo4j
                  ↓                    ↓
            BGE-M3 (local)      Gemini API (concepts)
```

| Component | File |
|-----------|------|
| Ingestion | `scripts/ingest_from_zotero.py` |
| Pipeline | `lib/ingest/pipeline.py` |
| Concepts | `lib/ingest/concept_extractor.py` |
| Batch API | `/home/user/polymath-repo/scripts/batch_concept_extraction_async.py` |
| Search | `lib/search/hybrid_search.py` |
| Neo4j Sync | `scripts/sync_neo4j.py` |

---

## GCP Config

| Resource | Value |
|----------|-------|
| Project | `fifth-branch-483806-m1` |
| Bucket | `gs://polymath-batch-jobs` |
| Credentials | `/home/user/.gcp/service-account.json` |

---

## Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| API truncation ("No JSON found") | `max_output_tokens: 16384` in concept_extractor.py |
| Numpy array truth value | Use `if (embeddings is not None and len(embeddings) > i)` |
| ON CONFLICT mismatch | PK is `(passage_id, concept_name, extractor_version)` |
| Slow /mnt/ access | Copy PDFs to `/home/user/work/` first |
| Neo4j down | `docker start neo4j` |

---

## Database Schema

```sql
documents (doc_id, title, authors, year, doi, ingest_batch)
passages (passage_id, doc_id, passage_text, embedding vector(1024))
passage_concepts (passage_id, concept_name, concept_type, confidence, extractor_version)
```
