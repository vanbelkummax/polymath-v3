# Polymath v3 - Production Pipeline Guide

## ACTIVE TASK (2026-01-17)

**Execute Plan:** `/home/user/polymath-v3/docs/plans/2026-01-17-batch-v1-integration.md`

**Goal:** Integrate 5,954 concepts from 973 passages (batch-v1 trial) into Postgres embeddings + Neo4j graph.

**Current State:**
| Component | Status | Count |
|-----------|--------|-------|
| Postgres: passage_concepts | ✅ Done | 5,954 concepts, 973 passages |
| Postgres: embeddings | ❌ Missing | 0/973 have embeddings |
| Neo4j: concepts | ❌ Missing | 0 batch-v1 concepts synced |
| Neo4j: MENTIONS | ❌ Missing | 0 relationships |

**Required Skills:**
- `superpowers:executing-plans` - Follow the plan task-by-task
- `superpowers:subagent-driven-development` - Dispatch subagents per task

---

## Key Directories

| Path | Purpose |
|------|---------|
| `/home/user/polymath-v3/` | **CANONICAL v3 CODE** |
| `/home/user/polymath-repo/` | Batch API scripts, legacy v2 |

---

## Environment Setup

```bash
export POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_PASSWORD="polymathic2026"
export GOOGLE_APPLICATION_CREDENTIALS="/home/user/.gcp/service-account.json"
```

---

## AI Models & Costs

| Stage | Model | Cost |
|-------|-------|------|
| Embeddings | BGE-M3 (1024-dim, local) | $0 |
| Concepts (batch API) | gemini-2.5-flash-lite | ~$0.00015/passage |
| Reranking | bge-reranker-v2-m3 (local) | $0 |

---

## Architecture

```
Zotero CSV → IngestPipeline → Postgres (pgvector) → Neo4j
                  ↓                    ↓
            BGE-M3 (local)      Gemini API (concepts)
```

| Component | File |
|-----------|------|
| Embeddings | `lib/embeddings/bge_m3.py` |
| Concepts | `lib/ingest/concept_extractor.py` |
| Batch API | `/home/user/polymath-repo/scripts/batch_concept_extraction_async.py` |
| Neo4j Sync | `scripts/sync_neo4j.py` |

---

## Database Schema

```sql
documents (doc_id, title, authors, year, doi, ingest_batch)
passages (passage_id, doc_id, passage_text, embedding vector(1024))
passage_concepts (passage_id, concept_name, concept_type, confidence, extractor_version)
```

---

## Verification Commands

```bash
# Check batch-v1 status
psql -U polymath -d polymath -c "
SELECT
  count(*) as concepts,
  count(DISTINCT pc.passage_id) as passages,
  (SELECT count(*) FROM passages p
   JOIN passage_concepts pc2 ON p.passage_id = pc2.passage_id
   WHERE pc2.extractor_version = 'batch-v1' AND p.embedding IS NOT NULL) as with_embeddings
FROM passage_concepts pc
WHERE pc.extractor_version = 'batch-v1';
"

# Check Neo4j batch-v1
docker exec polymax-neo4j cypher-shell -u neo4j -p polymathic2026 "
MATCH (p:Passage)-[r:MENTIONS]->(c:Concept)
WHERE r.synced_at > datetime() - duration('P1D')
RETURN count(DISTINCT p) as passages, count(r) as mentions
"
```

---

## Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| API truncation | `max_output_tokens: 16384` in concept_extractor.py |
| Neo4j full sync slow | Use targeted `sync_neo4j_batch.py` for specific batches |
| Slow /mnt/ access | Copy PDFs to `/home/user/work/` first |
