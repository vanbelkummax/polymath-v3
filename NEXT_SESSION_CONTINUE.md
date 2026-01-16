# Polymath v3 Prototype Validation - Continue Session

## Current State (2026-01-15)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: PDF Quality | ✅ Complete | 100% clean extraction |
| Phase 2: Ingest PDFs | ✅ Bug fixed, ready | Single PDF verified working |
| Phase 3: GCP Concepts | Pending | |
| Phase 4: Neo4j Sync | Pending | |
| Phase 5: Search Eval | Pending | |

## Bugs Fixed This Session

| Issue | Fix |
|-------|-----|
| Numpy array truth value (line 327) | `if embeddings` → `if (embeddings is not None and len(embeddings) > i)` |
| Missing char_start/char_end columns | Added to passages table |
| Missing embedding vector column | Added pgvector(1024) |
| page_char_start NOT NULL | Made nullable |

**Verified:** Single PDF ingestion works - 8 passages with 1024-dim BGE-M3 embeddings in 9.3s

## Continue: Run Batch Ingestion

**Using main `polymath` database** (schema already migrated):

```bash
cd /home/user/polymath-v3
export POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql"
export NEO4J_PASSWORD="polymathic2026"

# Ingest 200 PDFs (skip concepts for GCP batch)
python scripts/ingest_from_zotero.py \
  --csv '/mnt/c/Users/User/Downloads/Polymath_Full_.csv' \
  --csv '/mnt/c/Users/User/Downloads/polymath2_.csv' \
  --limit 200 --skip-concepts --workers 4

# Verify counts
psql -U polymath -d polymath -c "
  SELECT 'documents', count(*) FROM documents WHERE ingest_batch LIKE 'ingest_2026%'
  UNION ALL SELECT 'passages', count(*) FROM passages WHERE doc_id IN (
    SELECT doc_id FROM documents WHERE ingest_batch LIKE 'ingest_2026%'
  );
"
```

## Then: Phase 3 - GCP Batch Concepts

```bash
python scripts/batch_concept_extraction_async.py --limit 50 --wait
python scripts/process_batch_results.py --latest
```

## Then: Phase 4 - Neo4j Sync

```bash
docker start neo4j
python scripts/sync_neo4j.py --full
```

## Then: Phase 5 - Search Evaluation

Test 5 queries, verify <2s latency with reranking.

## Success Criteria

| Metric | Target |
|--------|--------|
| Documents ingested | 150-200 |
| Passages created | 2000-4000 |
| Embeddings | 100% coverage |
| Concepts extracted | >100 |
| Search latency | <2s with rerank |
