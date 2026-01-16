# Polymath v3 Prototype Validation - Session Prompt

## Context
Polymath v3 is a scientific knowledge base. We need to validate the pipeline with 200 PDFs before scaling to full 4300 PDF library.

## What To Do

Execute this 5-phase prototype validation:

### Phase 1: PDF Quality (10 min)
Create `scripts/validate_pdf_quality.py` and test 10 PDFs from Zotero storage to verify PyMuPDF extracts clean text (not OCR garbage).

### Phase 2: Ingest 200 PDFs (45 min)
```bash
# Create prototype DB
psql -U polymath -c "DROP DATABASE IF EXISTS polymath_prototype; CREATE DATABASE polymath_prototype;"
psql -U polymath -d polymath_prototype -f schema/postgres/001_core.sql
psql -U polymath -d polymath_prototype -f schema/postgres/002_concepts.sql

export POSTGRES_DSN="dbname=polymath_prototype user=polymath host=/var/run/postgresql"

python scripts/ingest_from_zotero.py \
  --csv '/mnt/c/Users/User/Downloads/Polymath_Full_.csv' \
  --csv '/mnt/c/Users/User/Downloads/polymath2_.csv' \
  --limit 200 --skip-concepts
```

### Phase 3: GCP Batch Concepts (20 min)
```bash
python scripts/batch_concept_extraction_async.py --limit 50 --wait
python scripts/process_batch_results.py --latest
```

### Phase 4: Neo4j Sync (15 min)
```bash
docker start neo4j
python scripts/sync_neo4j.py --full
```

### Phase 5: Search Evaluation (15 min)
Test 5 queries, check latency <2s with reranking.

---

## AI Models & Costs

| Model | Usage | Cost |
|-------|-------|------|
| BGE-M3 (local) | Embeddings | FREE |
| Gemini 2.5 Flash Lite | Concepts | **$0.03** (200 PDFs) / **$0.56** (full 4300) |
| Cross-encoder (local) | Reranking | FREE |

---

## Key Files
- `/home/user/polymath-v3/scripts/ingest_from_zotero.py` - Zotero ingestion
- `/home/user/polymath-v3/scripts/batch_concept_extraction_async.py` - GCP batch
- `/home/user/polymath-v3/scripts/sync_neo4j.py` - Neo4j sync
- `/home/user/polymath-v3/lib/search/hybrid_search.py` - Search

## Success Criteria
1. >90% PDFs extract clean text
2. 200 docs, ~2000 passages in Postgres
3. GCP batch succeeds, concepts stored
4. Neo4j synced with graph data
5. Search returns results <2s

## After Validation
Update `/home/user/polymath-v3/CLAUDE.md` with lean prototype guide reflecting actual workflow.
