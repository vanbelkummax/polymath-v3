# Polymath v3 - Reconstruction Guide

## Quick Start

```bash
# 1. Reset database
psql -U polymath -c "DROP DATABASE IF EXISTS polymath_v3; CREATE DATABASE polymath_v3;"
psql -U polymath -d polymath_v3 -f schema/postgres/001_core.sql
psql -U polymath -d polymath_v3 -f schema/postgres/002_concepts.sql
psql -U polymath -d polymath_v3 -f schema/postgres/005_soft_delete.sql

# 2. Ingest from Zotero CSVs (prototype first)
python scripts/ingest_from_zotero.py --csv /path/to/zotero.csv --limit 200

# 3. Full batch via GCP
python scripts/ingest_from_zotero.py --csv /path/to/zotero.csv --batch
```

---

## Data Sources

| Source | Location | Count |
|--------|----------|-------|
| Zotero CSV 1 | `/mnt/c/Users/User/Downloads/Polymath_Full_.csv` | 3,175 |
| Zotero CSV 2 | `/mnt/c/Users/User/Downloads/polymath2_.csv` | 1,124 |
| PDF Storage | `/mnt/c/Users/User/Zotero/storage/` | WSL path |

**CSV Columns**: Key, Title, Author, DOI, Abstract Note, File Attachments

---

## Architecture

```
Zotero CSV → IngestPipeline → Postgres (pgvector) → Neo4j (concepts)
                  ↓
         GCP Batch API (concepts)
```

| Component | Purpose |
|-----------|---------|
| `lib/ingest/pipeline.py` | PDF → chunks → embeddings → DB |
| `lib/ingest/metadata.py` | Zotero/CrossRef resolution |
| `lib/embeddings/bge_m3.py` | BGE-M3 1024-dim vectors |
| `batch/concept_extraction.py` | GCP Batch for Gemini |
| `scripts/sync_neo4j.py` | Postgres → Neo4j sync |

---

## GCP Batch Configuration

```bash
export GCP_PROJECT_ID="fifth-branch-483806-m1"
export GCP_BUCKET="gs://polymath-batch-jobs"
export GEMINI_API_KEY="your-key"
```

**Batch Concept Extraction:**
```bash
# Submit async batch job (50% cheaper)
python batch/submit_concept_job.py --input passages.jsonl

# Check status
python batch/submit_concept_job.py --status JOB_ID

# Process results
python batch/process_batch_results.py --job JOB_ID
```

---

## Ingestion Commands

### Prototype (local, 200 PDFs)
```bash
python scripts/ingest_from_zotero.py \
  --csv '/mnt/c/Users/User/Downloads/Polymath_Full_.csv' \
  --limit 200 \
  --workers 4
```

### Full Batch (GCP)
```bash
python scripts/ingest_from_zotero.py \
  --csv '/mnt/c/Users/User/Downloads/Polymath_Full_.csv' \
  --batch \
  --skip-concepts  # Extract concepts via GCP batch later
```

### Concept Backfill
```bash
# Export passages needing concepts
psql -U polymath -d polymath_v3 -c "
  SELECT passage_id, passage_text
  FROM passages p
  WHERE NOT EXISTS (
    SELECT 1 FROM passage_concepts pc WHERE pc.passage_id = p.passage_id
  )
" -t -A -F $'\t' > missing_concepts.tsv

# Submit to GCP Batch
python batch/submit_concept_job.py --input missing_concepts.tsv
```

---

## Database Schema (Essential)

```sql
-- Documents
CREATE TABLE documents (
  doc_id UUID PRIMARY KEY,
  title TEXT NOT NULL,
  title_hash VARCHAR(64) UNIQUE,
  doi VARCHAR(100) UNIQUE,
  authors TEXT[],
  year INTEGER,
  zotero_key VARCHAR(20)
);

-- Passages with vectors
CREATE TABLE passages (
  passage_id UUID PRIMARY KEY,
  doc_id UUID REFERENCES documents,
  passage_text TEXT,
  embedding vector(1024),
  is_superseded BOOLEAN DEFAULT FALSE
);

-- Concepts
CREATE TABLE passage_concepts (
  passage_id UUID REFERENCES passages,
  concept_name VARCHAR(200),
  concept_type VARCHAR(50),
  confidence REAL
);
```

---

## Validation

```bash
# Check counts
psql -U polymath -d polymath_v3 -c "
  SELECT 'documents' as tbl, count(*) FROM documents
  UNION ALL
  SELECT 'passages', count(*) FROM passages
  UNION ALL
  SELECT 'concepts', count(*) FROM passage_concepts;
"

# Test search
python -c "
from lib.search.hybrid_search import HybridSearcher
s = HybridSearcher()
r = s.search('spatial transcriptomics', n=5)
for x in r.results: print(f'{x.score:.3f} {x.title[:50]}')
"

# Run eval
python scripts/run_evaluation.py --eval-set data/eval_sets/core.jsonl
```

---

## Neo4j Sync

```bash
# Full sync after ingestion
python scripts/sync_neo4j.py --full

# Incremental (new docs only)
python scripts/sync_neo4j.py --incremental
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| PDF not found | Check WSL path: `/mnt/c/Users/User/Zotero/storage/` |
| OOM on embeddings | Reduce batch size: `BATCH_SIZE=50` |
| GCP auth fail | `gcloud auth application-default login` |
| Neo4j connection | `docker start neo4j` |

---

## Environment

```bash
# Required in .env
POSTGRES_DSN=dbname=polymath_v3 user=polymath host=/var/run/postgresql
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=polymathic2026
GEMINI_API_KEY=your-key
GCP_PROJECT_ID=fifth-branch-483806-m1
```
