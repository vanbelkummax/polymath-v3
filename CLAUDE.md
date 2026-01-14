# Claude Code Configuration for Polymath v3

## Quick Reference

| Component | Location | Purpose |
|-----------|----------|---------|
| **Code** | `/home/user/polymath-v3/` | Source of truth |
| **Postgres** | `postgresql://polymath@localhost/polymath_v3` | Documents, passages, vectors |
| **Neo4j** | `bolt://localhost:7687` (neo4j/polymathic2026) | Concept graph |
| **Config** | `lib/config.py` | Centralized configuration |
| **Env** | `.env` | Secrets (not in git) |

---

## Architecture Overview

```
polymath-v3/
├── lib/
│   ├── config.py           # Centralized config
│   ├── db/                  # Database connections
│   │   ├── postgres.py     # pgvector + pooling
│   │   └── neo4j.py        # Graph database
│   ├── embeddings/         # BGE-M3 1024-dim
│   ├── ingest/             # PDF → passages → DB
│   │   ├── pipeline.py     # Main orchestrator
│   │   ├── pdf_parser.py   # PyMuPDF extraction
│   │   ├── chunking.py     # Markdown-aware splitting
│   │   ├── metadata.py     # Zotero → CrossRef resolution
│   │   └── concept_extractor.py  # Gemini API
│   ├── search/             # Retrieval
│   │   ├── hybrid_search.py  # Vector + FTS + Graph + RRF
│   │   ├── jit_retrieval.py  # Query-time synthesis
│   │   └── reranker.py       # Neural reranking
│   ├── validation/         # Quality
│   │   └── hallucination.py  # 3-stage detection
│   └── bridgemine/         # Discovery
│       ├── gap_detection.py   # Method-problem gaps
│       └── novelty_check.py   # PubMed/S2 validation
├── mcp/
│   └── polymath_server.py  # MCP server
├── cli/
│   └── main.py             # Click CLI
├── scripts/
│   ├── reingest_archive.py
│   ├── backfill_concepts.py
│   └── sync_neo4j.py
└── batch/                  # GCP Batch jobs
```

---

## Common Commands

### Search

```bash
# CLI search
polymath search "spatial transcriptomics" -n 20 --rerank

# Python
from lib.search.hybrid_search import HybridSearcher
searcher = HybridSearcher()
response = searcher.search("query", n=20, rerank=True)
```

### Ingestion

```bash
# Single paper
polymath ingest /path/to/paper.pdf

# Batch
polymath ingest-batch /path/to/pdfs/ --workers 4

# Python
from lib.ingest.pipeline import IngestPipeline
pipeline = IngestPipeline()
result = pipeline.ingest_pdf(Path("/path/to/paper.pdf"))
```

### Validation

```bash
# Verify claim
polymath verify "BERT uses bidirectional attention"

# Python
from lib.validation.hallucination import verify_claim
result = verify_claim("claim text")
```

### BridgeMine

```bash
# Find gaps
polymath gaps "spatial_transcriptomics" --check-novelty

# Python
from lib.bridgemine.gap_detection import GapDetector
detector = GapDetector()
result = detector.find_gaps("spatial_transcriptomics")
```

---

## Database Schemas

### PostgreSQL (pgvector)

```sql
-- Core tables
documents (doc_id UUID, title, doi, pmid, arxiv_id, authors[], year, venue)
passages (passage_id UUID, doc_id FK, passage_text, section, embedding vector(1024))
passage_concepts (passage_id, concept_name, concept_type, confidence)

-- Indexes
idx_passage_embedding USING hnsw (embedding vector_cosine_ops)
idx_passage_search USING gin (search_vector)
```

### Neo4j

```cypher
// Node types
(:Paper {doc_id, title, year, doi})
(:Passage {passage_id, doc_id, section})
(:METHOD {name})
(:PROBLEM {name})
(:DOMAIN {name})

// Relationships
(p:Passage)-[:FROM_PAPER]->(paper:Paper)
(p:Passage)-[:MENTIONS {confidence}]->(c:METHOD)
(m1:METHOD)-[:SIMILAR_TO {score}]-(m2:METHOD)
```

---

## Metadata Resolution Priority

1. **Zotero CSV** (confidence > 0.95) - Fast, high coverage
2. **pdf2doi** - Extract DOI from PDF binary
3. **CrossRef API** - Authoritative for DOIs
4. **arXiv API** - For preprints
5. **Zotero relaxed** (confidence > 0.80)
6. **Filename parsing** - Last resort

---

## Key Design Decisions

### pgvector over ChromaDB
- Single source of truth (ACID)
- No sync issues
- Better SQL integration
- HNSW performs comparably

### Zotero-first metadata
- 600K+ entries from local library
- Faster than API calls
- Higher accuracy for known papers

### JIT Retrieval (GAM paper)
- No pre-computed summaries
- Fresh synthesis per query
- Better for evolving questions

### 3-Stage Hallucination (HaluMem paper)
- Extract: Pull verifiable claims
- Update: Contextualize with evidence
- QA: Verify each claim

---

## MCP Tools

| Tool | Purpose |
|------|---------|
| `semantic_search` | Hybrid search with RRF |
| `retrieve_and_synthesize` | JIT retrieval + answer |
| `verify_claim` | Check against KB |
| `detect_hallucinations` | Full hallucination report |
| `find_research_gaps` | BridgeMine discovery |
| `find_similar_passages` | Vector similarity |
| `ingest_pdf` | Add to knowledge base |
| `get_stats` | System statistics |
| `graph_query` | Raw Cypher execution |

---

## Environment Variables

Required in `.env`:

```bash
POSTGRES_DSN=postgresql://polymath:password@localhost/polymath_v3
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=polymathic2026
GEMINI_API_KEY=your_key
OPENALEX_EMAIL=your@email.com
ZOTERO_CSV_PATH=/path/to/zotero.csv
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| pgvector not found | `CREATE EXTENSION vector;` |
| Neo4j connection refused | Check Docker: `docker ps` |
| Gemini rate limit | Increase delay in backfill |
| Embedding OOM | Reduce batch size |
| Zotero match wrong | Check title normalization |

---

## Performance Targets

| Operation | Target |
|-----------|--------|
| Hybrid search | < 500ms |
| Single PDF ingest | < 30s |
| Batch ingest | 500 PDFs/hour |
| Concept extraction | 1000/hour (Gemini) |
| Neo4j sync | 10K nodes/minute |
