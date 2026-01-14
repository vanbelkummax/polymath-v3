# Polymath v3

A scientific knowledge base system with hybrid search, concept extraction, hallucination detection, and research gap discovery.

## Features

- **Hybrid Search**: Combines pgvector similarity, PostgreSQL full-text search, and Neo4j graph traversal with RRF fusion
- **JIT Retrieval**: Query-time synthesis instead of pre-computed summaries (based on GAM paper)
- **Hallucination Detection**: 3-stage verification (extract → update → verify) based on HaluMem paper
- **BridgeMine**: Cross-domain research gap detection with novelty validation
- **MCP Server**: Claude Code integration for conversational access
- **GCP Batch**: Scalable processing for large archives

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Polymath v3                              │
├─────────────────────────────────────────────────────────────────┤
│  CLI / MCP Server                                                │
│  ├── polymath search "query"                                     │
│  ├── polymath ingest /path/to/paper.pdf                         │
│  ├── polymath gaps "spatial_transcriptomics"                    │
│  └── polymath verify "claim to check"                           │
├─────────────────────────────────────────────────────────────────┤
│  Core Libraries                                                  │
│  ├── lib/search/      Hybrid search, JIT retrieval, reranking  │
│  ├── lib/ingest/      PDF parsing, chunking, metadata, concepts │
│  ├── lib/validation/  Hallucination detection                   │
│  └── lib/bridgemine/  Gap detection, novelty checking           │
├─────────────────────────────────────────────────────────────────┤
│  Storage                                                         │
│  ├── PostgreSQL       Documents, passages, concepts, vectors    │
│  │   └── pgvector     HNSW index for 1024-dim BGE-M3 embeddings│
│  └── Neo4j            Concept graph (METHOD, PROBLEM, DOMAIN)   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/vanbelkummax/polymath-v3.git
cd polymath-v3

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env with your credentials
```

### Database Setup

```bash
# PostgreSQL with pgvector
psql -U postgres -c "CREATE DATABASE polymath_v3;"
psql -U postgres -d polymath_v3 -c "CREATE EXTENSION vector;"

# Apply schema
polymath init-db

# Neo4j (Docker)
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/polymathic2026 \
  neo4j:5
```

### Basic Usage

```bash
# Search the knowledge base
polymath search "spatial transcriptomics deconvolution" -n 20

# Ask a question with synthesis
polymath ask "What methods exist for cell type deconvolution?"

# Ingest a paper
polymath ingest /path/to/paper.pdf

# Find research gaps
polymath gaps "spatial_transcriptomics"

# Verify a claim
polymath verify "Transformers outperform CNNs for image segmentation"
```

## MCP Integration

Add to `~/.mcp.json`:

```json
{
  "mcpServers": {
    "polymath": {
      "command": "python",
      "args": ["/home/user/polymath-v3/mcp/polymath_server.py"],
      "env": {
        "PYTHONPATH": "/home/user/polymath-v3"
      }
    }
  }
}
```

Available tools:
- `semantic_search` - Hybrid search with reranking
- `retrieve_and_synthesize` - JIT retrieval with answer generation
- `verify_claim` - Check claim against knowledge base
- `detect_hallucinations` - Find potential hallucinations in text
- `find_research_gaps` - BridgeMine gap detection
- `ingest_pdf` - Add paper to knowledge base
- `get_stats` - Knowledge base statistics
- `graph_query` - Execute Cypher queries

## Batch Processing

### Re-ingest Archive

```bash
# Full archive processing
python scripts/reingest_archive.py \
  --archive-dir /scratch/polymath_archive \
  --workers 8

# Resume from checkpoint
python scripts/reingest_archive.py --resume
```

### Concept Backfill

```bash
# Single worker
python scripts/backfill_concepts.py --limit 10000

# Distributed (8 workers)
for i in $(seq 0 7); do
  python scripts/backfill_concepts.py \
    --worker-id $i --num-workers 8 \
    --delay 0.2 &
done
```

### Neo4j Sync

```bash
# Full sync
python scripts/sync_neo4j.py --full

# Incremental (new data only)
python scripts/sync_neo4j.py --incremental

# Build similarity edges
python scripts/sync_neo4j.py --build-similarity
```

## Configuration

Environment variables (`.env`):

```bash
# Database
POSTGRES_DSN=postgresql://polymath:password@localhost:5432/polymath_v3
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=polymathic2026

# APIs
GEMINI_API_KEY=your_key_here
OPENALEX_EMAIL=your@email.com

# Metadata
ZOTERO_CSV_PATH=/path/to/zotero_export.csv

# GCP (for batch)
GCP_PROJECT_ID=your_project
GCP_BUCKET=your_bucket
```

## Database Schema

### PostgreSQL

- `documents` - Paper metadata (title, authors, DOI, etc.)
- `passages` - Text chunks with embeddings (1024-dim BGE-M3)
- `passage_concepts` - Extracted concepts with confidence scores
- `ingest_batches` - Batch processing tracking
- `kb_migrations` - Job progress tracking

### Neo4j

Node types:
- `Paper` - Document metadata
- `Passage` - Text chunks
- `METHOD`, `PROBLEM`, `DOMAIN`, `DATASET`, `METRIC`, `ENTITY` - Concepts

Relationships:
- `FROM_PAPER` - Passage belongs to paper
- `MENTIONS` - Passage mentions concept
- `SIMILAR_TO` - Concepts co-occur frequently

## Development

```bash
# Run tests
pytest tests/

# Format code
ruff format lib/ cli/ mcp/ scripts/

# Type check
mypy lib/
```

## License

MIT

## Citation

If you use Polymath in your research, please cite:

```bibtex
@software{polymath_v3,
  author = {Van Belkum, Max},
  title = {Polymath v3: Scientific Knowledge Base with Hybrid Search},
  year = {2026},
  url = {https://github.com/vanbelkummax/polymath-v3}
}
```
