# End-to-End Pipeline Test Plan

> **Goal:** Validate the complete Polymath pipeline from PDF ingestion to promoted skills on 20 carefully selected papers.

## Test Scope

### Input
- 20 diverse, high-quality papers covering:
  - Spatial transcriptomics methods (5 papers)
  - Deep learning / computer vision (5 papers)
  - Cross-domain methods (5 papers)
  - Foundational/classic papers (5 papers)

### Pipeline Components to Test

| Component | Status | Validation |
|-----------|--------|------------|
| PDF parsing → passages | Test | Count passages per doc |
| BGE-M3 embeddings | Test | Verify 1024-dim vectors |
| Batch API concept extraction | Test | Gemini batch job |
| Asset detection (GitHub/HF) | Test | Check `repo_queue`, `hf_models` |
| Skill extraction → drafts | Test | Check `skills_drafts/` |
| Citation network | Test | Check `citation_links` |
| Neo4j full sync | Test | Verify graph structure |
| Central asset registry | Test | Generate reference docs |

## Test Papers Selection

### Criteria
1. **Method-rich**: Contains extractable procedures/algorithms
2. **Asset-linked**: References GitHub repos or HuggingFace models
3. **Cross-domain potential**: Methods that transfer across fields
4. **Citation-worthy**: Well-cited or foundational

### Selected Papers (20)

#### Spatial Transcriptomics (5)
```
1. Squidpy spatial analysis framework
2. SpatialData unified data format
3. Cell2location deconvolution
4. BANKSY spatial clustering
5. Neighborhood enrichment methods
```

#### Deep Learning / Vision (5)
```
6. Vision Transformer (ViT)
7. DINO self-supervised learning
8. Segment Anything Model (SAM)
9. UNI pathology foundation model
10. Multiple Instance Learning for pathology
```

#### Cross-Domain Methods (5)
```
11. Optimal transport for single-cell
12. Graph neural networks for biology
13. Attention mechanisms (Transformer)
14. Contrastive learning (CLIP/SimCLR)
15. Variational autoencoders for biology
```

#### Foundational (5)
```
16. UMAP dimensionality reduction
17. Scanpy single-cell analysis
18. Seurat spatial transcriptomics
19. CellPose segmentation
20. StarDist nuclei detection
```

## Pipeline Steps

### Step 1: Paper Selection & Staging
```bash
# Create test batch directory
mkdir -p /home/user/work/e2e_test_batch/pdfs
mkdir -p /home/user/work/e2e_test_batch/results

# Copy selected PDFs (script will do this)
```

### Step 2: Ingestion (Passages + Embeddings)
```bash
# Run unified ingest with enhanced parser
for pdf in /home/user/work/e2e_test_batch/pdfs/*.pdf; do
    python lib/ingest/unified_ingest.py "$pdf" --enhanced-parser
done
```

### Step 3: Batch Concept Extraction
```bash
# Submit batch job for all new passages
POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql" \
python scripts/batch_concept_extraction_async.py --doc-ids <doc_ids> --limit 2000

# Monitor and process results
python scripts/batch_concept_extraction_async.py --status
python scripts/process_batch_results.py --latest
```

### Step 4: Asset Detection
```bash
# Run asset detector on new documents
python -c "
from lib.ingest.asset_detector import AssetDetector
import psycopg2

conn = psycopg2.connect('dbname=polymath user=polymath host=/var/run/postgresql')
detector = AssetDetector(conn)

# Get new doc_ids
cur = conn.cursor()
cur.execute('''SELECT doc_id FROM documents WHERE created_at > NOW() - INTERVAL '1 day' ''')
doc_ids = [row[0] for row in cur.fetchall()]

for doc_id in doc_ids:
    detector.detect_and_store(doc_id)
"
```

### Step 5: Skill Extraction
```bash
# Run skill extractor (writes to drafts only)
python -c "
import asyncio
from lib.ingest.skill_extractor import SkillExtractor
import psycopg2

async def extract_skills():
    conn = psycopg2.connect('dbname=polymath user=polymath host=/var/run/postgresql')
    extractor = SkillExtractor(conn)

    # Get passages with method concepts
    cur = conn.cursor()
    cur.execute('''
        SELECT DISTINCT p.doc_id, d.title
        FROM passages p
        JOIN documents d ON p.doc_id = d.doc_id
        JOIN passage_concepts pc ON p.passage_id = pc.passage_id
        WHERE pc.concept_type IN ('method', 'algorithm', 'workflow')
        AND d.created_at > NOW() - INTERVAL '1 day'
    ''')

    for doc_id, title in cur.fetchall():
        cur.execute('SELECT passage_id, passage_text FROM passages WHERE doc_id = %s', (doc_id,))
        passages = [{'passage_id': r[0], 'passage_text': r[1]} for r in cur.fetchall()]

        cur.execute('SELECT passage_id, concept_name, concept_type FROM passage_concepts WHERE passage_id IN (SELECT passage_id FROM passages WHERE doc_id = %s)', (doc_id,))
        concepts = [{'passage_id': r[0], 'concept_name': r[1], 'concept_type': r[2]} for r in cur.fetchall()]

        await extractor.extract_and_write_drafts(str(doc_id), passages, concepts, title)

asyncio.run(extract_skills())
"
```

### Step 6: Citation Network
```bash
# Extract and link citations
python scripts/extract_citations.py --recent
```

### Step 7: Neo4j Sync
```bash
# Full sync to Neo4j
python scripts/sync_neo4j.py --full
```

### Step 8: Generate Asset Registry
```bash
# Generate central reference documents
python scripts/generate_asset_registry.py
```

## Validation Queries

### Postgres Checks
```sql
-- Count test batch documents
SELECT COUNT(*) FROM documents WHERE created_at > NOW() - INTERVAL '1 day';

-- Passage statistics
SELECT d.title, COUNT(p.passage_id) as passages,
       COUNT(DISTINCT pc.concept_name) as concepts
FROM documents d
LEFT JOIN passages p ON d.doc_id = p.doc_id
LEFT JOIN passage_concepts pc ON p.passage_id = pc.passage_id
WHERE d.created_at > NOW() - INTERVAL '1 day'
GROUP BY d.doc_id, d.title;

-- Asset detection results
SELECT COUNT(*) FROM repo_queue WHERE created_at > NOW() - INTERVAL '1 day';
SELECT COUNT(*) FROM hf_models WHERE created_at > NOW() - INTERVAL '1 day';

-- Skill extraction results
SELECT COUNT(*) FROM paper_skills WHERE created_at > NOW() - INTERVAL '1 day';
```

### Neo4j Checks
```cypher
// Document-concept graph
MATCH (d:Document)-[:HAS_PASSAGE]->(p:Passage)-[:MENTIONS]->(c:Concept)
WHERE d.created_at > datetime() - duration('P1D')
RETURN d.title, count(DISTINCT p) as passages, count(DISTINCT c) as concepts

// Cross-domain skills
MATCH (s:Skill)-[:TRANSFERS_TO]->(domain:Domain)
RETURN s.skill_name, collect(domain.name) as domains
```

### Filesystem Checks
```bash
# Check skill drafts
ls -la ~/.claude/skills_drafts/

# Check asset registry
cat /home/user/polymath-v3/ASSET_REGISTRY.md
```

## Success Criteria

| Metric | Target |
|--------|--------|
| Papers ingested | 20/20 |
| Passages per paper | ≥50 avg |
| Concepts extracted | ≥100 per paper |
| GitHub repos detected | ≥5 total |
| HF models detected | ≥5 total |
| Skills extracted | ≥10 total |
| Neo4j nodes created | >1000 |
| Citation links | ≥20 |

## Output Artifacts

1. **ASSET_REGISTRY.md** - Central reference for GitHub repos and HF models
2. **E2E_TEST_REPORT.md** - Detailed test results
3. **skills_drafts/** - Extracted skill candidates
4. **Neo4j graph** - Updated knowledge graph

---

*Created: 2026-01-17*
