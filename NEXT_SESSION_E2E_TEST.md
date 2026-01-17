# Next Session: End-to-End Pipeline Test

> **Copy this prompt to start the test session.**

---

## Mission

Test the complete Polymath pipeline on 20 carefully selected papers:
1. **Ingest PDFs** → passages + embeddings
2. **Extract concepts** → Gemini batch API
3. **Detect assets** → GitHub repos + HuggingFace models
4. **Extract skills** → Method-rich passages → CANDIDATE.md drafts
5. **Build citations** → DOI/PMID network
6. **Sync Neo4j** → Knowledge graph
7. **Generate registry** → Central asset reference

---

## Quick Start

```bash
cd /home/user/polymath-v3

# Run full pipeline (recommended)
python scripts/e2e_pipeline_test.py --run-all --papers 20

# Or run step-by-step for debugging
python scripts/e2e_pipeline_test.py --step ingest --papers 20
python scripts/e2e_pipeline_test.py --step concepts
python scripts/e2e_pipeline_test.py --step assets
python scripts/e2e_pipeline_test.py --step skills
python scripts/e2e_pipeline_test.py --step citations
python scripts/e2e_pipeline_test.py --step neo4j
python scripts/e2e_pipeline_test.py --step registry
```

---

## Environment

```bash
export POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_PASSWORD="polymathic2026"
export GOOGLE_APPLICATION_CREDENTIALS="/home/user/.gcp/service-account.json"
```

---

## What the Test Does

### Step 1: Paper Selection & Staging
- Scans PDF directories for diverse papers
- Prioritizes: spatial transcriptomics, deep learning, segmentation
- Copies 20 papers to `/home/user/work/e2e_test_batch/<batch>/pdfs/`

### Step 2: PDF Ingestion
- Parses PDFs with enhanced parser
- Generates passages with BGE-M3 embeddings (1024-dim)
- Stores in Postgres `passages` table

### Step 3: Concept Extraction (Batch API)
- Submits passages to Gemini batch API (50% cost discount)
- Monitor with: `python scripts/batch_concept_extraction_async.py --status`
- Process results: `python scripts/process_batch_results.py --latest`

### Step 4: Asset Detection
- Scans passages for GitHub URLs → `repo_queue` table
- Detects HuggingFace model IDs → `hf_models` table
- Records context and source document

### Step 5: Skill Extraction
- Identifies passages with method/algorithm concepts
- Calls Gemini for skill extraction
- Writes to `~/.claude/skills_drafts/<skill>/CANDIDATE.md`
- Generates `evidence.json` for Gate 1 validation

### Step 6: Citation Network
- Extracts DOIs from passages using regex
- Populates `citation_links` table
- Links citing → cited documents

### Step 7: Neo4j Sync
- Syncs passages, concepts, skills to graph
- Creates cross-domain relationships
- Enables graph queries

### Step 8: Asset Registry
- Generates `ASSET_REGISTRY.md` with:
  - Top GitHub repos by citation count
  - Top HuggingFace models
  - Recent extracted skills
  - Quick access commands

---

## Expected Output

### Test Report Location
```
/home/user/work/e2e_test_batch/<batch>/
├── pdfs/                    # Staged papers
├── E2E_TEST_REPORT.md       # Human-readable report
└── results.json             # Machine-readable results
```

### Asset Registry
```
/home/user/polymath-v3/ASSET_REGISTRY.md
```

### Skill Drafts
```
~/.claude/skills_drafts/
├── <skill-1>/
│   ├── CANDIDATE.md
│   └── evidence.json
├── <skill-2>/
│   └── ...
```

---

## Success Criteria

| Metric | Target | Check Command |
|--------|--------|---------------|
| Papers ingested | 20 | `psql -c "SELECT COUNT(*) FROM documents WHERE created_at > NOW() - INTERVAL '1 day'"` |
| Passages/paper | ≥50 avg | `psql -c "SELECT AVG(cnt) FROM (SELECT COUNT(*) cnt FROM passages GROUP BY doc_id) t"` |
| Concepts extracted | ≥100/paper | Check batch job results |
| GitHub repos | ≥3 | `psql -c "SELECT COUNT(*) FROM repo_queue"` |
| HF models | ≥3 | `psql -c "SELECT COUNT(*) FROM hf_models"` |
| Skills extracted | ≥5 | `ls ~/.claude/skills_drafts/ \| wc -l` |
| Citations | ≥10 | `psql -c "SELECT COUNT(*) FROM citation_links"` |

---

## Verification Commands

### Postgres
```sql
-- Document summary
SELECT title,
       (SELECT COUNT(*) FROM passages p WHERE p.doc_id = d.doc_id) as passages,
       (SELECT COUNT(*) FROM passage_concepts pc
        JOIN passages p ON pc.passage_id = p.passage_id
        WHERE p.doc_id = d.doc_id) as concepts
FROM documents d
WHERE created_at > NOW() - INTERVAL '1 day';

-- Asset summary
SELECT 'repos' as type, COUNT(*) FROM repo_queue
UNION ALL
SELECT 'models', COUNT(*) FROM hf_models
UNION ALL
SELECT 'skills', COUNT(*) FROM paper_skills WHERE status = 'draft';

-- Citation network
SELECT COUNT(*) as links, COUNT(DISTINCT citing_doc_id) as citing_docs
FROM citation_links;
```

### Neo4j
```cypher
// Node counts
MATCH (n) RETURN labels(n)[0] as label, count(n) ORDER BY count(n) DESC

// Recent documents
MATCH (d:Document)-[:HAS_PASSAGE]->(p:Passage)-[:MENTIONS]->(c:Concept)
WHERE d.created_at > datetime() - duration('P1D')
RETURN d.title, count(DISTINCT p) as passages, count(DISTINCT c) as concepts
LIMIT 10
```

### Filesystem
```bash
# Check skill drafts
python scripts/promote_skill.py --list
python scripts/promote_skill.py --check-all

# Check asset registry
head -50 ASSET_REGISTRY.md
```

---

## Troubleshooting

### Batch API Not Processing
```bash
# Check job status
python /home/user/polymath-repo/scripts/batch_concept_extraction_async.py --status

# If stuck, check GCP console or resubmit
```

### Neo4j Connection Failed
```bash
# Check if running
docker ps | grep neo4j

# Restart if needed
docker restart polymax-neo4j
```

### Skill Extraction Timeout
```bash
# Run on single document for debugging
python -c "
import asyncio
from lib.ingest.skill_extractor import SkillExtractor
import psycopg2

async def test():
    conn = psycopg2.connect('dbname=polymath user=polymath host=/var/run/postgresql')
    extractor = SkillExtractor(conn)
    # ... debug specific doc
"
```

---

## After the Test

1. **Review extracted skills**: `cat ~/.claude/skills_drafts/*/CANDIDATE.md`
2. **Add oracle tests** to promising skills
3. **Promote validated skills**: `python scripts/promote_skill.py <skill> --bootstrap`
4. **Update asset registry** if needed
5. **Commit results to git**

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/e2e_pipeline_test.py` | Main test orchestrator |
| `lib/ingest/skill_extractor.py` | Skill extraction to drafts |
| `lib/ingest/asset_detector.py` | GitHub/HF detection |
| `scripts/promote_skill.py` | Skill promotion with 4 gates |
| `ASSET_REGISTRY.md` | Central asset reference |

---

## Session Goals

By end of session, have:
- [ ] 20 papers fully processed through pipeline
- [ ] Concepts extracted via batch API
- [ ] GitHub repos and HF models cataloged
- [ ] Skills extracted to drafts (method-rich papers only)
- [ ] Citation network populated
- [ ] Neo4j graph updated
- [ ] Asset registry generated
- [ ] Test report with metrics

---

*Created: 2026-01-17*
