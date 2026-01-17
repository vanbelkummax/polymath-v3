# Continue Session: Polymath E2E Testing & Asset Ingestion

> **Copy this entire prompt to start the next session.**

---

## Context

You are working on the **Polymath System** - a knowledge system that:
- Ingests papers and code repositories
- Extracts concepts and actionable skills
- Builds a cross-domain knowledge graph
- Discovers GitHub repos and HuggingFace models from literature
- Generates insights for research and writing

---

## Quick Start

```bash
cd /home/user/polymath-v3

# System health check (run first)
python scripts/system_report.py --quick

# Generate asset registry
python scripts/generate_asset_registry.py
```

---

## What's Available

### GitHub Ingestion

```bash
# Ingest a single repo
python scripts/github_ingest.py https://github.com/owner/repo

# Ingest all repos from a user/organization
python scripts/github_ingest.py --user mahmoodlab

# Process pending queue
python scripts/github_ingest.py --queue --limit 10

# List queue status
python scripts/github_ingest.py --list
```

### Asset Discovery

```bash
# Find GitHub repos mentioned in papers
python scripts/discover_assets.py --github --add-to-queue

# Find HuggingFace models in papers
python scripts/discover_assets.py --hf --save-hf

# Full recommendations
python scripts/discover_assets.py --recommend

# Find knowledge gaps
python scripts/discover_assets.py --gaps
```

### E2E Pipeline Test

```bash
# Run full pipeline on 20 papers
python scripts/e2e_pipeline_test.py --run-all --papers 20

# Run individual steps
python scripts/e2e_pipeline_test.py --step ingest --papers 20
python scripts/e2e_pipeline_test.py --step concepts
python scripts/e2e_pipeline_test.py --step assets
python scripts/e2e_pipeline_test.py --step skills
python scripts/e2e_pipeline_test.py --step neo4j
```

### Skill Management

```bash
# List skill drafts
python scripts/promote_skill.py --list

# Check all drafts for promotion readiness
python scripts/promote_skill.py --check-all

# Promote a skill
python scripts/promote_skill.py skill-name --bootstrap
```

---

## Priority Tasks

| Task | Command | Priority |
|------|---------|----------|
| Run E2E test on 20 papers | `python scripts/e2e_pipeline_test.py --run-all --papers 20` | HIGH |
| Ingest priority repos | `python scripts/github_ingest.py --user mahmoodlab` | HIGH |
| Discover assets from papers | `python scripts/discover_assets.py --recommend --add-to-queue` | MEDIUM |
| Generate asset registry | `python scripts/generate_asset_registry.py` | MEDIUM |
| Review skill drafts | `python scripts/promote_skill.py --check-all` | LOW |

---

## Environment

```bash
export POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_PASSWORD="polymathic2026"
export GOOGLE_APPLICATION_CREDENTIALS="/home/user/.gcp/service-account.json"
```

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/github_ingest.py` | Ingest repos by URL, user, or from queue |
| `scripts/discover_assets.py` | Find repos/models from papers |
| `scripts/e2e_pipeline_test.py` | Full pipeline validation |
| `scripts/promote_skill.py` | Skill promotion with 4 gates |
| `scripts/system_report.py` | System health reports |
| `scripts/generate_asset_registry.py` | Generate ASSET_REGISTRY.md |
| `~/.claude/skills/polymath-system-analysis.md` | Meta-skill for system analysis |

---

## Priority GitHub Repos to Ingest

These are high-value repos mentioned in papers or known to be important:

```bash
# Spatial transcriptomics
python scripts/github_ingest.py scverse/squidpy
python scripts/github_ingest.py scverse/spatialdata
python scripts/github_ingest.py BayraktarLab/cell2location

# Pathology foundation models
python scripts/github_ingest.py mahmoodlab/UNI
python scripts/github_ingest.py mahmoodlab/CONCH
python scripts/github_ingest.py mahmoodlab/CLAM

# Segmentation
python scripts/github_ingest.py MouseLand/cellpose
python scripts/github_ingest.py stardist/stardist

# Single cell
python scripts/github_ingest.py scverse/scanpy
python scripts/github_ingest.py YosefLab/scvi-tools
```

---

## HuggingFace Models Reference

These are tracked for reference (not downloaded):

| Model | Domain | Use Case |
|-------|--------|----------|
| `MahmoodLab/UNI` | Pathology | H&E feature extraction |
| `MahmoodLab/CONCH` | Pathology | Multi-modal pathology |
| `prov-gigapath/prov-gigapath` | Pathology | Large foundation model |
| `facebook/dinov2-large` | Vision | Self-supervised features |
| `ctheodoris/Geneformer` | Single Cell | Gene expression |

Run `python scripts/discover_assets.py --hf --save-hf` to update `HF_MODELS_REFERENCE.md`.

---

## Verification Commands

### Database Counts

```bash
psql -U polymath -d polymath -c "
SELECT 'documents' as t, COUNT(*) FROM documents
UNION ALL SELECT 'passages', COUNT(*) FROM passages
UNION ALL SELECT 'concepts', COUNT(*) FROM passage_concepts
UNION ALL SELECT 'code_chunks', COUNT(*) FROM code_chunks
UNION ALL SELECT 'repo_queue', COUNT(*) FROM repo_queue
"
```

### Neo4j Status

```bash
docker exec polymax-neo4j cypher-shell -u neo4j -p polymathic2026 "
MATCH (n) RETURN labels(n)[0] as type, count(n) ORDER BY count(n) DESC LIMIT 10
"
```

---

## Session Goals

Choose based on priority:

### Option A: E2E Pipeline Test
- [ ] Run E2E test on 20 diverse papers
- [ ] Verify all pipeline stages complete
- [ ] Review extracted skills
- [ ] Generate test report

### Option B: Asset Ingestion
- [ ] Ingest priority GitHub repos (mahmoodlab, scverse)
- [ ] Run discovery on papers
- [ ] Update asset registry
- [ ] Track HF models

### Option C: System Review
- [ ] Run full system report
- [ ] Identify gaps and issues
- [ ] Plan improvements
- [ ] Update documentation

---

## Troubleshooting

### Neo4j Not Running
```bash
docker restart polymax-neo4j
# Wait 30 seconds
docker exec polymax-neo4j cypher-shell -u neo4j -p polymathic2026 "RETURN 1"
```

### Batch API Issues
```bash
# Check status
python /home/user/polymath-repo/scripts/batch_concept_extraction_async.py --status

# Process results
python /home/user/polymath-repo/scripts/process_batch_results.py --latest
```

### Slow Performance
- Copy PDFs to `/home/user/work/` (not `/mnt/`)
- Use batch operations where possible
- Check disk space: `df -h /`

---

*Created: 2026-01-17*
