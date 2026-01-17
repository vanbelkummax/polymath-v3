# Polymath Working Memory

> **Load this at session start.** It contains everything needed to continue work on the Polymath system.

## Current Status

| Component | Status | Location |
|-----------|--------|----------|
| **Full Spec** | ✅ Complete | `docs/POLYMATH_SYSTEM_SPEC.md` |
| **Schema Migration** | ✅ Complete | `scripts/migrations/002_polymath_assets.sql` |
| **Asset Detector** | ✅ Complete | `lib/ingest/asset_detector.py` |
| **Skill Extractor** | ✅ Complete | `lib/ingest/skill_extractor.py` |
| **Bridge Analyzer** | ⏳ Pending | `lib/analysis/bridge_analyzer.py` |
| **Gap Detector** | ⏳ Pending | `lib/analysis/gap_detector.py` |
| **Neo4j Full Sync** | ⏳ Pending | `lib/db/neo4j_sync.py` |

## Quick Context

**What is Polymath?** A knowledge system that:
1. Ingests papers → extracts concepts, skills, assets (GitHub/HuggingFace)
2. Builds a knowledge graph (Postgres + Neo4j)
3. Discovers cross-domain skill transfers (biology ↔ AI/ML)
4. Detects gaps and recommends papers to acquire
5. Promotes validated skills to `~/.claude/skills/`

**Key Insight:** Skills from one domain often transfer to another when they share mathematical structure (e.g., spatial autocorrelation works for both transcriptomics and ecology).

## Database Connections

```bash
# Postgres
export POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql"
psql -U polymath -d polymath

# Neo4j
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_PASSWORD="polymathic2026"
docker exec polymax-neo4j cypher-shell -u neo4j -p polymathic2026
```

## Key Tables (Existing)

| Table | Purpose | Count |
|-------|---------|-------|
| `documents` | Paper metadata | ~750 |
| `passages` | Text chunks + embeddings | ~750K |
| `passage_concepts` | Extracted concepts | ~6K (batch-v1) |
| `code_chunks` | AST-parsed code | ~575K |
| `code_files` | Source files | ~50K |

## New Tables (To Create)

| Table | Purpose |
|-------|---------|
| `repo_queue` | GitHub repos to download |
| `paper_repos` | Paper ↔ repo links |
| `hf_models` | HuggingFace model catalog |
| `paper_hf_models` | Paper ↔ model links |
| `paper_skills` | Extracted actionable skills |
| `skill_bridges` | Cross-domain transfers |
| `citation_links` | Paper citation network |
| `knowledge_gaps` | Detected gaps + recommendations |

## Priority Domains

**Biology:** spatial_transcriptomics, single_cell, pathology, genomics
**AI/ML:** deep_learning, computer_vision, nlp, graph_neural_networks

**Priority Bridges:** Find skills that connect these domains.

## Domain Abstractions (Cross-Domain Transfer)

| Abstraction | Shared By |
|-------------|-----------|
| Spatial autocorrelation | spatial_transcriptomics, geostatistics, time_series |
| Graph message passing | GNN, belief propagation, molecular dynamics |
| Attention mechanism | NLP, vision, genomics, spatial |
| Contrastive learning | Vision, NLP, molecular, pathology |
| Optimal transport | Single-cell, domain adaptation, image registration |
| Permutation testing | Spatial stats, genomics, ecology |
| Multiple instance learning | Pathology, drug discovery, remote sensing |

## Implementation Order

1. **Schema Migration** - Create new tables
2. **Asset Detector** - Find GitHub/HuggingFace in papers
3. **Skill Extractor** - LLM-based skill extraction
4. **Neo4j Sync** - Full graph sync
5. **Bridge Analyzer** - Cross-domain discovery
6. **Gap Detector** - Find missing knowledge
7. **Maintenance Scripts** - Daily/weekly jobs
8. **Skill Promoter** - Push to `~/.claude/skills/`

## Files to Create

```
lib/ingest/
├── asset_detector.py      # GitHub + HuggingFace detection
├── skill_extractor.py     # LLM-based skill extraction
└── citation_extractor.py  # DOI extraction + resolution

lib/analysis/
├── bridge_analyzer.py     # Cross-domain skill discovery
├── gap_detector.py        # Knowledge gap detection
└── recommender.py         # Paper recommendations

lib/db/
└── neo4j_sync.py          # Full Neo4j sync (extended)

scripts/
├── migrations/002_polymath_assets.sql
├── process_repo_queue.py
├── detect_gaps.py
├── recommend_papers.py
└── promote_skills.py

scripts/maintenance/
├── daily_analysis.py
└── weekly_analysis.py
```

## Testing the System

```bash
# After migration, test with a paper
python lib/ingest/pipeline.py /path/to/paper.pdf --full

# Check extracted assets
psql -U polymath -d polymath -c "SELECT * FROM paper_repos LIMIT 5"
psql -U polymath -d polymath -c "SELECT * FROM paper_skills LIMIT 5"

# Check Neo4j
docker exec polymax-neo4j cypher-shell -u neo4j -p polymathic2026 "
MATCH (s:Skill)-[:BRIDGES]->(s2:Skill) RETURN s.skill_name, s2.skill_name LIMIT 5
"
```

## Next Steps

When continuing work:
1. Read this file
2. Check status table above
3. Continue with next pending component
4. Update status when complete

---

*Last updated: 2026-01-17*
