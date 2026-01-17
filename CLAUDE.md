# Polymath v3 - Production Pipeline Guide

## ACTIVE TASK (2026-01-17)

**Current Focus:** SkillOps Layer - Automated skill extraction and promotion pipeline

**SkillOps Status:**
| Component | Status | Location |
|-----------|--------|----------|
| Skill Directory Structure | ✅ Done | `~/.claude/skills/`, `~/.claude/skills_drafts/` |
| SKILL.md Template | ✅ Done | `~/.claude/SKILL_TEMPLATE.md` |
| Skill Router Contract | ✅ Done | `~/.claude/SKILL_ROUTER.md` |
| Promotion Script (4 gates) | ✅ Done | `scripts/promote_skill.py` |
| SkillOps Schema | ✅ Done | `scripts/migrations/003_skillops.sql` |
| SkillExtractor (drafts-only) | ✅ Done | `lib/ingest/skill_extractor.py` |
| E2E Pipeline Test | ⏳ Next | `scripts/e2e_pipeline_test.py` |

**Previous Work (Batch-v1 Integration):**
- 5,954 concepts from 973 passages integrated
- Scripts: `scripts/backfill_embeddings_batch.py`, `scripts/sync_neo4j_batch.py`

**Skill Promotion Gates:**
1. Evidence: ≥2 source passages OR 1 passage + 1 code link
2. Oracle: Runnable test exists and passes
3. Dedup: Cosine similarity < 0.85 to existing skills
4. Usage: ≥1 logged successful use (bootstrap-skippable)

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
-- Core tables
documents (doc_id, title, authors, year, doi, ingest_batch)
passages (passage_id, doc_id, passage_text, embedding vector(1024))
passage_concepts (passage_id, concept_name, concept_type, confidence, extractor_version)

-- SkillOps tables (003_skillops.sql)
paper_skills (skill_id, skill_name, skill_type, description, embedding, status, is_canonical, canonical_skill_id)
skill_usage_log (usage_id, skill_name, outcome, oracle_passed, task_description)
hf_model_mentions (mention_id, doc_id, model_id_raw, resolved, resolved_to_model_id)
skill_bridges (source_skill_id, target_skill_id, validation_status, usage_count)
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

## E2E Pipeline Test

```bash
# Run complete pipeline on 20 papers
python scripts/e2e_pipeline_test.py --run-all --papers 20

# Run individual steps
python scripts/e2e_pipeline_test.py --step ingest
python scripts/e2e_pipeline_test.py --step concepts
python scripts/e2e_pipeline_test.py --step assets
python scripts/e2e_pipeline_test.py --step skills
python scripts/e2e_pipeline_test.py --step citations
python scripts/e2e_pipeline_test.py --step neo4j
python scripts/e2e_pipeline_test.py --step registry

# Generate report only
python scripts/e2e_pipeline_test.py --report
```

---

## Skill Promotion Commands

```bash
# List draft skills
python scripts/promote_skill.py --list

# Check all drafts (gate validation without promoting)
python scripts/promote_skill.py --check-all

# Promote a skill (validates all 4 gates)
python scripts/promote_skill.py <skill-name>

# Bootstrap promotion (skip usage gate for initial skills)
python scripts/promote_skill.py <skill-name> --bootstrap
```

---

## Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| API truncation | `max_output_tokens: 16384` in concept_extractor.py |
| Neo4j full sync slow | Use targeted `sync_neo4j_batch.py` for specific batches |
| Slow /mnt/ access | Copy PDFs to `/home/user/work/` first |
