# Continue Session: SkillOps Implementation

> **Copy this entire prompt to start the next session.**

---

## Context

You are implementing the **SkillOps layer** for the Polymath System - a knowledge system that extracts actionable skills from papers, links code implementations, and discovers cross-domain insights.

---

## Load Context First

```bash
cd /home/user/polymath-v3
cat POLYMATH_WORKING_MEMORY.md
```

---

## What's Been Completed

### SkillOps (COMPLETE - 6/6 Tasks)

| Task | Status | Location |
|------|--------|----------|
| Skill directory structure | ✅ Done | `~/.claude/skills/`, `~/.claude/skills_drafts/` |
| SKILL.md template | ✅ Done | `~/.claude/SKILL_TEMPLATE.md` |
| Skill Router contract | ✅ Done | `~/.claude/SKILL_ROUTER.md` |
| Promotion script | ✅ Done | `scripts/promote_skill.py` |
| SkillOps schema | ✅ Done | `scripts/migrations/003_skillops.sql` |
| SkillExtractor (drafts-only) | ✅ Done | `lib/ingest/skill_extractor.py` |

### Earlier Work (All Complete)

| Component | Status | Location |
|-----------|--------|----------|
| Schema migration (8 tables) | ✅ Done | `scripts/migrations/002_polymath_assets.sql` |
| AssetDetector | ✅ Done | `lib/ingest/asset_detector.py` |
| SkillExtractor | ✅ Done | `lib/ingest/skill_extractor.py` |
| Batch-v1 integration | ✅ Done | 973 passages, 5,954 concepts in Neo4j |

---

## Current Task: Resume Original Roadmap

SkillOps layer is COMPLETE. Continue with knowledge graph and analysis components:

### Next Priority: Neo4j Full Sync
Extend `lib/db/neo4j_sync.py` to sync:
- All passages with embeddings
- Skills and skill bridges
- Cross-domain transfer relationships

### After That
1. **BridgeAnalyzer** (`lib/analysis/bridge_analyzer.py`) - Cross-domain discovery
2. **GapDetector** (`lib/analysis/gap_detector.py`) - Knowledge gap detection
3. **End-to-end test** - Full pipeline from paper to promoted skill

---

## Resume Original Roadmap

| Task | Priority |
|------|----------|
| Neo4j full sync | Next |
| BridgeAnalyzer | After |
| GapDetector | After |
| End-to-end test | Final |

---

## Environment

```bash
export POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_PASSWORD="polymathic2026"
```

Working directory: `/home/user/polymath-v3`

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `~/.claude/SKILL_TEMPLATE.md` | Standard SKILL.md format with oracles |
| `~/.claude/SKILL_ROUTER.md` | Deterministic routing protocol |
| `scripts/promote_skill.py` | Skill promotion with 4 gates |
| `lib/ingest/skill_extractor.py` | LLM-based skill extraction |
| `lib/ingest/asset_detector.py` | GitHub/HF/citation detection |
| `docs/POLYMATH_SYSTEM_SPEC.md` | Full architecture spec |

---

## System Notes

- Node.js heap increased to 16GB (`~/.bashrc` updated)
- 196GB RAM, 323GB disk free
- All prior work survived crash, files intact

---

## Start Command

```
Run E2E pipeline test on 20 papers:
python scripts/e2e_pipeline_test.py --run-all --papers 20

Or see detailed instructions: cat NEXT_SESSION_E2E_TEST.md
```
