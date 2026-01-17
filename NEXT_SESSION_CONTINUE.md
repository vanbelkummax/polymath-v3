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

### SkillOps (Tasks 1-5 of 6)

| Task | Status | Location |
|------|--------|----------|
| Skill directory structure | ✅ Done | `~/.claude/skills/`, `~/.claude/skills_drafts/` |
| SKILL.md template | ✅ Done | `~/.claude/SKILL_TEMPLATE.md` |
| Skill Router contract | ✅ Done | `~/.claude/SKILL_ROUTER.md` |
| Promotion script | ✅ Done | `scripts/promote_skill.py` |
| SkillOps schema | ✅ Done | `scripts/migrations/003_skillops.sql` |
| Update SkillExtractor | ⏳ Next | `lib/ingest/skill_extractor.py` |

### Earlier Work (All Complete)

| Component | Status | Location |
|-----------|--------|----------|
| Schema migration (8 tables) | ✅ Done | `scripts/migrations/002_polymath_assets.sql` |
| AssetDetector | ✅ Done | `lib/ingest/asset_detector.py` |
| SkillExtractor | ✅ Done | `lib/ingest/skill_extractor.py` |
| Batch-v1 integration | ✅ Done | 973 passages, 5,954 concepts in Neo4j |

---

## Current Task: Update SkillExtractor

Modify `lib/ingest/skill_extractor.py` to:

1. **Write to drafts only** - Output to `~/.claude/skills_drafts/` not `skills/`
2. **Generate CANDIDATE.md** - Minimal skill template, not full SKILL.md
3. **Generate evidence.json** - Track passage IDs and code links for Gate 1

### CANDIDATE.md Format
```markdown
---
name: skill-name
status: candidate
extracted_from: doc_id
confidence: 0.8
---
# Skill Name
[Description from LLM extraction]

## Procedure
[Steps extracted]

## Evidence
See evidence.json
```

---

## Then Resume Original Roadmap

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
Read POLYMATH_WORKING_MEMORY.md, then update lib/ingest/skill_extractor.py to write drafts only.
Use TodoWrite to track progress.
```
