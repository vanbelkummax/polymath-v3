# New Session Prompt: Execute Batch-v1 Integration Plan

Copy and paste this entire prompt to start a new Claude Code session:

---

## Task

Execute the implementation plan at `/home/user/polymath-v3/docs/plans/2026-01-17-batch-v1-integration.md`

This plan integrates 5,954 concepts from 973 passages (batch-v1 trial run) into:
1. Postgres embeddings (currently 0/973 have embeddings)
2. Neo4j graph (currently 0 batch-v1 concepts synced)

## Required Skills

**You MUST use these skills:**

1. **`superpowers:executing-plans`** - Load and execute the plan task-by-task
2. **`superpowers:subagent-driven-development`** - Dispatch fresh subagent for each task with code review between tasks

## Key Files

| File | Purpose |
|------|---------|
| `/home/user/polymath-v3/CLAUDE.md` | Project context and current state |
| `/home/user/polymath-v3/docs/plans/2026-01-17-batch-v1-integration.md` | **THE PLAN TO EXECUTE** |
| `/home/user/polymath-v3/lib/embeddings/bge_m3.py` | BGE-M3 embedder |
| `/home/user/polymath-v3/scripts/sync_neo4j.py` | Reference for Neo4j sync patterns |

## Environment

```bash
cd /home/user/polymath-v3
export POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_PASSWORD="polymathic2026"
```

## Execution Instructions

1. **First:** Read `/home/user/polymath-v3/CLAUDE.md` for context
2. **Then:** Use `superpowers:executing-plans` skill to load the plan
3. **Execute:** Each task using subagents, with code review between tasks
4. **Verify:** Run verification commands after each task
5. **Report:** Final status when complete

## Success Criteria

After execution, these should all pass:

```bash
# 973 passages should have embeddings
psql -U polymath -d polymath -c "
SELECT count(*) FROM passages p
JOIN passage_concepts pc ON p.passage_id = pc.passage_id
WHERE pc.extractor_version = 'batch-v1' AND p.embedding IS NOT NULL;
"
# Expected: 973

# Neo4j should have batch-v1 data
docker exec polymax-neo4j cypher-shell -u neo4j -p polymathic2026 "
MATCH (p:Passage)-[r:MENTIONS]->(c:Concept)
WHERE r.synced_at > datetime() - duration('P1D')
RETURN count(DISTINCT p) as passages, count(r) as mentions
"
# Expected: passages=973, mentions=5954
```

## Do NOT

- Do NOT skip the plan - execute it task by task
- Do NOT modify the plan without asking
- Do NOT run the full `sync_neo4j.py --incremental` (it takes 6+ hours) - use the targeted script in the plan

## Start

Begin by reading CLAUDE.md, then invoke the `superpowers:executing-plans` skill.
