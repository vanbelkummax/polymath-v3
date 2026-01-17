# Batch-v1 Trial Run Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fully integrate 5,954 concepts from 973 passages (batch-v1 trial run) into Postgres embeddings and Neo4j graph.

**Architecture:**
1. Generate BGE-M3 embeddings for 973 passages missing embeddings
2. Sync batch-v1 concepts to Neo4j as typed nodes (METHOD, PROBLEM, DOMAIN, ENTITY)
3. Create MENTIONS relationships linking passages to concepts
4. Verify end-to-end traceability

**Tech Stack:** BGE-M3 (local GPU), Postgres/pgvector, Neo4j, Python

---

## Task 1: Create Targeted Embedding Backfill Script

**Files:**
- Create: `/home/user/polymath-v3/scripts/backfill_embeddings_batch.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""
Backfill embeddings for passages that have batch-v1 concepts but no embeddings.

Usage:
    python scripts/backfill_embeddings_batch.py --dry-run
    python scripts/backfill_embeddings_batch.py
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import numpy as np
from lib.config import config
from lib.embeddings.bge_m3 import Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 32  # GPU batch size


def get_passages_needing_embeddings(conn):
    """Get passages with batch-v1 concepts but no embeddings."""
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT p.passage_id, p.passage_text
        FROM passages p
        JOIN passage_concepts pc ON p.passage_id = pc.passage_id
        WHERE pc.extractor_version = 'batch-v1'
        AND p.embedding IS NULL
        ORDER BY p.passage_id
    """)
    results = cur.fetchall()
    cur.close()
    return results


def update_embedding(conn, passage_id, embedding):
    """Update passage with embedding."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE passages
        SET embedding = %s, embedding_model = 'bge-m3'
        WHERE passage_id = %s
    """, (embedding.tolist(), passage_id))
    conn.commit()
    cur.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    args = parser.parse_args()

    conn = psycopg2.connect(config.POSTGRES_DSN)

    passages = get_passages_needing_embeddings(conn)
    logger.info(f"Found {len(passages)} passages needing embeddings")

    if args.dry_run:
        logger.info("[DRY RUN] Would generate embeddings for these passages")
        conn.close()
        return

    if not passages:
        logger.info("No passages need embeddings")
        conn.close()
        return

    # Initialize embedder
    embedder = Embedder()

    # Process in batches
    for i in range(0, len(passages), BATCH_SIZE):
        batch = passages[i:i + BATCH_SIZE]
        texts = [p[1] for p in batch]
        ids = [p[0] for p in batch]

        logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(passages) + BATCH_SIZE - 1)//BATCH_SIZE}")

        # Generate embeddings
        embeddings = embedder.encode(texts)

        # Store each embedding
        for j, (passage_id, embedding) in enumerate(zip(ids, embeddings)):
            update_embedding(conn, passage_id, embedding)

        logger.info(f"  Stored {len(batch)} embeddings")

    conn.close()
    logger.info(f"Done! Generated embeddings for {len(passages)} passages")


if __name__ == "__main__":
    main()
```

**Step 2: Verify script syntax**

Run: `python -m py_compile scripts/backfill_embeddings_batch.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add scripts/backfill_embeddings_batch.py
git commit -m "feat: add targeted embedding backfill script for batch-v1 passages"
```

---

## Task 2: Generate Embeddings for batch-v1 Passages

**Files:**
- Execute: `/home/user/polymath-v3/scripts/backfill_embeddings_batch.py`

**Step 1: Dry run to verify**

Run:
```bash
cd /home/user/polymath-v3
export POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql"
python scripts/backfill_embeddings_batch.py --dry-run
```
Expected: `Found 973 passages needing embeddings`

**Step 2: Generate embeddings**

Run:
```bash
python scripts/backfill_embeddings_batch.py
```
Expected: ~30 batches processed, `Done! Generated embeddings for 973 passages`

**Step 3: Verify embeddings stored**

Run:
```bash
psql -U polymath -d polymath -c "
SELECT count(*) as with_embedding
FROM passages p
JOIN passage_concepts pc ON p.passage_id = pc.passage_id
WHERE pc.extractor_version = 'batch-v1'
AND p.embedding IS NOT NULL;
"
```
Expected: `973`

---

## Task 3: Create Targeted Neo4j Sync Script for batch-v1

**Files:**
- Create: `/home/user/polymath-v3/scripts/sync_neo4j_batch.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""
Sync batch-v1 concepts and their passages to Neo4j.

Creates:
- Passage nodes for passages with batch-v1 concepts
- Concept nodes (typed: METHOD, PROBLEM, DOMAIN, ENTITY)
- MENTIONS relationships
- FROM_PAPER relationships

Usage:
    python scripts/sync_neo4j_batch.py --dry-run
    python scripts/sync_neo4j_batch.py
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from neo4j import GraphDatabase
from lib.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 500


def get_batch_v1_data(conn):
    """Get all batch-v1 passages, concepts, and documents."""
    cur = conn.cursor()

    # Get passages
    cur.execute("""
        SELECT DISTINCT p.passage_id, p.doc_id, p.section,
               LEFT(p.passage_text, 200) as text_preview
        FROM passages p
        JOIN passage_concepts pc ON p.passage_id = pc.passage_id
        WHERE pc.extractor_version = 'batch-v1'
    """)
    passages = cur.fetchall()

    # Get concepts
    cur.execute("""
        SELECT DISTINCT concept_name, concept_type
        FROM passage_concepts
        WHERE extractor_version = 'batch-v1'
    """)
    concepts = cur.fetchall()

    # Get mentions (passage -> concept relationships)
    cur.execute("""
        SELECT passage_id, concept_name, concept_type, confidence
        FROM passage_concepts
        WHERE extractor_version = 'batch-v1'
    """)
    mentions = cur.fetchall()

    # Get documents
    cur.execute("""
        SELECT DISTINCT d.doc_id, d.title, d.year, d.doi
        FROM documents d
        JOIN passages p ON d.doc_id = p.doc_id
        JOIN passage_concepts pc ON p.passage_id = pc.passage_id
        WHERE pc.extractor_version = 'batch-v1'
    """)
    documents = cur.fetchall()

    cur.close()
    return passages, concepts, mentions, documents


def sync_to_neo4j(driver, passages, concepts, mentions, documents, dry_run=False):
    """Sync data to Neo4j."""

    if dry_run:
        logger.info(f"[DRY RUN] Would sync:")
        logger.info(f"  - {len(documents)} Paper nodes")
        logger.info(f"  - {len(passages)} Passage nodes")
        logger.info(f"  - {len(concepts)} Concept nodes")
        logger.info(f"  - {len(mentions)} MENTIONS relationships")
        return

    # 1. Sync Paper nodes
    logger.info(f"Syncing {len(documents)} Paper nodes...")
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        params = [{"doc_id": str(d[0]), "title": d[1], "year": d[2], "doi": d[3]} for d in batch]
        driver.execute_query("""
            UNWIND $docs as d
            MERGE (p:Paper {doc_id: d.doc_id})
            SET p.title = d.title, p.year = d.year, p.doi = d.doi, p.synced_at = datetime()
        """, docs=params)
    logger.info(f"  ✓ Papers synced")

    # 2. Sync Passage nodes with FROM_PAPER
    logger.info(f"Syncing {len(passages)} Passage nodes...")
    for i in range(0, len(passages), BATCH_SIZE):
        batch = passages[i:i + BATCH_SIZE]
        params = [{"passage_id": str(p[0]), "doc_id": str(p[1]), "section": p[2], "text_preview": p[3]} for p in batch]
        driver.execute_query("""
            UNWIND $passages as p
            MERGE (passage:Passage {passage_id: p.passage_id})
            SET passage.doc_id = p.doc_id,
                passage.section = p.section,
                passage.text_preview = p.text_preview,
                passage.synced_at = datetime()
            WITH passage, p
            MATCH (paper:Paper {doc_id: p.doc_id})
            MERGE (passage)-[:FROM_PAPER]->(paper)
        """, passages=params)
    logger.info(f"  ✓ Passages synced")

    # 3. Sync Concept nodes (typed)
    logger.info(f"Syncing {len(concepts)} Concept nodes...")
    by_type = {}
    for name, ctype in concepts:
        if ctype not in by_type:
            by_type[ctype] = []
        by_type[ctype].append(name)

    for ctype, names in by_type.items():
        driver.execute_query("""
            UNWIND $names as name
            MERGE (c:Concept {name: name, type: $ctype})
            SET c.synced_at = datetime()
        """, names=names, ctype=ctype)
        logger.info(f"    ✓ {len(names)} {ctype} concepts")
    logger.info(f"  ✓ Concepts synced")

    # 4. Sync MENTIONS relationships
    logger.info(f"Syncing {len(mentions)} MENTIONS relationships...")
    for i in range(0, len(mentions), BATCH_SIZE):
        batch = mentions[i:i + BATCH_SIZE]
        params = [{"passage_id": str(m[0]), "concept_name": m[1], "concept_type": m[2], "confidence": m[3]} for m in batch]
        driver.execute_query("""
            UNWIND $mentions as m
            MATCH (p:Passage {passage_id: m.passage_id})
            MATCH (c:Concept {name: m.concept_name, type: m.concept_type})
            MERGE (p)-[r:MENTIONS]->(c)
            SET r.confidence = m.confidence, r.synced_at = datetime()
        """, mentions=params)
        if (i + BATCH_SIZE) % 2000 == 0:
            logger.info(f"    Progress: {i + len(batch)}/{len(mentions)}")
    logger.info(f"  ✓ MENTIONS synced")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Connect to Postgres
    conn = psycopg2.connect(config.POSTGRES_DSN)

    # Get data
    logger.info("Fetching batch-v1 data from Postgres...")
    passages, concepts, mentions, documents = get_batch_v1_data(conn)
    conn.close()

    logger.info(f"Found: {len(passages)} passages, {len(concepts)} concepts, {len(mentions)} mentions, {len(documents)} documents")

    # Connect to Neo4j
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=("neo4j", config.NEO4J_PASSWORD)
    )

    # Sync
    sync_to_neo4j(driver, passages, concepts, mentions, documents, dry_run=args.dry_run)

    driver.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
```

**Step 2: Verify script syntax**

Run: `python -m py_compile scripts/sync_neo4j_batch.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add scripts/sync_neo4j_batch.py
git commit -m "feat: add targeted Neo4j sync script for batch-v1 data"
```

---

## Task 4: Sync batch-v1 Data to Neo4j

**Files:**
- Execute: `/home/user/polymath-v3/scripts/sync_neo4j_batch.py`

**Step 1: Ensure Neo4j is running**

Run:
```bash
docker ps | grep neo4j || docker start neo4j
```
Expected: Container running

**Step 2: Dry run to verify**

Run:
```bash
cd /home/user/polymath-v3
export POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_PASSWORD="polymathic2026"
python scripts/sync_neo4j_batch.py --dry-run
```
Expected:
```
[DRY RUN] Would sync:
  - 740 Paper nodes
  - 973 Passage nodes
  - ~1000 Concept nodes
  - 5954 MENTIONS relationships
```

**Step 3: Execute sync**

Run:
```bash
python scripts/sync_neo4j_batch.py
```
Expected: All items synced successfully (~1-2 minutes)

**Step 4: Verify in Neo4j**

Run:
```bash
docker exec polymax-neo4j cypher-shell -u neo4j -p polymathic2026 "
MATCH (p:Passage)-[:MENTIONS]->(c:Concept)
WHERE c.synced_at > datetime() - duration('PT1H')
RETURN count(DISTINCT p) as passages, count(DISTINCT c) as concepts
"
```
Expected: `passages: 973, concepts: ~1000`

---

## Task 5: Verify End-to-End Integration

**Step 1: Test Postgres traceability**

Run:
```bash
psql -U polymath -d polymath -c "
SELECT
  pc.concept_name,
  LEFT(p.passage_text, 100) as passage,
  d.title,
  d.doi,
  p.embedding IS NOT NULL as has_embedding
FROM passage_concepts pc
JOIN passages p ON pc.passage_id = p.passage_id
JOIN documents d ON p.doc_id = d.doc_id
WHERE pc.extractor_version = 'batch-v1'
LIMIT 3;
"
```
Expected: 3 rows with concept, passage text, title, doi, has_embedding=t

**Step 2: Test Neo4j graph traversal**

Run:
```bash
docker exec polymax-neo4j cypher-shell -u neo4j -p polymathic2026 "
MATCH (c:Concept)<-[:MENTIONS]-(p:Passage)-[:FROM_PAPER]->(paper:Paper)
WHERE c.type = 'method'
RETURN c.name, p.text_preview, paper.title, paper.doi
LIMIT 3
"
```
Expected: 3 rows with full chain: concept → passage → paper

**Step 3: Test vector search on new embeddings**

Run:
```bash
psql -U polymath -d polymath -c "
SELECT p.passage_id, LEFT(p.passage_text, 100),
       p.embedding <-> (SELECT embedding FROM passages WHERE passage_text LIKE '%spatial transcriptomics%' LIMIT 1) as distance
FROM passages p
JOIN passage_concepts pc ON p.passage_id = pc.passage_id
WHERE pc.extractor_version = 'batch-v1'
AND p.embedding IS NOT NULL
ORDER BY distance
LIMIT 3;
"
```
Expected: 3 passages ranked by semantic similarity

**Step 4: Commit verification results**

```bash
git add -A
git commit -m "docs: verify batch-v1 integration complete"
```

---

## Summary Checklist

| Task | Description | Verification |
|------|-------------|--------------|
| 1 | Create embedding backfill script | Script compiles |
| 2 | Generate embeddings for 973 passages | `count(*) = 973` with embeddings |
| 3 | Create Neo4j sync script | Script compiles |
| 4 | Sync to Neo4j | 973 passages, ~1000 concepts, 5954 relationships |
| 5 | End-to-end verification | Postgres + Neo4j + Vector search work |

**Total estimated time:** 15-20 minutes (mostly embedding generation)
