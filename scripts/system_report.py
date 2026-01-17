#!/usr/bin/env python3
"""
Polymath System Report Generator

Generates comprehensive system status reports for tracking health,
identifying issues, and planning improvements.

Usage:
    python scripts/system_report.py              # Full report to stdout
    python scripts/system_report.py --quick      # Quick summary
    python scripts/system_report.py -o FILE.md   # Save to file
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))

def get_db_connection():
    dsn = os.environ.get('POSTGRES_DSN', 'dbname=polymath user=polymath host=/var/run/postgresql')
    return psycopg2.connect(dsn)


def get_counts(conn) -> Dict:
    """Get all table counts."""
    cur = conn.cursor()
    counts = {}

    tables = [
        ('documents', 'documents'),
        ('passages', 'passages'),
        ('concepts', 'passage_concepts'),
        ('code_files', 'code_files'),
        ('code_chunks', 'code_chunks'),
        ('repo_queue', 'repo_queue'),
        ('skills', "paper_skills WHERE status = 'promoted'"),
        ('skill_drafts', "paper_skills WHERE status = 'draft'"),
        ('hf_models', 'hf_model_mentions'),
        ('citations', 'citation_links'),
    ]

    for name, query in tables:
        try:
            if ' WHERE ' in query:
                cur.execute(f"SELECT COUNT(*) FROM {query}")
            else:
                cur.execute(f"SELECT COUNT(*) FROM {query}")
            counts[name] = cur.fetchone()[0]
        except:
            counts[name] = 0

    return counts


def get_embedding_coverage(conn) -> Dict:
    """Get embedding coverage stats."""
    cur = conn.cursor()

    cur.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(embedding) as with_embedding
        FROM passages
    """)
    row = cur.fetchone()
    passages_total, passages_embedded = row

    return {
        'passages_total': passages_total,
        'passages_embedded': passages_embedded,
        'passages_pct': round(100 * passages_embedded / max(passages_total, 1), 1)
    }


def get_domain_distribution(conn) -> List[Dict]:
    """Get paper distribution by domain."""
    cur = conn.cursor()

    cur.execute("""
        SELECT
            CASE
                WHEN title ~* 'spatial|visium|xenium|merfish|cosmx|stereo' THEN 'Spatial Transcriptomics'
                WHEN title ~* 'pathology|histology|wsi|h&e|slide' THEN 'Pathology'
                WHEN title ~* 'single.cell|scrna|scatac|10x|droplet' THEN 'Single Cell'
                WHEN title ~* 'transformer|attention|bert|gpt|llm' THEN 'Deep Learning/NLP'
                WHEN title ~* 'graph|gnn|gcn|network' THEN 'Graph Methods'
                WHEN title ~* 'foundation.model|pretrain|self.supervis' THEN 'Foundation Models'
                ELSE 'Other'
            END as domain,
            COUNT(*) as papers
        FROM documents
        GROUP BY domain
        ORDER BY papers DESC
    """)

    return [{'domain': row[0], 'papers': row[1]} for row in cur.fetchall()]


def get_queue_status(conn) -> Dict:
    """Get repo queue status."""
    cur = conn.cursor()

    cur.execute("""
        SELECT status, COUNT(*)
        FROM repo_queue
        GROUP BY status
    """)

    return {row[0]: row[1] for row in cur.fetchall()}


def get_skill_stats(conn) -> Dict:
    """Get skill pipeline stats."""
    cur = conn.cursor()

    cur.execute("""
        SELECT status, COUNT(*),
               ROUND(AVG(COALESCE(evidence_count, 0)), 1) as avg_evidence
        FROM paper_skills
        GROUP BY status
    """)

    return {row[0]: {'count': row[1], 'avg_evidence': row[2]} for row in cur.fetchall()}


def get_recent_activity(conn) -> Dict:
    """Get recent activity stats."""
    cur = conn.cursor()

    activity = {}

    # Recent documents
    cur.execute("SELECT COUNT(*) FROM documents WHERE created_at > NOW() - INTERVAL '7 days'")
    activity['docs_7d'] = cur.fetchone()[0]

    # Recent concepts
    cur.execute("SELECT COUNT(*) FROM passage_concepts WHERE created_at > NOW() - INTERVAL '7 days'")
    activity['concepts_7d'] = cur.fetchone()[0]

    # Recent skills
    cur.execute("SELECT COUNT(*) FROM paper_skills WHERE created_at > NOW() - INTERVAL '7 days'")
    activity['skills_7d'] = cur.fetchone()[0]

    return activity


def check_services() -> Dict:
    """Check service health."""
    services = {}

    # Neo4j
    try:
        result = subprocess.run(
            ['docker', 'exec', 'polymax-neo4j', 'cypher-shell',
             '-u', 'neo4j', '-p', 'polymathic2026', 'RETURN 1'],
            capture_output=True, timeout=10
        )
        services['neo4j'] = 'running' if result.returncode == 0 else 'error'
    except:
        services['neo4j'] = 'not_running'

    # PostgreSQL
    try:
        conn = get_db_connection()
        conn.close()
        services['postgres'] = 'running'
    except:
        services['postgres'] = 'not_running'

    return services


def generate_quick_report(conn) -> str:
    """Generate quick summary report."""
    counts = get_counts(conn)
    emb = get_embedding_coverage(conn)
    services = check_services()

    report = f"""
# Polymath Quick Status - {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Counts
- Documents: {counts['documents']:,}
- Passages: {counts['passages']:,} ({emb['passages_pct']}% embedded)
- Concepts: {counts['concepts']:,}
- Code chunks: {counts['code_chunks']:,}
- Skills: {counts['skills']} promoted, {counts['skill_drafts']} drafts
- Repo queue: {counts['repo_queue']}

## Services
- PostgreSQL: {services['postgres']}
- Neo4j: {services['neo4j']}
"""
    return report


def generate_full_report(conn) -> str:
    """Generate comprehensive system report."""
    counts = get_counts(conn)
    emb = get_embedding_coverage(conn)
    domains = get_domain_distribution(conn)
    queue = get_queue_status(conn)
    skills = get_skill_stats(conn)
    activity = get_recent_activity(conn)
    services = check_services()

    report = f"""# Polymath System Status Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Documents | {counts['documents']:,} |
| Total Passages | {counts['passages']:,} |
| Embedding Coverage | {emb['passages_pct']}% |
| Total Concepts | {counts['concepts']:,} |
| Code Chunks | {counts['code_chunks']:,} |
| Promoted Skills | {counts['skills']} |

---

## Service Health

| Service | Status |
|---------|--------|
| PostgreSQL | {'✅' if services['postgres'] == 'running' else '❌'} {services['postgres']} |
| Neo4j | {'✅' if services['neo4j'] == 'running' else '❌'} {services['neo4j']} |

---

## Content Distribution

### By Domain

| Domain | Papers |
|--------|--------|
"""

    for d in domains:
        report += f"| {d['domain']} | {d['papers']:,} |\n"

    report += f"""
### Quality Metrics

| Metric | Total | With Data | Coverage |
|--------|-------|-----------|----------|
| Passage Embeddings | {emb['passages_total']:,} | {emb['passages_embedded']:,} | {emb['passages_pct']}% |

---

## Asset Pipeline

### GitHub Repository Queue

| Status | Count |
|--------|-------|
"""

    for status, count in queue.items():
        report += f"| {status} | {count} |\n"

    report += """
### Skill Pipeline

| Status | Count | Avg Evidence |
|--------|-------|--------------|
"""

    for status, data in skills.items():
        report += f"| {status} | {data['count']} | {data['avg_evidence']} |\n"

    report += f"""
---

## Recent Activity (7 days)

| Metric | Count |
|--------|-------|
| New Documents | {activity['docs_7d']} |
| New Concepts | {activity['concepts_7d']} |
| New Skills | {activity['skills_7d']} |

---

## Recommended Actions

"""

    # Generate recommendations
    actions = []

    if emb['passages_pct'] < 95:
        actions.append(f"- **Backfill embeddings**: {emb['passages_total'] - emb['passages_embedded']:,} passages missing embeddings")

    if queue.get('pending', 0) > 0:
        actions.append(f"- **Process repo queue**: {queue.get('pending', 0)} pending repos")

    if counts['skill_drafts'] > 10:
        actions.append(f"- **Review skill drafts**: {counts['skill_drafts']} drafts awaiting promotion")

    if not actions:
        actions.append("- System healthy, no immediate actions needed")

    report += '\n'.join(actions)

    report += """

---

## Quick Commands

```bash
# Process repo queue
python scripts/github_ingest.py --queue --limit 5

# Review skill drafts
python scripts/promote_skill.py --check-all

# Discover new assets
python scripts/discover_assets.py --recommend

# Backfill embeddings
python scripts/backfill_embeddings_batch.py --limit 1000
```

---

*Report generated by `scripts/system_report.py`*
"""

    return report


def main():
    parser = argparse.ArgumentParser(description='Polymath System Report')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick summary only')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    args = parser.parse_args()

    conn = get_db_connection()

    if args.quick:
        report = generate_quick_report(conn)
    else:
        report = generate_full_report(conn)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)

    conn.close()


if __name__ == '__main__':
    main()
