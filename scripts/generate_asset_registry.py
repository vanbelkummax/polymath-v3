#!/usr/bin/env python3
"""
Asset Registry Generator for Polymath

Generates ASSET_REGISTRY.md with organized references to:
- GitHub repositories (ingested and queued)
- HuggingFace models
- Extracted skills
- Key code implementations

Usage:
    python scripts/generate_asset_registry.py
    python scripts/generate_asset_registry.py --output ASSET_REGISTRY.md
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_db_connection():
    dsn = os.environ.get('POSTGRES_DSN', 'dbname=polymath user=polymath host=/var/run/postgresql')
    return psycopg2.connect(dsn)


def get_github_repos(conn) -> List[Dict]:
    """Get GitHub repositories with stats."""
    cur = conn.cursor()

    # Get from code_files (ingested)
    cur.execute("""
        SELECT
            repo_name,
            repo_url,
            COUNT(DISTINCT file_id) as files,
            SUM(loc) as total_loc,
            MAX(head_commit_sha) as commit_sha
        FROM code_files
        GROUP BY repo_name, repo_url
        ORDER BY files DESC
        LIMIT 50
    """)

    repos = []
    for row in cur.fetchall():
        repos.append({
            'name': row[0],
            'url': row[1] or f"https://github.com/{row[0]}",
            'files': row[2],
            'loc': row[3] or 0,
            'commit': row[4][:8] if row[4] else 'unknown',
            'status': 'ingested'
        })

    # Get from queue (pending)
    cur.execute("""
        SELECT owner, repo_name, repo_url, status, priority
        FROM repo_queue
        WHERE status != 'completed'
        ORDER BY priority DESC, created_at DESC
        LIMIT 30
    """)

    for row in cur.fetchall():
        repos.append({
            'name': f"{row[0]}/{row[1]}",
            'url': row[2],
            'files': 0,
            'loc': 0,
            'status': row[3],
            'priority': row[4]
        })

    return repos


def get_hf_models(conn) -> List[Dict]:
    """Get HuggingFace models."""
    cur = conn.cursor()

    # From mentions
    cur.execute("""
        SELECT
            model_id_raw,
            resolved,
            resolved_to_model_id,
            COUNT(*) as mention_count
        FROM hf_model_mentions
        GROUP BY model_id_raw, resolved, resolved_to_model_id
        ORDER BY mention_count DESC
        LIMIT 30
    """)

    models = []
    for row in cur.fetchall():
        models.append({
            'id': row[2] if row[1] else row[0],
            'resolved': row[1],
            'mentions': row[3]
        })

    # Add priority models not in DB
    priority_models = [
        ('MahmoodLab/UNI', 'pathology'),
        ('MahmoodLab/CONCH', 'pathology'),
        ('prov-gigapath/prov-gigapath', 'pathology'),
        ('facebook/dinov2-large', 'vision'),
        ('ctheodoris/Geneformer', 'single_cell'),
    ]

    existing_ids = {m['id'] for m in models}
    for model_id, domain in priority_models:
        if model_id not in existing_ids:
            models.append({
                'id': model_id,
                'resolved': False,
                'mentions': 0,
                'domain': domain,
                'priority': True
            })

    return models


def get_skills(conn) -> List[Dict]:
    """Get extracted skills."""
    cur = conn.cursor()

    cur.execute("""
        SELECT
            skill_name,
            skill_type,
            description,
            status,
            evidence_count,
            created_at
        FROM paper_skills
        ORDER BY
            CASE status WHEN 'promoted' THEN 0 WHEN 'draft' THEN 1 ELSE 2 END,
            created_at DESC
        LIMIT 50
    """)

    skills = []
    for row in cur.fetchall():
        skills.append({
            'name': row[0],
            'type': row[1],
            'description': row[2][:100] if row[2] else '',
            'status': row[3],
            'evidence': row[4] or 0,
            'created': row[5].strftime('%Y-%m-%d') if row[5] else ''
        })

    return skills


def categorize_repos(repos: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize repos by domain."""
    categories = defaultdict(list)

    domain_patterns = {
        'Spatial Transcriptomics': ['squidpy', 'spatial', 'visium', 'xenium', 'cell2location', 'banksy'],
        'Pathology': ['clam', 'uni', 'conch', 'hipt', 'pathology', 'wsi', 'mil'],
        'Single Cell': ['scanpy', 'scvi', 'scatac', 'anndata', 'seurat'],
        'Deep Learning': ['transformer', 'attention', 'pytorch', 'torch', 'dino'],
        'Graph Methods': ['gnn', 'gcn', 'graph', 'pyg', 'geometric'],
    }

    for repo in repos:
        categorized = False
        name_lower = repo['name'].lower()

        for category, patterns in domain_patterns.items():
            if any(p in name_lower for p in patterns):
                categories[category].append(repo)
                categorized = True
                break

        if not categorized:
            categories['Other'].append(repo)

    return dict(categories)


def generate_registry(conn) -> str:
    """Generate complete asset registry."""
    repos = get_github_repos(conn)
    models = get_hf_models(conn)
    skills = get_skills(conn)

    categorized_repos = categorize_repos(repos)

    # Count stats
    ingested_repos = len([r for r in repos if r['status'] == 'ingested'])
    queued_repos = len([r for r in repos if r['status'] != 'ingested'])
    promoted_skills = len([s for s in skills if s['status'] == 'promoted'])
    draft_skills = len([s for s in skills if s['status'] == 'draft'])

    registry = f"""# Polymath Asset Registry

> Central reference for GitHub repositories, HuggingFace models, and extracted skills.

**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Summary

| Asset Type | Ingested | Pending | Total |
|------------|----------|---------|-------|
| GitHub Repos | {ingested_repos} | {queued_repos} | {len(repos)} |
| HuggingFace Models | {len([m for m in models if m.get('resolved')])} | {len([m for m in models if not m.get('resolved')])} | {len(models)} |
| Skills | {promoted_skills} | {draft_skills} | {len(skills)} |

---

## GitHub Repositories

"""

    # Add repos by category
    for category in ['Spatial Transcriptomics', 'Pathology', 'Single Cell', 'Deep Learning', 'Graph Methods', 'Other']:
        if category in categorized_repos:
            category_repos = categorized_repos[category]
            registry += f"\n### {category}\n\n"
            registry += "| Repository | Files | LoC | Status |\n"
            registry += "|------------|-------|-----|--------|\n"

            for repo in category_repos[:10]:
                status_icon = '✅' if repo['status'] == 'ingested' else '⏳'
                registry += f"| [{repo['name']}]({repo['url']}) | {repo['files']} | {repo['loc']:,} | {status_icon} {repo['status']} |\n"

    registry += """
---

## HuggingFace Models

> Reference list of relevant models. Not downloaded - tracked for future use.

### Priority Models

| Model ID | Domain | Mentions | Notes |
|----------|--------|----------|-------|
"""

    priority_models = [m for m in models if m.get('priority')]
    for model in priority_models:
        registry += f"| `{model['id']}` | {model.get('domain', '-')} | {model['mentions']} | Priority |\n"

    registry += """
### Detected in Papers

| Model ID | Mentions | Resolved |
|----------|----------|----------|
"""

    paper_models = [m for m in models if not m.get('priority')][:15]
    for model in paper_models:
        resolved = '✅' if model['resolved'] else '❌'
        registry += f"| `{model['id']}` | {model['mentions']} | {resolved} |\n"

    registry += """
---

## Extracted Skills

### Promoted Skills

| Skill | Type | Description |
|-------|------|-------------|
"""

    for skill in [s for s in skills if s['status'] == 'promoted'][:20]:
        registry += f"| `{skill['name']}` | {skill['type']} | {skill['description']} |\n"

    registry += """
### Draft Skills (Pending Review)

| Skill | Type | Evidence | Created |
|-------|------|----------|---------|
"""

    for skill in [s for s in skills if s['status'] == 'draft'][:15]:
        registry += f"| `{skill['name']}` | {skill['type']} | {skill['evidence']} | {skill['created']} |\n"

    registry += """
---

## Quick Commands

### GitHub Operations

```bash
# Ingest a specific repo
python scripts/github_ingest.py https://github.com/owner/repo

# Ingest all repos from a user/org
python scripts/github_ingest.py --user scverse

# Process pending queue
python scripts/github_ingest.py --queue --limit 10

# List queue status
python scripts/github_ingest.py --list
```

### Discovery

```bash
# Discover new assets from papers
python scripts/discover_assets.py --recommend

# Find GitHub repos
python scripts/discover_assets.py --github --add-to-queue

# Find HuggingFace models
python scripts/discover_assets.py --hf --save-hf
```

### Skills

```bash
# List skill drafts
python scripts/promote_skill.py --list

# Check all drafts for promotion readiness
python scripts/promote_skill.py --check-all

# Promote a specific skill
python scripts/promote_skill.py skill-name --bootstrap
```

---

## Adding Assets

### To Add a GitHub Repository

1. **Quick add**: `python scripts/github_ingest.py https://github.com/owner/repo`
2. **Queue for later**: `python scripts/github_ingest.py https://github.com/owner/repo --add-only`

### To Track a HuggingFace Model

Add to `HF_MODELS_REFERENCE.md` or run:
```bash
python scripts/discover_assets.py --hf --save-hf
```

### To Add a Skill

Skills are auto-extracted from papers. To manually add:
1. Create `~/.claude/skills_drafts/skill-name/CANDIDATE.md`
2. Add `evidence.json` with source passages
3. Run `python scripts/promote_skill.py skill-name --bootstrap`

---

*Auto-generated by `scripts/generate_asset_registry.py`*
"""

    return registry


def main():
    parser = argparse.ArgumentParser(description='Generate Asset Registry')
    parser.add_argument('--output', '-o', default='ASSET_REGISTRY.md',
                       help='Output file (default: ASSET_REGISTRY.md)')
    args = parser.parse_args()

    conn = get_db_connection()
    registry = generate_registry(conn)
    conn.close()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path('/home/user/polymath-v3') / output_path

    output_path.write_text(registry)
    print(f"Asset registry saved to {output_path}")


if __name__ == '__main__':
    main()
