#!/usr/bin/env python3
"""
Asset Discovery System for Polymath

Discovers GitHub repos and HuggingFace models that should be ingested based on:
1. Paper mentions (repos/models cited in papers)
2. Knowledge gaps (topics we have papers on but no code)
3. Dependency analysis (repos used by repos we already have)
4. Research domain priorities

Usage:
    python scripts/discover_assets.py --github        # Discover GitHub repos
    python scripts/discover_assets.py --hf            # Discover HuggingFace models
    python scripts/discover_assets.py --gaps          # Find knowledge gaps
    python scripts/discover_assets.py --recommend     # Full recommendations
"""

import os
import re
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime
import psycopg2
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Priority domains for asset discovery
PRIORITY_DOMAINS = {
    'spatial_transcriptomics': [
        'visium', 'xenium', 'merfish', 'cosmx', 'spatial', 'slide-seq',
        'stereo-seq', 'st-', 'spot deconvolution', 'squidpy', 'scanpy'
    ],
    'pathology': [
        'wsi', 'histopathology', 'h&e', 'pathology', 'mil', 'slide',
        'tissue', 'nuclei segmentation', 'cellpose', 'stardist'
    ],
    'foundation_models': [
        'foundation model', 'pretrained', 'self-supervised', 'contrastive',
        'dino', 'vit', 'transformer', 'uni', 'conch', 'gigapath'
    ],
    'single_cell': [
        'single-cell', 'scrna', 'scatac', 'multiome', 'anndata',
        'scanpy', 'seurat', 'cell2location', 'scvi'
    ],
    'deep_learning': [
        'pytorch', 'tensorflow', 'attention', 'transformer', 'unet',
        'resnet', 'graph neural', 'gnn', 'variational'
    ],
}

# Known high-value repos (prioritize these)
PRIORITY_REPOS = {
    # Spatial transcriptomics
    'scverse/squidpy': 10,
    'scverse/spatialdata': 10,
    'BayraktarLab/cell2location': 10,
    'prabhakarlab/Banksy': 9,
    'theislab/scanpy': 10,
    # Pathology
    'mahmoodlab/UNI': 10,
    'mahmoodlab/CONCH': 10,
    'mahmoodlab/CLAM': 10,
    'mahmoodlab/HIPT': 9,
    'MouseLand/cellpose': 10,
    'stardist/stardist': 10,
    # Foundation models
    'facebookresearch/dino': 9,
    'facebookresearch/dinov2': 9,
    'google-research/vision_transformer': 9,
    # Deep learning
    'pyg-team/pytorch_geometric': 9,
    'huggingface/transformers': 8,
    # Single cell
    'scverse/scvi-tools': 10,
    'YosefLab/scvi-tools': 10,
    'satijalab/seurat': 9,
}

# Known high-value HF models
PRIORITY_HF_MODELS = {
    # Pathology foundation models
    'MahmoodLab/UNI': {'domain': 'pathology', 'priority': 10},
    'MahmoodLab/CONCH': {'domain': 'pathology', 'priority': 10},
    'prov-gigapath/prov-gigapath': {'domain': 'pathology', 'priority': 10},
    'owkin/phikon': {'domain': 'pathology', 'priority': 9},
    'paige-ai/Virchow': {'domain': 'pathology', 'priority': 9},
    # General vision
    'facebook/dino-vitb16': {'domain': 'vision', 'priority': 8},
    'facebook/dinov2-large': {'domain': 'vision', 'priority': 8},
    'google/vit-base-patch16-224': {'domain': 'vision', 'priority': 7},
    # Single cell
    'genentech/scBERT': {'domain': 'single_cell', 'priority': 8},
    'ctheodoris/Geneformer': {'domain': 'single_cell', 'priority': 9},
}


def get_db_connection():
    """Get database connection."""
    dsn = os.environ.get('POSTGRES_DSN', 'dbname=polymath user=polymath host=/var/run/postgresql')
    return psycopg2.connect(dsn)


def extract_github_urls(text: str) -> List[Tuple[str, str]]:
    """Extract GitHub owner/repo pairs from text."""
    pattern = r'github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    # Clean up repo names (remove trailing .git, periods, etc)
    return [(owner, repo.rstrip('.').rstrip(',').replace('.git', ''))
            for owner, repo in matches]


def extract_hf_models(text: str) -> List[str]:
    """Extract HuggingFace model identifiers from text."""
    patterns = [
        r'huggingface\.co/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)',
        r'from_pretrained\(["\']([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)["\']',
        r'model_id\s*=\s*["\']([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)["\']',
    ]

    models = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        models.update(matches)

    return list(models)


def discover_github_from_papers(conn) -> List[Dict]:
    """Discover GitHub repos mentioned in papers."""
    cur = conn.cursor()

    # Search passages for GitHub URLs
    cur.execute("""
        SELECT p.passage_id, p.passage_text, d.title, d.doc_id
        FROM passages p
        JOIN documents d ON p.doc_id = d.doc_id
        WHERE p.passage_text ~* 'github\.com/[a-z0-9_-]+/[a-z0-9_.-]+'
        LIMIT 1000
    """)

    repo_mentions = defaultdict(lambda: {'count': 0, 'papers': set(), 'contexts': []})

    for passage_id, text, title, doc_id in cur.fetchall():
        repos = extract_github_urls(text)
        for owner, repo in repos:
            key = f"{owner}/{repo}"
            repo_mentions[key]['count'] += 1
            repo_mentions[key]['papers'].add(title)
            repo_mentions[key]['contexts'].append(text[:200])

    # Check which are already in queue
    cur.execute("SELECT CONCAT(owner, '/', repo_name) FROM repo_queue")
    existing = {row[0] for row in cur.fetchall()}

    # Format results
    results = []
    for repo_key, data in repo_mentions.items():
        if repo_key in existing:
            continue

        owner, repo = repo_key.split('/', 1)
        priority = PRIORITY_REPOS.get(repo_key, 5)

        # Boost priority based on mentions
        if data['count'] >= 3:
            priority = min(10, priority + 2)
        elif data['count'] >= 2:
            priority = min(10, priority + 1)

        results.append({
            'owner': owner,
            'repo': repo,
            'full_name': repo_key,
            'mentions': data['count'],
            'papers': list(data['papers'])[:5],
            'priority': priority,
            'context': data['contexts'][0] if data['contexts'] else '',
            'status': 'new'
        })

    # Sort by priority and mentions
    results.sort(key=lambda x: (-x['priority'], -x['mentions']))
    return results


def discover_hf_from_papers(conn) -> List[Dict]:
    """Discover HuggingFace models mentioned in papers."""
    cur = conn.cursor()

    # Search passages for HF model references
    cur.execute("""
        SELECT p.passage_id, p.passage_text, d.title, d.doc_id
        FROM passages p
        JOIN documents d ON p.doc_id = d.doc_id
        WHERE p.passage_text ~* 'huggingface|from_pretrained|model_id'
        LIMIT 1000
    """)

    model_mentions = defaultdict(lambda: {'count': 0, 'papers': set(), 'contexts': []})

    for passage_id, text, title, doc_id in cur.fetchall():
        models = extract_hf_models(text)
        for model_id in models:
            model_mentions[model_id]['count'] += 1
            model_mentions[model_id]['papers'].add(title)
            model_mentions[model_id]['contexts'].append(text[:200])

    # Check existing
    cur.execute("SELECT model_id FROM hf_model_mentions")
    existing = {row[0] for row in cur.fetchall()}

    # Format results
    results = []
    for model_id, data in model_mentions.items():
        priority_info = PRIORITY_HF_MODELS.get(model_id, {'domain': 'unknown', 'priority': 5})

        # Boost based on mentions
        priority = priority_info['priority']
        if data['count'] >= 3:
            priority = min(10, priority + 2)

        results.append({
            'model_id': model_id,
            'mentions': data['count'],
            'papers': list(data['papers'])[:5],
            'priority': priority,
            'domain': priority_info.get('domain', 'unknown'),
            'context': data['contexts'][0] if data['contexts'] else '',
            'in_db': model_id in existing
        })

    # Add priority models not yet mentioned
    for model_id, info in PRIORITY_HF_MODELS.items():
        if model_id not in model_mentions and model_id not in existing:
            results.append({
                'model_id': model_id,
                'mentions': 0,
                'papers': [],
                'priority': info['priority'],
                'domain': info['domain'],
                'context': 'Priority model - not yet mentioned in papers',
                'in_db': False
            })

    results.sort(key=lambda x: (-x['priority'], -x['mentions']))
    return results


def find_knowledge_gaps(conn) -> List[Dict]:
    """Find topics we have papers on but no code for."""
    cur = conn.cursor()

    # Get concepts from papers
    cur.execute("""
        SELECT concept_name, COUNT(*) as cnt
        FROM passage_concepts
        WHERE concept_type IN ('method', 'algorithm', 'tool')
        GROUP BY concept_name
        HAVING COUNT(*) >= 3
        ORDER BY cnt DESC
        LIMIT 100
    """)

    paper_concepts = {row[0]: row[1] for row in cur.fetchall()}

    # Get concepts from code
    cur.execute("""
        SELECT UNNEST(concepts), COUNT(*) as cnt
        FROM code_chunks
        WHERE concepts IS NOT NULL
        GROUP BY UNNEST(concepts)
    """)

    code_concepts = {row[0]: row[1] for row in cur.fetchall()}

    # Find gaps - concepts in papers but not in code
    gaps = []
    for concept, paper_count in paper_concepts.items():
        code_count = code_concepts.get(concept, 0)

        # Gap if we have papers but little/no code
        if code_count < paper_count * 0.2:
            gaps.append({
                'concept': concept,
                'paper_mentions': paper_count,
                'code_mentions': code_count,
                'gap_ratio': (paper_count - code_count) / paper_count,
                'suggested_search': f'{concept} implementation github'
            })

    gaps.sort(key=lambda x: -x['gap_ratio'])
    return gaps[:30]


def generate_recommendations(conn) -> Dict:
    """Generate full asset recommendations."""
    recommendations = {
        'timestamp': datetime.now().isoformat(),
        'github_repos': [],
        'hf_models': [],
        'knowledge_gaps': [],
        'priority_queue': [],
    }

    # GitHub repos
    github_repos = discover_github_from_papers(conn)
    recommendations['github_repos'] = github_repos[:20]

    # HuggingFace models
    hf_models = discover_hf_from_papers(conn)
    recommendations['hf_models'] = hf_models[:20]

    # Knowledge gaps
    gaps = find_knowledge_gaps(conn)
    recommendations['knowledge_gaps'] = gaps[:10]

    # Priority queue (combine all)
    priority = []
    for repo in github_repos[:10]:
        priority.append({
            'type': 'github',
            'id': repo['full_name'],
            'priority': repo['priority'],
            'action': f"python scripts/github_ingest.py {repo['full_name']}"
        })

    for model in hf_models[:5]:
        if not model['in_db']:
            priority.append({
                'type': 'huggingface',
                'id': model['model_id'],
                'priority': model['priority'],
                'action': f"Add to HF_MODELS_REFERENCE.md"
            })

    priority.sort(key=lambda x: -x['priority'])
    recommendations['priority_queue'] = priority

    return recommendations


def save_hf_reference(models: List[Dict], output_path: Path = None):
    """Save HuggingFace models to reference file."""
    if output_path is None:
        output_path = Path('/home/user/polymath-v3/HF_MODELS_REFERENCE.md')

    # Group by domain
    by_domain = defaultdict(list)
    for model in models:
        by_domain[model['domain']].append(model)

    content = """# HuggingFace Models Reference

> Auto-generated list of HuggingFace models relevant to our research.
> These are for reference - not downloaded, just tracked.

**Generated:** {timestamp}

---

""".format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'))

    for domain in sorted(by_domain.keys()):
        content += f"\n## {domain.replace('_', ' ').title()}\n\n"
        content += "| Model | Priority | Mentions | Papers |\n"
        content += "|-------|----------|----------|--------|\n"

        for model in sorted(by_domain[domain], key=lambda x: -x['priority']):
            papers = ', '.join(model['papers'][:2]) if model['papers'] else '-'
            if len(papers) > 50:
                papers = papers[:47] + '...'
            content += f"| `{model['model_id']}` | {model['priority']} | {model['mentions']} | {papers} |\n"

    # Quick reference section
    content += """
---

## Quick Reference

### Loading Models (Python)

```python
from transformers import AutoModel, AutoImageProcessor

# Pathology foundation models
model = AutoModel.from_pretrained("MahmoodLab/UNI")
processor = AutoImageProcessor.from_pretrained("MahmoodLab/UNI")

# Vision models
model = AutoModel.from_pretrained("facebook/dinov2-large")
```

### Priority Models for Download

When ready to download, prioritize these:
1. MahmoodLab/UNI - Pathology foundation (best for H&E)
2. MahmoodLab/CONCH - Multi-modal pathology
3. prov-gigapath/prov-gigapath - Large pathology foundation
4. ctheodoris/Geneformer - Single-cell foundation

---

*This file is auto-generated. Do not edit directly.*
"""

    output_path.write_text(content)
    logger.info(f"Saved HF reference to {output_path}")
    return output_path


def print_github_recommendations(repos: List[Dict]):
    """Print GitHub repo recommendations."""
    print("\n" + "=" * 80)
    print("GITHUB REPOSITORY RECOMMENDATIONS")
    print("=" * 80)
    print(f"\n{'Repo':<45} {'Pri':<4} {'Mentions':<8} {'Papers':<30}")
    print("-" * 80)

    for repo in repos[:20]:
        papers = repo['papers'][0][:27] + '...' if repo['papers'] else '-'
        print(f"{repo['full_name']:<45} {repo['priority']:<4} {repo['mentions']:<8} {papers:<30}")

    print("\n" + "-" * 40)
    print("Quick ingest commands:")
    for repo in repos[:5]:
        print(f"  python scripts/github_ingest.py {repo['full_name']}")


def print_hf_recommendations(models: List[Dict]):
    """Print HuggingFace model recommendations."""
    print("\n" + "=" * 80)
    print("HUGGINGFACE MODEL RECOMMENDATIONS")
    print("=" * 80)
    print(f"\n{'Model':<45} {'Pri':<4} {'Domain':<15} {'Mentions':<8}")
    print("-" * 80)

    for model in models[:20]:
        status = "✓" if model['in_db'] else " "
        print(f"{status} {model['model_id']:<43} {model['priority']:<4} {model['domain']:<15} {model['mentions']:<8}")

    print("\n" + "-" * 40)
    print("Legend: ✓ = already tracked")


def main():
    parser = argparse.ArgumentParser(description='Asset Discovery for Polymath')
    parser.add_argument('--github', '-g', action='store_true', help='Discover GitHub repos')
    parser.add_argument('--hf', action='store_true', help='Discover HuggingFace models')
    parser.add_argument('--gaps', action='store_true', help='Find knowledge gaps')
    parser.add_argument('--recommend', '-r', action='store_true', help='Full recommendations')
    parser.add_argument('--save-hf', action='store_true', help='Save HF models to reference file')
    parser.add_argument('--add-to-queue', action='store_true', help='Add discovered repos to queue')
    parser.add_argument('--output', '-o', help='Output file for recommendations (JSON)')
    args = parser.parse_args()

    conn = get_db_connection()

    if args.github or args.recommend:
        repos = discover_github_from_papers(conn)
        print_github_recommendations(repos)

        if args.add_to_queue:
            cur = conn.cursor()
            added = 0
            for repo in repos[:20]:
                cur.execute("""
                    INSERT INTO repo_queue (owner, repo_name, repo_url, source, priority, status)
                    VALUES (%s, %s, %s, %s, %s, 'pending')
                    ON CONFLICT DO NOTHING
                """, (repo['owner'], repo['repo'],
                      f"https://github.com/{repo['full_name']}",
                      'discovery', repo['priority']))
                if cur.rowcount:
                    added += 1
            conn.commit()
            print(f"\nAdded {added} repos to queue")

    if args.hf or args.recommend:
        models = discover_hf_from_papers(conn)
        print_hf_recommendations(models)

        if args.save_hf:
            save_hf_reference(models)

    if args.gaps or args.recommend:
        gaps = find_knowledge_gaps(conn)
        print("\n" + "=" * 80)
        print("KNOWLEDGE GAPS (Papers without Code)")
        print("=" * 80)
        print(f"\n{'Concept':<30} {'Papers':<8} {'Code':<8} {'Gap':<8}")
        print("-" * 60)
        for gap in gaps[:15]:
            print(f"{gap['concept']:<30} {gap['paper_mentions']:<8} {gap['code_mentions']:<8} {gap['gap_ratio']:.0%}")

    if args.output:
        recommendations = generate_recommendations(conn)
        with open(args.output, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        print(f"\nSaved recommendations to {args.output}")

    conn.close()


if __name__ == '__main__':
    main()
