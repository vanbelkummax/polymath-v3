#!/usr/bin/env python3
"""
Asset Detection for Polymath System.

Detects GitHub repositories, HuggingFace models, and citations in paper text.
Queues assets for download/cataloging and links them to source papers.

Usage:
    from lib.ingest.asset_detector import AssetDetector

    detector = AssetDetector(pg_conn)
    assets = await detector.detect_and_store(doc_id, passages)
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)


@dataclass
class DetectedAsset:
    """A detected asset (repo, model, or citation)."""
    asset_type: str  # 'github', 'huggingface', 'citation'
    identifier: str
    context: str
    passage_id: str
    extra: Dict[str, Any] = field(default_factory=dict)


class AssetDetector:
    """Detect and catalog assets referenced in papers."""

    # GitHub patterns
    GITHUB_PATTERNS = [
        r'github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)',
        r'github\.io/([a-zA-Z0-9_-]+)',
    ]

    # HuggingFace patterns
    HF_PATTERNS = [
        r'huggingface\.co/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)',
        r'from_pretrained\s*\(\s*["\']([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)["\']',
        r'(?:model|checkpoint|weights).*["\']([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)["\']',
    ]

    # Known model names (without org prefix) - commonly referenced
    KNOWN_MODELS = {
        # Language models
        'bert-base', 'bert-large', 'bert-base-uncased', 'bert-large-uncased',
        'roberta-base', 'roberta-large', 'gpt2', 'gpt2-medium', 'gpt2-large',
        'gpt-neo', 'gpt-j', 'llama', 'llama-2', 'llama-3', 'mistral', 'phi', 'gemma',

        # Vision models
        'vit-base', 'vit-large', 'vit-huge', 'dinov2', 'dino',
        'clip', 'openclip', 'siglip', 'resnet', 'resnet50', 'resnet101',
        'efficientnet', 'convnext', 'swin', 'swin-transformer',

        # Pathology foundation models
        'UNI', 'CONCH', 'HIPT', 'CTransPath', 'Phikon', 'Virchow',
        'GigaPath', 'PLIP', 'BiomedCLIP', 'PubMedCLIP',

        # Diffusion models
        'stable-diffusion', 'sdxl', 'controlnet', 'ip-adapter',

        # Single-cell models
        'scGPT', 'scBERT', 'Geneformer', 'scFoundation',
    }

    # DOI pattern
    DOI_PATTERN = r'(?:doi\.org/|DOI:?\s*)(10\.\d{4,}/[^\s\]>)]+)'

    def __init__(self, conn=None):
        """
        Initialize detector.

        Args:
            conn: Postgres connection (optional, will create if needed)
        """
        self.conn = conn
        self._hf_api = None

    @property
    def hf_api(self):
        """Lazy load HuggingFace API."""
        if self._hf_api is None:
            try:
                from huggingface_hub import HfApi
                self._hf_api = HfApi()
            except ImportError:
                logger.warning("huggingface_hub not installed, HF metadata fetch disabled")
                self._hf_api = False
        return self._hf_api

    def detect_all(self, passages: List[Dict]) -> Dict[str, List[DetectedAsset]]:
        """
        Detect all assets in passages.

        Args:
            passages: List of passage dicts with 'passage_id' and 'passage_text'

        Returns:
            Dict with 'github', 'huggingface', 'citations' lists
        """
        assets = {
            'github': [],
            'huggingface': [],
            'citations': []
        }

        for passage in passages:
            text = passage.get('passage_text', '')
            passage_id = str(passage.get('passage_id', ''))

            # GitHub repos
            for pattern in self.GITHUB_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    repo = match.group(1).rstrip('.')
                    # Clean up common suffixes
                    repo = re.sub(r'\.(git|zip|tar\.gz)$', '', repo)

                    # Parse owner/name
                    parts = repo.split('/')
                    owner = parts[0] if len(parts) > 0 else None
                    name = parts[1] if len(parts) > 1 else parts[0]

                    assets['github'].append(DetectedAsset(
                        asset_type='github',
                        identifier=f'https://github.com/{repo}',
                        context=self._get_context(text, match.start()),
                        passage_id=passage_id,
                        extra={'owner': owner, 'name': name}
                    ))

            # HuggingFace models (explicit URLs)
            for pattern in self.HF_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    model_id = match.group(1)

                    # Parse org/model
                    parts = model_id.split('/')
                    org = parts[0] if len(parts) > 1 else None
                    name = parts[-1]

                    assets['huggingface'].append(DetectedAsset(
                        asset_type='huggingface',
                        identifier=model_id,
                        context=self._get_context(text, match.start()),
                        passage_id=passage_id,
                        extra={'organization': org, 'model_name': name}
                    ))

            # Known models (by name, need resolution)
            for model in self.KNOWN_MODELS:
                pattern = rf'\b{re.escape(model)}\b'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    assets['huggingface'].append(DetectedAsset(
                        asset_type='huggingface',
                        identifier=model,
                        context=self._get_context(text, match.start()),
                        passage_id=passage_id,
                        extra={'needs_resolution': True}
                    ))

            # DOIs (citations)
            for match in re.finditer(self.DOI_PATTERN, text, re.IGNORECASE):
                doi = match.group(1).rstrip('.,;')
                assets['citations'].append(DetectedAsset(
                    asset_type='citation',
                    identifier=doi,
                    context=self._get_context(text, match.start()),
                    passage_id=passage_id
                ))

        # Deduplicate
        for asset_type in assets:
            seen = set()
            unique = []
            for asset in assets[asset_type]:
                key = asset.identifier.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(asset)
            assets[asset_type] = unique

        return assets

    def _get_context(self, text: str, pos: int, window: int = 200) -> str:
        """Get surrounding context for an asset mention."""
        start = max(0, pos - window)
        end = min(len(text), pos + window)
        return text[start:end]

    def store_github_repos(self, doc_id: str, repos: List[DetectedAsset]):
        """Store detected GitHub repos."""
        if not repos:
            return

        cur = self.conn.cursor()

        for repo in repos:
            # Add to paper_repos
            cur.execute("""
                INSERT INTO paper_repos (doc_id, repo_url, repo_owner, repo_name, passage_id, context)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id, repo_url) DO NOTHING
            """, (
                doc_id,
                repo.identifier,
                repo.extra.get('owner'),
                repo.extra.get('name'),
                repo.passage_id if repo.passage_id else None,
                repo.context[:500]
            ))

            # Add to queue (or update priority)
            cur.execute("""
                INSERT INTO repo_queue (repo_url, repo_owner, repo_name, first_seen_doc_id, source_doc_count)
                VALUES (%s, %s, %s, %s, 1)
                ON CONFLICT (repo_url) DO UPDATE SET
                    source_doc_count = repo_queue.source_doc_count + 1,
                    priority = repo_queue.priority + 1,
                    updated_at = now()
            """, (
                repo.identifier,
                repo.extra.get('owner'),
                repo.extra.get('name'),
                doc_id
            ))

        self.conn.commit()
        logger.info(f"Stored {len(repos)} GitHub repos for doc {doc_id}")

    def store_hf_models(self, doc_id: str, models: List[DetectedAsset]):
        """Store detected HuggingFace models."""
        if not models:
            return

        cur = self.conn.cursor()

        for model in models:
            model_id = model.identifier

            # Check if model exists
            cur.execute("SELECT model_id FROM hf_models WHERE model_id = %s", (model_id,))
            exists = cur.fetchone()

            if not exists:
                # Create new model entry
                cur.execute("""
                    INSERT INTO hf_models (model_id, model_name, organization, first_seen_doc_id)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (model_id) DO NOTHING
                """, (
                    model_id,
                    model.extra.get('model_name', model_id.split('/')[-1]),
                    model.extra.get('organization'),
                    doc_id
                ))
            else:
                # Increment citation count
                cur.execute("""
                    UPDATE hf_models SET citation_count = citation_count + 1
                    WHERE model_id = %s
                """, (model_id,))

            # Link to paper
            cur.execute("""
                INSERT INTO paper_hf_models (doc_id, model_id, passage_id, context)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (doc_id, model_id) DO NOTHING
            """, (
                doc_id,
                model_id,
                model.passage_id if model.passage_id else None,
                model.context[:500]
            ))

        self.conn.commit()
        logger.info(f"Stored {len(models)} HF models for doc {doc_id}")

    def store_citations(self, doc_id: str, citations: List[DetectedAsset]):
        """Store detected citations (DOIs)."""
        if not citations:
            return

        cur = self.conn.cursor()

        for citation in citations:
            doi = citation.identifier

            # Check if cited paper is in our corpus
            cur.execute("SELECT doc_id FROM documents WHERE doi = %s", (doi,))
            cited_doc = cur.fetchone()

            cur.execute("""
                INSERT INTO citation_links (citing_doc_id, cited_doc_id, cited_doi, citation_context, in_corpus)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (citing_doc_id, cited_doi) DO NOTHING
            """, (
                doc_id,
                cited_doc[0] if cited_doc else None,
                doi,
                citation.context[:500],
                cited_doc is not None
            ))

        self.conn.commit()
        logger.info(f"Stored {len(citations)} citations for doc {doc_id}")

    def detect_and_store(self, doc_id: str, passages: List[Dict]) -> Dict[str, int]:
        """
        Detect all assets and store them.

        Args:
            doc_id: Document ID
            passages: List of passage dicts

        Returns:
            Dict with counts: {'github': n, 'huggingface': n, 'citations': n}
        """
        assets = self.detect_all(passages)

        self.store_github_repos(doc_id, assets['github'])
        self.store_hf_models(doc_id, assets['huggingface'])
        self.store_citations(doc_id, assets['citations'])

        return {
            'github': len(assets['github']),
            'huggingface': len(assets['huggingface']),
            'citations': len(assets['citations'])
        }

    def fetch_hf_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Fetch HuggingFace model metadata (without downloading weights).

        Args:
            model_id: Model identifier (e.g., "facebook/dinov2-base")

        Returns:
            Metadata dict
        """
        if not self.hf_api:
            return {'model_id': model_id, 'error': 'HF API not available'}

        try:
            info = self.hf_api.model_info(model_id)
            return {
                'model_id': model_id,
                'model_name': info.modelId.split('/')[-1] if info.modelId else model_id,
                'organization': info.modelId.split('/')[0] if '/' in (info.modelId or '') else None,
                'pipeline_tag': info.pipeline_tag,
                'library_name': info.library_name,
                'architectures': (info.config or {}).get('architectures', []),
                'tags': info.tags or [],
                'downloads_30d': info.downloads,
                'likes': info.likes,
                'model_card_summary': ((info.card_data or {}).get('description') or '')[:500],
            }
        except Exception as e:
            logger.warning(f"Failed to fetch HF metadata for {model_id}: {e}")
            return {'model_id': model_id, 'error': str(e)}

    def refresh_hf_metadata(self, limit: int = 50):
        """Refresh HF metadata for models missing it."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT model_id FROM hf_models
            WHERE last_metadata_fetch IS NULL
            OR last_metadata_fetch < now() - interval '7 days'
            ORDER BY citation_count DESC
            LIMIT %s
        """, (limit,))

        models = cur.fetchall()
        updated = 0

        for (model_id,) in models:
            metadata = self.fetch_hf_metadata(model_id)

            if 'error' not in metadata:
                cur.execute("""
                    UPDATE hf_models SET
                        model_name = %s,
                        organization = %s,
                        pipeline_tag = %s,
                        library_name = %s,
                        architectures = %s,
                        tags = %s,
                        downloads_30d = %s,
                        likes = %s,
                        model_card_summary = %s,
                        last_metadata_fetch = now()
                    WHERE model_id = %s
                """, (
                    metadata.get('model_name'),
                    metadata.get('organization'),
                    metadata.get('pipeline_tag'),
                    metadata.get('library_name'),
                    metadata.get('architectures'),
                    metadata.get('tags'),
                    metadata.get('downloads_30d'),
                    metadata.get('likes'),
                    metadata.get('model_card_summary'),
                    model_id
                ))
                updated += 1

        self.conn.commit()
        logger.info(f"Refreshed metadata for {updated} HF models")
        return updated


# ============================================================
# CLI for testing
# ============================================================

if __name__ == '__main__':
    import sys

    # Test with sample text
    test_text = """
    We used the pre-trained UNI model (https://github.com/mahmoodlab/UNI)
    for feature extraction. The model was loaded using
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True).

    Our approach builds on HIPT (Chen et al., 2022, doi:10.1109/CVPR52688.2022.00464)
    and the spatial transcriptomics framework from squidpy (github.com/scverse/squidpy).

    We also compared against facebook/dinov2-base and CONCH from the
    HuggingFace hub (huggingface.co/MahmoodLab/CONCH).
    """

    detector = AssetDetector()
    passages = [{'passage_id': 'test-1', 'passage_text': test_text}]

    assets = detector.detect_all(passages)

    print("GitHub repos:")
    for a in assets['github']:
        print(f"  - {a.identifier}")

    print("\nHuggingFace models:")
    for a in assets['huggingface']:
        print(f"  - {a.identifier}")

    print("\nCitations:")
    for a in assets['citations']:
        print(f"  - {a.identifier}")
