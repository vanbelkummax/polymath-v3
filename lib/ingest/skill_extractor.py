#!/usr/bin/env python3
"""
Skill Extraction for Polymath System.

Extracts actionable skills (methods, algorithms, workflows) from paper passages
using LLM analysis. Identifies cross-domain transfer potential.

IMPORTANT: Skills are written to ~/.claude/skills_drafts/ only.
Use scripts/promote_skill.py to promote validated skills to ~/.claude/skills/.

Usage:
    from lib.ingest.skill_extractor import SkillExtractor

    extractor = SkillExtractor(pg_conn)
    skills = await extractor.extract_and_store(doc_id, passages, concepts)
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import psycopg2
import numpy as np

logger = logging.getLogger(__name__)

# Skill output directories
SKILLS_DRAFTS_DIR = Path.home() / ".claude" / "skills_drafts"


# ============================================================
# Skill Extraction Prompt
# ============================================================

SKILL_EXTRACTION_PROMPT = """Analyze this scientific paper passage for extractable, actionable skills.

Paper: {title}
Passage:
{passage_text}

Detected concepts: {concepts}

A "skill" is actionable knowledge that someone could apply to solve problems. Extract ANY of:

1. **Procedural Method**: Step-by-step workflow that can be repeated
2. **Algorithm**: Computational technique (pseudocode, equations, key logic)
3. **Heuristic**: Decision rule for choosing between approaches
4. **Parameter Guidance**: Empirical findings about settings/values
5. **Failure Mode**: What breaks and how to diagnose/fix
6. **Workflow**: Multi-step pipeline combining multiple methods

For each skill found, provide:

```json
{{
  "skill_name": "descriptive-kebab-case-name",
  "skill_type": "method|algorithm|heuristic|parameter|failure_mode|workflow",
  "description": "What it does and why useful (2-3 sentences)",
  "prerequisites": ["python", "pytorch", ...],
  "steps": [
    {{"order": 1, "description": "First step", "code": "optional code snippet"}}
  ],
  "parameters": [
    {{"name": "k", "default": 15, "range": "5-50", "guidance": "Higher for sparser data"}}
  ],
  "failure_modes": ["When it fails and why"],
  "cross_domain_potential": {{
    "abstract_principle": "The generalizable mathematical/logical core",
    "original_domain": "spatial_transcriptomics",
    "transferable_to": ["ecology", "urban_planning"],
    "transfer_insight": "Why it transfers - what structure is shared"
  }}
}}
```

Return a JSON array of skills. Return `[]` if no actionable skills found.
Focus on NOVEL, SPECIFIC skills - not generic knowledge like "use normalization".

JSON array:"""


# ============================================================
# Domain Abstractions for Cross-Domain Detection
# ============================================================

DOMAIN_ABSTRACTIONS = {
    "spatial_autocorrelation": {
        "domains": ["spatial_transcriptomics", "geostatistics", "time_series", "image_analysis", "ecology"],
        "description": "Detecting non-random clustering in space or time",
        "keywords": ["moran", "geary", "variogram", "correlogram", "spatial correlation", "autocorrelation"],
    },
    "graph_message_passing": {
        "domains": ["gnn", "belief_propagation", "cellular_automata", "social_networks", "molecular_dynamics"],
        "description": "Iterative information exchange between connected nodes",
        "keywords": ["message passing", "gcn", "gat", "mpnn", "graph neural", "node aggregation"],
    },
    "attention_mechanism": {
        "domains": ["nlp", "computer_vision", "genomics", "recommendation", "spatial_transcriptomics"],
        "description": "Learned weighted aggregation of relevant context",
        "keywords": ["attention", "transformer", "self-attention", "cross-attention", "multi-head"],
    },
    "contrastive_learning": {
        "domains": ["computer_vision", "nlp", "molecular", "audio", "pathology"],
        "description": "Learning by comparing positive and negative pairs",
        "keywords": ["contrastive", "simclr", "moco", "clip", "infonce", "triplet"],
    },
    "optimal_transport": {
        "domains": ["single_cell", "domain_adaptation", "economics", "logistics", "image_registration"],
        "description": "Finding minimum-cost mappings between distributions",
        "keywords": ["wasserstein", "sinkhorn", "optimal transport", "earth mover", "ot"],
    },
    "variational_inference": {
        "domains": ["vae", "bayesian_nn", "topic_models", "phylogenetics", "single_cell"],
        "description": "Approximating intractable posteriors with tractable distributions",
        "keywords": ["variational", "elbo", "kl divergence", "reparameterization", "vae"],
    },
    "permutation_testing": {
        "domains": ["spatial_statistics", "genomics", "ecology", "neuroscience", "epidemiology"],
        "description": "Null distribution via random shuffling",
        "keywords": ["permutation", "bootstrap", "monte carlo", "shuffle", "null distribution"],
    },
    "dimensionality_reduction": {
        "domains": ["single_cell", "nlp", "computer_vision", "finance", "genomics"],
        "description": "Projecting high-dimensional data to interpretable space",
        "keywords": ["pca", "umap", "tsne", "diffusion map", "embedding", "manifold"],
    },
    "multiple_instance_learning": {
        "domains": ["pathology", "drug_discovery", "remote_sensing", "video_classification"],
        "description": "Learning from bags of instances with bag-level labels",
        "keywords": ["mil", "multiple instance", "bag", "abmil", "transmil", "attention mil"],
    },
}


@dataclass
class ExtractedSkill:
    """An extracted skill from a paper."""
    skill_name: str
    skill_type: str
    description: str
    prerequisites: List[str] = field(default_factory=list)
    inputs: List[Dict] = field(default_factory=list)
    outputs: List[Dict] = field(default_factory=list)
    steps: List[Dict] = field(default_factory=list)
    parameters: List[Dict] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    original_domain: Optional[str] = None
    transferable_to: List[str] = field(default_factory=list)
    transfer_insights: Optional[str] = None
    abstract_principle: Optional[str] = None
    confidence: float = 0.7
    source_passage_id: Optional[str] = None
    source_doc_id: Optional[str] = None


class SkillExtractor:
    """Extract actionable skills from paper passages using LLM."""

    def __init__(self, conn=None, model: str = "gemini-2.5-flash"):
        """
        Initialize extractor.

        Args:
            conn: Postgres connection
            model: LLM model to use for extraction
        """
        self.conn = conn
        self.model = model
        self._embedder = None

    @property
    def embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            from lib.embeddings.bge_m3 import Embedder
            self._embedder = Embedder()
        return self._embedder

    async def extract_from_passage(
        self,
        passage: Dict,
        concepts: List[Dict],
        title: str = "Unknown"
    ) -> List[ExtractedSkill]:
        """
        Extract skills from a single passage.

        Args:
            passage: Passage dict with 'passage_id', 'passage_text'
            concepts: List of concept dicts with 'concept_name', 'concept_type'
            title: Paper title

        Returns:
            List of extracted skills
        """
        passage_text = passage.get('passage_text', '')
        passage_id = str(passage.get('passage_id', ''))

        # Filter concepts for this passage
        passage_concepts = [c['concept_name'] for c in concepts
                          if str(c.get('passage_id', '')) == passage_id]

        # Build prompt
        prompt = SKILL_EXTRACTION_PROMPT.format(
            title=title,
            passage_text=passage_text[:3000],  # Limit length
            concepts=', '.join(passage_concepts[:20])
        )

        # Call LLM
        response = await self._call_llm(prompt)
        skills_data = self._parse_response(response)

        # Convert to ExtractedSkill objects
        skills = []
        for data in skills_data:
            cross_domain = data.get('cross_domain_potential', {})

            skill = ExtractedSkill(
                skill_name=data.get('skill_name', 'unknown-skill'),
                skill_type=data.get('skill_type', 'method'),
                description=data.get('description', ''),
                prerequisites=data.get('prerequisites', []),
                steps=data.get('steps', []),
                parameters=data.get('parameters', []),
                failure_modes=data.get('failure_modes', []),
                original_domain=cross_domain.get('original_domain'),
                transferable_to=cross_domain.get('transferable_to', []),
                transfer_insights=cross_domain.get('transfer_insight'),
                abstract_principle=cross_domain.get('abstract_principle'),
                source_passage_id=passage_id,
            )

            # Detect additional cross-domain potential from keywords
            self._enhance_cross_domain(skill, passage_text)

            skills.append(skill)

        return skills

    def _enhance_cross_domain(self, skill: ExtractedSkill, text: str):
        """Enhance cross-domain detection using keyword matching."""
        text_lower = text.lower()

        for abstraction, info in DOMAIN_ABSTRACTIONS.items():
            # Check if keywords match
            matches = sum(1 for kw in info['keywords'] if kw in text_lower)

            if matches >= 2:  # At least 2 keywords match
                # Add to transferable domains if not already there
                for domain in info['domains']:
                    if domain != skill.original_domain and domain not in skill.transferable_to:
                        skill.transferable_to.append(domain)

                # Set abstract principle if not set
                if not skill.abstract_principle:
                    skill.abstract_principle = info['description']

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for skill extraction."""
        try:
            import google.generativeai as genai
            from lib.config import config

            genai.configure(api_key=config.GEMINI_API_KEY)
            model = genai.GenerativeModel(self.model)

            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=4096,
                )
            )

            return response.text

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "[]"

    def _parse_response(self, response: str) -> List[Dict]:
        """Parse LLM response to extract skills JSON."""
        try:
            # Try to extract JSON from response
            # Handle markdown code blocks
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                response = response[start:end]
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                response = response[start:end]

            # Find JSON array
            start = response.find('[')
            end = response.rfind(']') + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)

            return []

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse skill extraction response: {e}")
            return []

    async def extract_from_passages(
        self,
        passages: List[Dict],
        concepts: List[Dict],
        title: str = "Unknown",
        doc_id: str = None
    ) -> List[ExtractedSkill]:
        """
        Extract skills from multiple passages.

        Focuses on passages with method/algorithm concepts.

        Args:
            passages: List of passage dicts
            concepts: List of concept dicts
            title: Paper title
            doc_id: Document ID

        Returns:
            List of extracted skills
        """
        all_skills = []

        # Filter to passages with actionable concepts
        method_types = {'method', 'algorithm', 'mechanism', 'workflow'}
        passage_ids_with_methods = {
            str(c.get('passage_id', ''))
            for c in concepts
            if c.get('concept_type', '').lower() in method_types
        }

        # Also include passages with certain keywords
        method_keywords = ['we propose', 'our method', 'algorithm', 'procedure',
                         'step 1', 'first,', 'pipeline', 'workflow']

        for passage in passages:
            passage_id = str(passage.get('passage_id', ''))
            text = passage.get('passage_text', '').lower()

            # Check if passage should be processed
            has_method_concept = passage_id in passage_ids_with_methods
            has_method_keyword = any(kw in text for kw in method_keywords)

            if has_method_concept or has_method_keyword:
                skills = await self.extract_from_passage(
                    passage,
                    [c for c in concepts if str(c.get('passage_id', '')) == passage_id],
                    title
                )

                for skill in skills:
                    skill.source_doc_id = doc_id
                    all_skills.append(skill)

        return all_skills

    def store_skills(self, skills: List[ExtractedSkill], doc_id: str):
        """Store extracted skills in database."""
        if not skills:
            return []

        cur = self.conn.cursor()
        skill_ids = []

        for skill in skills:
            skill_id = str(uuid.uuid4())

            # Generate embedding for skill description
            embedding = None
            try:
                embedding = self.embedder.encode(skill.description)
            except Exception as e:
                logger.warning(f"Failed to generate skill embedding: {e}")

            # Insert skill
            cur.execute("""
                INSERT INTO paper_skills (
                    skill_id, skill_name, skill_type, description,
                    prerequisites, steps, parameters, failure_modes,
                    original_domain, transferable_to, transfer_insights, abstract_principle,
                    source_doc_ids, source_passages, embedding, confidence, status
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'draft'
                )
                RETURNING skill_id
            """, (
                skill_id,
                skill.skill_name,
                skill.skill_type,
                skill.description,
                skill.prerequisites,
                json.dumps(skill.steps) if skill.steps else None,
                json.dumps(skill.parameters) if skill.parameters else None,
                skill.failure_modes,
                skill.original_domain,
                skill.transferable_to,
                skill.transfer_insights,
                skill.abstract_principle,
                [doc_id],
                json.dumps([{'passage_id': skill.source_passage_id, 'relevance': 1.0}]),
                embedding.tolist() if embedding is not None else None,
                skill.confidence
            ))

            result = cur.fetchone()
            if result:
                skill_ids.append(result[0])

            # Link to paper
            cur.execute("""
                INSERT INTO paper_skill_contributions (doc_id, skill_id, contribution_type)
                VALUES (%s, %s, 'origin')
                ON CONFLICT (doc_id, skill_id) DO NOTHING
            """, (doc_id, skill_id))

        self.conn.commit()
        logger.info(f"Stored {len(skill_ids)} skills for doc {doc_id}")
        return skill_ids

    async def extract_and_store(
        self,
        doc_id: str,
        passages: List[Dict],
        concepts: List[Dict],
        title: str = "Unknown"
    ) -> List[str]:
        """
        Extract skills from passages and store in database.

        Args:
            doc_id: Document ID
            passages: List of passage dicts
            concepts: List of concept dicts
            title: Paper title

        Returns:
            List of created skill IDs
        """
        skills = await self.extract_from_passages(passages, concepts, title, doc_id)
        skill_ids = self.store_skills(skills, doc_id)
        return skill_ids

    def find_similar_skills(self, skill_id: str, threshold: float = 0.7) -> List[Dict]:
        """Find skills similar to the given skill using embedding similarity."""
        cur = self.conn.cursor()

        cur.execute("""
            SELECT s2.skill_id, s2.skill_name, s2.original_domain, s2.description,
                   1 - (s1.embedding <=> s2.embedding) as similarity
            FROM paper_skills s1, paper_skills s2
            WHERE s1.skill_id = %s
            AND s2.skill_id != %s
            AND s1.embedding IS NOT NULL
            AND s2.embedding IS NOT NULL
            AND 1 - (s1.embedding <=> s2.embedding) > %s
            ORDER BY similarity DESC
            LIMIT 10
        """, (skill_id, skill_id, threshold))

        return [
            {'skill_id': row[0], 'skill_name': row[1], 'domain': row[2],
             'description': row[3], 'similarity': row[4]}
            for row in cur.fetchall()
        ]

    # =========================================================================
    # DRAFT SKILL FILE GENERATION
    # =========================================================================

    def write_to_drafts(
        self,
        skills: List[ExtractedSkill],
        doc_id: str,
        title: str = "Unknown",
        code_links: List[Dict] = None
    ) -> List[Path]:
        """
        Write extracted skills to ~/.claude/skills_drafts/ as CANDIDATE.md files.

        This is the ONLY place skills should be written. Use scripts/promote_skill.py
        to promote validated skills to ~/.claude/skills/.

        Args:
            skills: List of extracted skills
            doc_id: Source document ID
            title: Paper title
            code_links: Optional list of code references {'repo_url': ..., 'file_path': ...}

        Returns:
            List of paths to created skill directories
        """
        if not skills:
            return []

        # Ensure drafts directory exists
        SKILLS_DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

        created_paths = []
        code_links = code_links or []

        for skill in skills:
            # Create skill directory
            skill_dir = SKILLS_DRAFTS_DIR / skill.skill_name
            skill_dir.mkdir(parents=True, exist_ok=True)

            # Generate CANDIDATE.md
            candidate_content = self._generate_candidate_md(skill, doc_id, title)
            candidate_path = skill_dir / "CANDIDATE.md"
            candidate_path.write_text(candidate_content)

            # Generate evidence.json
            evidence = self._generate_evidence_json(skill, doc_id, code_links)
            evidence_path = skill_dir / "evidence.json"
            evidence_path.write_text(json.dumps(evidence, indent=2))

            created_paths.append(skill_dir)
            logger.info(f"Wrote draft skill to {skill_dir}")

        return created_paths

    def _generate_candidate_md(
        self,
        skill: ExtractedSkill,
        doc_id: str,
        title: str
    ) -> str:
        """Generate CANDIDATE.md content for a skill draft."""
        today = datetime.now().strftime('%Y-%m-%d')

        # Build steps section
        steps_md = ""
        if skill.steps:
            for i, step in enumerate(skill.steps, 1):
                step_desc = step.get('description', '') if isinstance(step, dict) else str(step)
                step_code = step.get('code', '') if isinstance(step, dict) else ''
                steps_md += f"\n### Step {i}: {step_desc[:50]}\n\n"
                if step_code:
                    steps_md += f"```python\n{step_code}\n```\n"

        # Build parameters section
        params_md = ""
        if skill.parameters:
            params_md = "\n## Parameters\n\n| Parameter | Default | Range | Guidance |\n|-----------|---------|-------|----------|\n"
            for param in skill.parameters:
                if isinstance(param, dict):
                    name = param.get('name', '?')
                    default = param.get('default', '?')
                    range_val = param.get('range', '?')
                    guidance = param.get('guidance', '')
                    params_md += f"| {name} | {default} | {range_val} | {guidance} |\n"

        # Build failure modes section
        failure_md = ""
        if skill.failure_modes:
            failure_md = "\n## Failure Modes\n\n"
            for mode in skill.failure_modes:
                failure_md += f"- {mode}\n"

        # Build cross-domain section
        cross_domain_md = ""
        if skill.transferable_to or skill.abstract_principle:
            cross_domain_md = "\n## Cross-Domain Potential\n\n"
            if skill.abstract_principle:
                cross_domain_md += f"**Abstract Principle:** {skill.abstract_principle}\n\n"
            if skill.original_domain:
                cross_domain_md += f"**Original Domain:** {skill.original_domain}\n\n"
            if skill.transferable_to:
                cross_domain_md += f"**Transferable To:** {', '.join(skill.transferable_to)}\n\n"
            if skill.transfer_insights:
                cross_domain_md += f"**Transfer Insight:** {skill.transfer_insights}\n"

        content = f"""---
name: {skill.skill_name}
version: 0.1
tier: LOW
status: candidate
domains: [{skill.original_domain or 'unknown'}]
extracted_from: {doc_id}
extracted_date: {today}
confidence: {skill.confidence}
---

# {skill.skill_name.replace('-', ' ').title()}

> **STATUS: CANDIDATE** - This skill was auto-extracted and needs validation.
> Run `python scripts/promote_skill.py {skill.skill_name} --bootstrap` after adding an oracle test.

## Source

- **Paper:** {title}
- **Document ID:** {doc_id}

## Description

{skill.description}

## When to Use

Use this skill when:
- [Add specific trigger conditions after review]

## When NOT to Use

Do NOT use this skill when:
- [Add exclusion conditions after review]

## Prerequisites

{chr(10).join(f'- {p}' for p in skill.prerequisites) if skill.prerequisites else '- [Add prerequisites]'}

## Procedure
{steps_md if steps_md else '''
### Step 1: [First step]

```python
# Add implementation
```
'''}
{params_md}
{failure_md}
{cross_domain_md}
## Oracle (Verification Test)

```python
def test_skill():
    \"\"\"
    TODO: Implement oracle test.
    This test MUST pass before the skill can be promoted.
    \"\"\"
    # Setup: Create minimal toy data
    # test_data = ...

    # Execute: Run the core skill operation
    # result = skill_function(test_data)

    # Verify: Check expected properties
    # assert result is not None
    # assert ...

    raise NotImplementedError("Oracle test not yet implemented")

if __name__ == "__main__":
    test_skill()
```

## Evidence

See `evidence.json` for source passages and code links.

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | {today} | Auto-extracted from paper |
"""
        return content

    def _generate_evidence_json(
        self,
        skill: ExtractedSkill,
        doc_id: str,
        code_links: List[Dict]
    ) -> Dict:
        """Generate evidence.json for Gate 1 validation."""
        return {
            "skill_name": skill.skill_name,
            "extracted_date": datetime.now().isoformat(),
            "source_doc_id": doc_id,
            "passages": [
                {
                    "passage_id": skill.source_passage_id,
                    "relevance": 1.0,
                    "role": "primary_source"
                }
            ],
            "code_links": [
                {
                    "repo_url": link.get('repo_url', ''),
                    "file_path": link.get('file_path', ''),
                    "relevance": link.get('relevance', 0.8)
                }
                for link in code_links
                if link.get('repo_url')
            ],
            "concepts": [],  # Will be populated if concepts are linked
            "validation_status": {
                "gate_1_evidence": len(code_links) > 0 or False,  # Will need 2+ passages
                "gate_2_oracle": False,
                "gate_3_dedup": None,
                "gate_4_usage": False
            }
        }

    async def extract_and_write_drafts(
        self,
        doc_id: str,
        passages: List[Dict],
        concepts: List[Dict],
        title: str = "Unknown",
        code_links: List[Dict] = None
    ) -> List[Path]:
        """
        Extract skills and write to drafts directory (not database).

        This is the recommended method for new skill extraction.
        Skills are written to ~/.claude/skills_drafts/ only.

        Args:
            doc_id: Document ID
            passages: List of passage dicts
            concepts: List of concept dicts
            title: Paper title
            code_links: Optional code references

        Returns:
            List of paths to created skill draft directories
        """
        skills = await self.extract_from_passages(passages, concepts, title, doc_id)
        return self.write_to_drafts(skills, doc_id, title, code_links)


# ============================================================
# CLI for testing
# ============================================================

if __name__ == '__main__':
    import asyncio

    test_passage = {
        'passage_id': 'test-1',
        'passage_text': """
        We propose a neighborhood enrichment analysis to quantify cell type colocalization.
        The method works as follows:

        Step 1: Build a spatial connectivity graph where cells are nodes and edges connect
        neighboring cells within a specified radius (default 50 microns).

        Step 2: For each pair of cell types (A, B), count the number of A-B edges.

        Step 3: Generate a null distribution by permuting cell type labels 1000 times
        and counting A-B edges for each permutation.

        Step 4: Compute a z-score comparing observed A-B edges to the null distribution.
        Positive z-scores indicate colocalization; negative indicate avoidance.

        This approach is sensitive to the radius parameter - larger radii detect broader
        spatial patterns but may miss fine-grained colocalization.
        """
    }

    test_concepts = [
        {'passage_id': 'test-1', 'concept_name': 'neighborhood enrichment', 'concept_type': 'method'},
        {'passage_id': 'test-1', 'concept_name': 'spatial connectivity graph', 'concept_type': 'method'},
        {'passage_id': 'test-1', 'concept_name': 'permutation test', 'concept_type': 'method'},
    ]

    async def test():
        extractor = SkillExtractor()
        skills = await extractor.extract_from_passage(
            test_passage,
            test_concepts,
            title="Spatial Analysis Methods"
        )

        for skill in skills:
            print(f"\nSkill: {skill.skill_name}")
            print(f"Type: {skill.skill_type}")
            print(f"Description: {skill.description}")
            print(f"Domain: {skill.original_domain}")
            print(f"Transfers to: {skill.transferable_to}")
            print(f"Abstract principle: {skill.abstract_principle}")

    # Note: Requires GEMINI_API_KEY to be set
    # asyncio.run(test())
    print("SkillExtractor loaded. Set GEMINI_API_KEY and run test() to test extraction.")
