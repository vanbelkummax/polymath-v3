# Polymath System Specification v1.0

> **Purpose**: A polymathic knowledge system that extracts actionable skills from papers, links code implementations, catalogs models, and discovers cross-domain insights between biology/spatial analysis and AI/ML.

> **For Claude**: Load this document at session start. It defines the complete architecture you wield.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Database Schema](#2-database-schema)
3. [Ingestion Pipeline](#3-ingestion-pipeline)
4. [Cross-Domain Skill Discovery](#4-cross-domain-skill-discovery)
5. [Knowledge Graph Analysis](#5-knowledge-graph-analysis)
6. [Gap Detection & Recommendations](#6-gap-detection--recommendations)
7. [Query Patterns](#7-query-patterns)
8. [Maintenance Cycles](#8-maintenance-cycles)
9. [Implementation Checklist](#9-implementation-checklist)

---

## 1. System Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           POLYMATH SYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   INGEST     │───▶│   EXTRACT    │───▶│   CONNECT    │                   │
│  │   Papers     │    │   Assets     │    │   Knowledge  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                      POSTGRES                                │            │
│  │  • passages (text + embeddings)                             │            │
│  │  • passage_concepts (extracted concepts)                    │            │
│  │  • paper_skills (actionable knowledge)                      │            │
│  │  • code_chunks (AST-parsed implementations)                 │            │
│  │  • hf_models (model catalog)                                │            │
│  │  • repo_queue (pending downloads)                           │            │
│  └─────────────────────────────────────────────────────────────┘            │
│         │                                                                    │
│         │ sync                                                               │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                       NEO4J                                  │            │
│  │  Nodes: Paper, Passage, Concept, Skill, CodeChunk, Model    │            │
│  │  Edges: MENTIONS, IMPLEMENTS, CITES, TRANSFERS_TO, BRIDGES  │            │
│  └─────────────────────────────────────────────────────────────┘            │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    ANALYSIS LAYER                            │            │
│  │  • Cross-domain skill discovery                              │            │
│  │  • Citation network analysis                                 │            │
│  │  • Gap detection & recommendations                           │            │
│  │  • Skill promotion to ~/.claude/skills/                      │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. PARSE: Extract text, chunk into passages                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. EMBED: Generate BGE-M3 embeddings (1024-dim)                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. EXTRACT CONCEPTS: Gemini batch API                            │
│    Types: method, problem, domain, entity, mechanism             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. DETECT ASSETS:                                                │
│    • GitHub URLs → repo_queue (async clone + ingest)            │
│    • HuggingFace models → hf_models (API metadata)              │
│    • Citations → citation_links (DOI resolution)                 │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. EXTRACT SKILLS:                                               │
│    • Procedural methods (steps to reproduce)                     │
│    • Algorithms (pseudocode, equations)                          │
│    • Parameter guidance (what values work)                       │
│    • Failure modes (when it breaks)                              │
│    • Cross-domain potential (where else it applies)              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. SYNC TO NEO4J:                                                │
│    • Create/update nodes                                         │
│    • Create relationships                                        │
│    • Compute graph metrics                                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. ANALYZE & BRIDGE:                                             │
│    • Find cross-domain skill transfers                           │
│    • Identify citation clusters                                  │
│    • Detect knowledge gaps                                       │
│    • Recommend papers to acquire                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Database Schema

### 2.1 Postgres Schema (New Tables)

```sql
-- ============================================================
-- ASSET DETECTION: GitHub Repos
-- ============================================================

CREATE TABLE repo_queue (
    queue_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_url TEXT UNIQUE NOT NULL,
    repo_owner TEXT,
    repo_name TEXT,

    -- Priority & Status
    priority INTEGER DEFAULT 5,  -- Higher = more papers cite it
    status TEXT DEFAULT 'pending',  -- pending, downloading, ingesting, complete, failed

    -- Tracking
    source_doc_count INTEGER DEFAULT 1,
    first_seen_doc_id UUID REFERENCES documents(doc_id),
    local_path TEXT,
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE paper_repos (
    paper_repo_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    repo_url TEXT NOT NULL,
    repo_owner TEXT,
    repo_name TEXT,

    -- Context
    passage_id UUID REFERENCES passages(passage_id),
    context TEXT,  -- Surrounding text where URL was found

    -- Timestamps
    discovered_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(doc_id, repo_url)
);

-- ============================================================
-- ASSET DETECTION: HuggingFace Models
-- ============================================================

CREATE TABLE hf_models (
    model_id TEXT PRIMARY KEY,  -- e.g., "facebook/dinov2-base"
    model_name TEXT,
    organization TEXT,

    -- Metadata from HF API
    pipeline_tag TEXT,  -- "image-classification", "text-generation"
    library_name TEXT,  -- "transformers", "timm", "diffusers"
    architectures TEXT[],
    model_card_summary TEXT,
    tags TEXT[],
    downloads_30d INTEGER,
    likes INTEGER,

    -- Provenance
    first_seen_doc_id UUID REFERENCES documents(doc_id),
    citation_count INTEGER DEFAULT 1,

    -- Timestamps
    last_metadata_fetch TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE paper_hf_models (
    paper_hf_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL REFERENCES hf_models(model_id),

    -- Context
    passage_id UUID REFERENCES passages(passage_id),
    context TEXT,
    usage_type TEXT,  -- 'pretrained_backbone', 'fine_tuned', 'compared_against'

    -- Timestamps
    discovered_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(doc_id, model_id)
);

-- ============================================================
-- SKILL EXTRACTION
-- ============================================================

CREATE TABLE paper_skills (
    skill_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill_name TEXT NOT NULL,
    skill_type TEXT,  -- 'method', 'algorithm', 'heuristic', 'workflow', 'parameter', 'failure_mode'

    -- Content
    description TEXT,
    prerequisites TEXT[],
    inputs JSONB,  -- [{name, type, description}]
    outputs JSONB,  -- [{name, type, description}]
    steps JSONB,  -- [{order, description, code_snippet}]
    parameters JSONB,  -- [{name, default, range, guidance}]
    failure_modes TEXT[],

    -- Provenance
    source_doc_ids UUID[],
    source_passages JSONB,  -- [{passage_id, relevance_score}]
    source_code_chunks UUID[],

    -- Cross-domain
    original_domain TEXT,
    transferable_to TEXT[],
    transfer_insights TEXT,
    abstract_principle TEXT,  -- The generalizable core idea

    -- Lifecycle
    confidence FLOAT,
    status TEXT DEFAULT 'draft',  -- draft, validated, promoted, deprecated
    promoted_to_skill_file TEXT,  -- Path if promoted to ~/.claude/skills/
    validation_notes TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE paper_skill_contributions (
    contribution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    skill_id UUID NOT NULL REFERENCES paper_skills(skill_id) ON DELETE CASCADE,

    contribution_type TEXT,  -- 'origin', 'enhancement', 'validation', 'cross_domain', 'contradiction'
    contribution_detail TEXT,

    -- Timestamps
    extracted_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(doc_id, skill_id)
);

CREATE TABLE skill_bridges (
    bridge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source
    source_skill_id UUID REFERENCES paper_skills(skill_id),
    source_domain TEXT NOT NULL,

    -- Target
    target_skill_id UUID REFERENCES paper_skills(skill_id),
    target_domain TEXT NOT NULL,

    -- The insight
    bridge_type TEXT,  -- 'same_math', 'analogous_structure', 'shared_abstraction'
    bridge_insight TEXT NOT NULL,
    abstract_principle TEXT,

    -- Evidence
    supporting_doc_ids UUID[],
    confidence FLOAT,
    validated BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- CITATION NETWORK
-- ============================================================

CREATE TABLE citation_links (
    citation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    citing_doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    cited_doc_id UUID REFERENCES documents(doc_id) ON DELETE SET NULL,

    -- If cited paper not in our corpus
    cited_doi TEXT,
    cited_title TEXT,
    cited_authors TEXT,
    cited_year INTEGER,

    -- Context
    citation_context TEXT,  -- Surrounding text
    citation_intent TEXT,  -- 'background', 'method', 'comparison', 'extension'

    -- Status
    in_corpus BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- GAP ANALYSIS
-- ============================================================

CREATE TABLE knowledge_gaps (
    gap_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- What's missing
    gap_type TEXT,  -- 'sparse_domain', 'missing_bridge', 'no_implementation', 'outdated'
    gap_description TEXT,

    -- Where it was detected
    related_concepts TEXT[],
    related_domains TEXT[],
    related_skill_ids UUID[],

    -- Recommendations
    recommended_searches TEXT[],  -- Search queries to find papers
    recommended_dois TEXT[],  -- Specific papers to acquire
    priority INTEGER DEFAULT 5,

    -- Status
    status TEXT DEFAULT 'open',  -- open, addressed, wont_fix
    addressed_by_doc_id UUID REFERENCES documents(doc_id),

    -- Timestamps
    detected_at TIMESTAMPTZ DEFAULT now(),
    addressed_at TIMESTAMPTZ
);

-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX idx_repo_queue_status ON repo_queue(status);
CREATE INDEX idx_repo_queue_priority ON repo_queue(priority DESC);
CREATE INDEX idx_paper_repos_doc ON paper_repos(doc_id);
CREATE INDEX idx_paper_hf_doc ON paper_hf_models(doc_id);
CREATE INDEX idx_paper_skills_domain ON paper_skills(original_domain);
CREATE INDEX idx_paper_skills_status ON paper_skills(status);
CREATE INDEX idx_skill_bridges_domains ON skill_bridges(source_domain, target_domain);
CREATE INDEX idx_citation_links_citing ON citation_links(citing_doc_id);
CREATE INDEX idx_citation_links_cited ON citation_links(cited_doc_id);
CREATE INDEX idx_knowledge_gaps_status ON knowledge_gaps(status, priority DESC);
```

### 2.2 Neo4j Schema

```cypher
// ============================================================
// NODE TYPES
// ============================================================

// Papers (synced from Postgres documents)
CREATE CONSTRAINT paper_doc_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.doc_id IS UNIQUE;

// Passages (synced from Postgres passages)
CREATE CONSTRAINT passage_id IF NOT EXISTS FOR (p:Passage) REQUIRE p.passage_id IS UNIQUE;

// Concepts (typed: Method, Problem, Domain, Entity, Mechanism)
CREATE CONSTRAINT concept_unique IF NOT EXISTS FOR (c:Concept) REQUIRE (c.name, c.type) IS UNIQUE;

// Skills (extracted actionable knowledge)
CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (s:Skill) REQUIRE s.skill_id IS UNIQUE;

// Code chunks (from GitHub repos)
CREATE CONSTRAINT code_chunk_id IF NOT EXISTS FOR (c:CodeChunk) REQUIRE c.chunk_id IS UNIQUE;

// HuggingFace models
CREATE CONSTRAINT hf_model_id IF NOT EXISTS FOR (m:HFModel) REQUIRE m.model_id IS UNIQUE;

// Domains (high-level categories)
CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE;

// ============================================================
// RELATIONSHIP TYPES
// ============================================================

// Paper structure
// (Passage)-[:FROM_PAPER]->(Paper)
// (Paper)-[:CITES]->(Paper)  -- citation network

// Concept extraction
// (Passage)-[:MENTIONS {confidence}]->(Concept)
// (Concept)-[:BELONGS_TO]->(Domain)

// Skill relationships
// (Paper)-[:TEACHES]->(Skill)
// (Skill)-[:APPLIES_TO]->(Domain)
// (Skill)-[:IMPLEMENTS]->(Concept)
// (Skill)-[:BRIDGES {insight}]->(Skill)  -- cross-domain transfer

// Code relationships
// (CodeChunk)-[:FROM_REPO {repo_name}]->(Paper)  -- paper that referenced the repo
// (CodeChunk)-[:IMPLEMENTS]->(Skill)
// (CodeChunk)-[:USES]->(Concept)

// Model relationships
// (Paper)-[:USES_MODEL]->(HFModel)
// (HFModel)-[:ARCHITECTURE]->(Concept)

// ============================================================
// DOMAIN HIERARCHY
// ============================================================

// Create core domains
MERGE (d:Domain {name: 'biology'})
MERGE (d:Domain {name: 'spatial_transcriptomics'})
MERGE (d:Domain {name: 'single_cell'})
MERGE (d:Domain {name: 'pathology'})
MERGE (d:Domain {name: 'genomics'})

MERGE (d:Domain {name: 'ai_ml'})
MERGE (d:Domain {name: 'deep_learning'})
MERGE (d:Domain {name: 'computer_vision'})
MERGE (d:Domain {name: 'nlp'})
MERGE (d:Domain {name: 'graph_neural_networks'})

MERGE (d:Domain {name: 'mathematics'})
MERGE (d:Domain {name: 'statistics'})
MERGE (d:Domain {name: 'spatial_statistics'})
MERGE (d:Domain {name: 'optimization'})
MERGE (d:Domain {name: 'information_theory'})

// Domain hierarchy
MATCH (child:Domain {name: 'spatial_transcriptomics'}), (parent:Domain {name: 'biology'})
MERGE (child)-[:SUBFIELD_OF]->(parent);

MATCH (child:Domain {name: 'deep_learning'}), (parent:Domain {name: 'ai_ml'})
MERGE (child)-[:SUBFIELD_OF]->(parent);
```

---

## 3. Ingestion Pipeline

### 3.1 Pipeline Stages

```python
# lib/ingest/pipeline.py

class PolymathIngestionPipeline:
    """
    Complete ingestion pipeline for the Polymath system.

    Stages:
    1. Parse PDF → passages
    2. Generate embeddings
    3. Extract concepts (Gemini batch)
    4. Detect assets (GitHub, HuggingFace, citations)
    5. Extract skills
    6. Sync to Neo4j
    7. Analyze cross-domain bridges
    """

    def __init__(self):
        self.pdf_parser = EnhancedPDFParser()
        self.embedder = Embedder()
        self.concept_extractor = ConceptExtractor()
        self.asset_detector = AssetDetector()
        self.skill_extractor = SkillExtractor()
        self.neo4j_syncer = Neo4jSyncer()
        self.bridge_analyzer = BridgeAnalyzer()

    async def ingest_paper(self, pdf_path: Path, metadata: dict = None) -> IngestResult:
        """Ingest a single paper through all stages."""

        # Stage 1: Parse
        doc_id, passages = await self.pdf_parser.parse(pdf_path, metadata)

        # Stage 2: Embed
        embeddings = self.embedder.encode([p.text for p in passages])
        await self.store_embeddings(passages, embeddings)

        # Stage 3: Extract concepts
        concepts = await self.concept_extractor.extract_batch(passages)
        await self.store_concepts(concepts)

        # Stage 4: Detect assets
        assets = await self.asset_detector.detect_all(passages)
        await self.queue_assets(assets)

        # Stage 5: Extract skills
        skills = await self.skill_extractor.extract(passages, concepts)
        await self.store_skills(skills)

        # Stage 6: Sync to Neo4j
        await self.neo4j_syncer.sync_paper(doc_id)

        # Stage 7: Analyze bridges (async, can be batched)
        await self.bridge_analyzer.analyze_new_skills(skills)

        return IngestResult(
            doc_id=doc_id,
            passages=len(passages),
            concepts=len(concepts),
            skills=len(skills),
            assets=assets
        )
```

### 3.2 Asset Detection

```python
# lib/ingest/asset_detector.py

import re
from dataclasses import dataclass
from typing import List, Optional
from huggingface_hub import HfApi

@dataclass
class DetectedAsset:
    asset_type: str  # 'github', 'huggingface', 'citation'
    identifier: str
    context: str
    passage_id: str

class AssetDetector:
    """Detect GitHub repos, HuggingFace models, and citations in paper text."""

    GITHUB_PATTERNS = [
        r'github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)',
        r'github\.io/([a-zA-Z0-9_-]+)',
    ]

    HF_PATTERNS = [
        r'huggingface\.co/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)',
        r'from_pretrained\(["\']([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)["\']',
        r'(?:model|checkpoint).*["\']([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)["\']',
    ]

    # Common model name patterns (without org prefix)
    KNOWN_MODELS = {
        'bert-base', 'bert-large', 'roberta', 'gpt2', 'gpt-neo',
        'vit-base', 'vit-large', 'dinov2', 'clip', 'resnet',
        'llama', 'mistral', 'phi', 'gemma',
        'stable-diffusion', 'controlnet',
        'UNI', 'CONCH', 'HIPT', 'CTransPath', 'Phikon',
    }

    DOI_PATTERN = r'10\.\d{4,}/[^\s\]>)]+'

    def __init__(self):
        self.hf_api = HfApi()

    async def detect_all(self, passages: List[Passage]) -> dict:
        """Detect all assets in passages."""
        assets = {
            'github': [],
            'huggingface': [],
            'citations': []
        }

        for passage in passages:
            text = passage.passage_text

            # GitHub
            for pattern in self.GITHUB_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    repo = match.group(1)
                    assets['github'].append(DetectedAsset(
                        asset_type='github',
                        identifier=f'https://github.com/{repo}',
                        context=self._get_context(text, match.start()),
                        passage_id=passage.passage_id
                    ))

            # HuggingFace
            for pattern in self.HF_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    model_id = match.group(1)
                    assets['huggingface'].append(DetectedAsset(
                        asset_type='huggingface',
                        identifier=model_id,
                        context=self._get_context(text, match.start()),
                        passage_id=passage.passage_id
                    ))

            # Known models without org prefix
            for model in self.KNOWN_MODELS:
                if re.search(rf'\b{model}\b', text, re.IGNORECASE):
                    assets['huggingface'].append(DetectedAsset(
                        asset_type='huggingface',
                        identifier=model,  # Will need resolution
                        context=self._get_context(text, text.lower().find(model.lower())),
                        passage_id=passage.passage_id
                    ))

            # DOIs (citations)
            for match in re.finditer(self.DOI_PATTERN, text):
                assets['citations'].append(DetectedAsset(
                    asset_type='citation',
                    identifier=match.group(0),
                    context=self._get_context(text, match.start()),
                    passage_id=passage.passage_id
                ))

        return assets

    def _get_context(self, text: str, pos: int, window: int = 200) -> str:
        """Get surrounding context for an asset mention."""
        start = max(0, pos - window)
        end = min(len(text), pos + window)
        return text[start:end]

    async def fetch_hf_metadata(self, model_id: str) -> dict:
        """Fetch HuggingFace model metadata without downloading weights."""
        try:
            info = self.hf_api.model_info(model_id)
            return {
                'model_id': model_id,
                'model_name': info.modelId.split('/')[-1],
                'organization': info.modelId.split('/')[0] if '/' in info.modelId else None,
                'pipeline_tag': info.pipeline_tag,
                'library_name': info.library_name,
                'architectures': info.config.get('architectures', []) if info.config else [],
                'tags': info.tags or [],
                'downloads_30d': info.downloads,
                'likes': info.likes,
                'model_card_summary': (info.card_data or {}).get('description', '')[:500],
            }
        except Exception as e:
            return {'model_id': model_id, 'error': str(e)}
```

### 3.3 Skill Extraction

```python
# lib/ingest/skill_extractor.py

SKILL_EXTRACTION_PROMPT = """
Analyze this scientific paper passage for extractable, actionable skills.

Paper: {title}
Passage: {passage_text}
Detected concepts: {concepts}

Extract ANY skills that could be reused. A skill is actionable knowledge that someone could apply to solve problems.

For each skill found, provide:

1. **skill_name**: Descriptive kebab-case name (e.g., "neighborhood-enrichment-permutation-test")

2. **skill_type**: One of:
   - "method": Step-by-step procedure
   - "algorithm": Computational technique with pseudocode
   - "heuristic": Decision rule or selection criteria
   - "parameter": Guidance on values/settings
   - "failure_mode": What breaks and how to fix
   - "workflow": Multi-step pipeline

3. **description**: What it does and why it's useful (2-3 sentences)

4. **prerequisites**: What you need to know/have (e.g., ["python", "pytorch", "linear algebra"])

5. **steps**: Ordered list of steps with optional code snippets
   [{"order": 1, "description": "...", "code": "..."}]

6. **parameters**: Key parameters with guidance
   [{"name": "k", "default": 15, "range": "5-50", "guidance": "Higher for sparser data"}]

7. **failure_modes**: When it doesn't work and how to diagnose
   ["Fails when data has < 100 cells", "Sensitive to batch effects"]

8. **cross_domain_potential**: Where else this could apply
   {
     "abstract_principle": "The generalizable mathematical/logical core",
     "original_domain": "spatial_transcriptomics",
     "transferable_to": ["ecology", "urban_planning", "social_networks"],
     "transfer_insight": "Any domain with spatial entities and categorical labels..."
   }

Return JSON array of skills found. Return empty array [] if no actionable skills.

Example output:
[
  {
    "skill_name": "spatial-neighborhood-enrichment",
    "skill_type": "method",
    "description": "Quantifies whether cell types co-localize beyond random chance using permutation testing.",
    "prerequisites": ["python", "squidpy", "anndata"],
    "steps": [
      {"order": 1, "description": "Build spatial graph", "code": "sq.gr.spatial_neighbors(adata)"},
      {"order": 2, "description": "Run enrichment test", "code": "sq.gr.nhood_enrichment(adata, cluster_key='cell_type')"}
    ],
    "parameters": [
      {"name": "n_perms", "default": 1000, "range": "100-10000", "guidance": "More = stable p-values"}
    ],
    "failure_modes": ["Misleading with very unequal cell type frequencies"],
    "cross_domain_potential": {
      "abstract_principle": "Permutation test for categorical co-occurrence in spatial data",
      "original_domain": "spatial_transcriptomics",
      "transferable_to": ["ecology", "epidemiology", "urban_planning"],
      "transfer_insight": "Works for any spatially-embedded categorical data"
    }
  }
]
"""

class SkillExtractor:
    """Extract actionable skills from paper passages using LLM."""

    def __init__(self):
        self.model = "gemini-2.5-flash"

    async def extract(self, passages: List[Passage], concepts: List[Concept]) -> List[PaperSkill]:
        """Extract skills from passages that have method/algorithm concepts."""

        skills = []

        # Focus on passages with actionable concepts
        method_passages = [
            p for p in passages
            if any(c.concept_type in ('method', 'algorithm', 'mechanism')
                   for c in concepts if c.passage_id == p.passage_id)
        ]

        for passage in method_passages:
            passage_concepts = [c for c in concepts if c.passage_id == passage.passage_id]

            prompt = SKILL_EXTRACTION_PROMPT.format(
                title=passage.doc_title or "Unknown",
                passage_text=passage.passage_text,
                concepts=[c.concept_name for c in passage_concepts]
            )

            response = await self._call_llm(prompt)
            extracted = self._parse_response(response)

            for skill_data in extracted:
                skill = PaperSkill(
                    skill_name=skill_data['skill_name'],
                    skill_type=skill_data['skill_type'],
                    description=skill_data['description'],
                    prerequisites=skill_data.get('prerequisites', []),
                    inputs=skill_data.get('inputs'),
                    outputs=skill_data.get('outputs'),
                    steps=skill_data.get('steps'),
                    parameters=skill_data.get('parameters'),
                    failure_modes=skill_data.get('failure_modes', []),
                    original_domain=skill_data.get('cross_domain_potential', {}).get('original_domain'),
                    transferable_to=skill_data.get('cross_domain_potential', {}).get('transferable_to', []),
                    transfer_insights=skill_data.get('cross_domain_potential', {}).get('transfer_insight'),
                    abstract_principle=skill_data.get('cross_domain_potential', {}).get('abstract_principle'),
                    source_doc_ids=[passage.doc_id],
                    source_passages=[{'passage_id': str(passage.passage_id), 'relevance': 1.0}],
                    confidence=0.7  # Default, can be adjusted
                )
                skills.append(skill)

        return skills
```

---

## 4. Cross-Domain Skill Discovery

### 4.1 Domain Abstraction Mappings

```python
# lib/analysis/domain_abstractions.py

# Skills that share mathematical/logical structure across domains
DOMAIN_ABSTRACTIONS = {
    "spatial_autocorrelation": {
        "domains": ["spatial_transcriptomics", "geostatistics", "time_series", "image_analysis", "ecology"],
        "description": "Detecting non-random clustering in space or time",
        "key_methods": ["morans_i", "gearys_c", "variogram", "correlogram"],
    },

    "graph_message_passing": {
        "domains": ["gnn", "belief_propagation", "cellular_automata", "social_networks", "molecular_dynamics"],
        "description": "Iterative information exchange between connected nodes",
        "key_methods": ["gcn", "gat", "mpnn", "loopy_bp"],
    },

    "attention_mechanism": {
        "domains": ["nlp", "computer_vision", "genomics", "recommendation", "spatial_transcriptomics"],
        "description": "Learned weighted aggregation of relevant context",
        "key_methods": ["self_attention", "cross_attention", "multi_head"],
    },

    "contrastive_learning": {
        "domains": ["computer_vision", "nlp", "molecular", "audio", "pathology"],
        "description": "Learning by comparing positive and negative pairs",
        "key_methods": ["simclr", "moco", "clip", "infonce"],
    },

    "optimal_transport": {
        "domains": ["single_cell", "domain_adaptation", "economics", "logistics", "image_registration"],
        "description": "Finding minimum-cost mappings between distributions",
        "key_methods": ["wasserstein", "sinkhorn", "ot_mapping"],
    },

    "variational_inference": {
        "domains": ["vae", "bayesian_nn", "topic_models", "phylogenetics", "single_cell"],
        "description": "Approximating intractable posteriors with tractable distributions",
        "key_methods": ["elbo", "reparameterization", "mean_field"],
    },

    "permutation_testing": {
        "domains": ["spatial_statistics", "genomics", "ecology", "neuroscience", "epidemiology"],
        "description": "Null distribution via random shuffling",
        "key_methods": ["permutation_test", "bootstrap", "monte_carlo"],
    },

    "dimensionality_reduction": {
        "domains": ["single_cell", "nlp", "computer_vision", "finance", "genomics"],
        "description": "Projecting high-dimensional data to interpretable space",
        "key_methods": ["pca", "umap", "tsne", "diffusion_maps"],
    },

    "multiple_instance_learning": {
        "domains": ["pathology", "drug_discovery", "remote_sensing", "video_classification"],
        "description": "Learning from bags of instances with bag-level labels",
        "key_methods": ["attention_mil", "abmil", "transmil"],
    },

    "self_supervised_pretraining": {
        "domains": ["pathology", "nlp", "computer_vision", "genomics", "chemistry"],
        "description": "Learning representations from unlabeled data",
        "key_methods": ["masked_prediction", "contrastive", "reconstruction"],
    },
}

# Specific bridges we're interested in (biology <-> AI/ML)
PRIORITY_BRIDGES = [
    ("spatial_transcriptomics", "computer_vision"),
    ("single_cell", "nlp"),
    ("pathology", "deep_learning"),
    ("genomics", "graph_neural_networks"),
    ("molecular", "attention_mechanism"),
]
```

### 4.2 Bridge Discovery

```python
# lib/analysis/bridge_analyzer.py

class BridgeAnalyzer:
    """Discover cross-domain skill transfers."""

    def __init__(self, pg_conn, neo4j_driver):
        self.pg = pg_conn
        self.neo4j = neo4j_driver

    async def analyze_new_skills(self, skills: List[PaperSkill]):
        """Find cross-domain bridges for newly extracted skills."""

        bridges = []

        for skill in skills:
            # Check if skill's domain maps to a known abstraction
            for abstraction, info in DOMAIN_ABSTRACTIONS.items():
                if skill.original_domain in info['domains']:
                    # This skill might transfer to other domains
                    for target_domain in info['domains']:
                        if target_domain != skill.original_domain:
                            # Check if this is a priority bridge
                            is_priority = (skill.original_domain, target_domain) in PRIORITY_BRIDGES or \
                                         (target_domain, skill.original_domain) in PRIORITY_BRIDGES

                            bridge = SkillBridge(
                                source_skill_id=skill.skill_id,
                                source_domain=skill.original_domain,
                                target_domain=target_domain,
                                bridge_type='shared_abstraction',
                                bridge_insight=f"{skill.skill_name} uses {abstraction}, which also applies to {target_domain}",
                                abstract_principle=skill.abstract_principle or info['description'],
                                confidence=0.8 if is_priority else 0.5,
                            )
                            bridges.append(bridge)

            # Also check for semantic similarity with existing skills in other domains
            similar_skills = await self._find_similar_skills(skill)
            for similar in similar_skills:
                if similar.original_domain != skill.original_domain:
                    bridges.append(SkillBridge(
                        source_skill_id=skill.skill_id,
                        source_domain=skill.original_domain,
                        target_skill_id=similar.skill_id,
                        target_domain=similar.original_domain,
                        bridge_type='semantic_similarity',
                        bridge_insight=f"{skill.skill_name} is semantically similar to {similar.skill_name}",
                        confidence=similar.similarity_score,
                    ))

        return bridges

    async def _find_similar_skills(self, skill: PaperSkill, threshold: float = 0.7) -> List:
        """Find semantically similar skills using embeddings."""
        # Embed skill description
        skill_embedding = self.embedder.encode(skill.description)

        # Query similar skills
        cur = self.pg.cursor()
        cur.execute("""
            SELECT skill_id, skill_name, original_domain, description,
                   1 - (embedding <=> %s) as similarity
            FROM paper_skills
            WHERE skill_id != %s
            AND 1 - (embedding <=> %s) > %s
            ORDER BY similarity DESC
            LIMIT 10
        """, (skill_embedding.tolist(), skill.skill_id, skill_embedding.tolist(), threshold))

        return cur.fetchall()

    async def periodic_bridge_analysis(self):
        """
        Run periodic analysis on entire knowledge graph to find bridges.
        Called by maintenance cycle, not per-paper ingestion.
        """

        # Find skills with same abstract_principle but different domains
        cur = self.pg.cursor()
        cur.execute("""
            SELECT s1.skill_id, s1.skill_name, s1.original_domain, s1.abstract_principle,
                   s2.skill_id, s2.skill_name, s2.original_domain
            FROM paper_skills s1
            JOIN paper_skills s2 ON s1.abstract_principle = s2.abstract_principle
            WHERE s1.original_domain != s2.original_domain
            AND s1.skill_id < s2.skill_id  -- avoid duplicates
            AND NOT EXISTS (
                SELECT 1 FROM skill_bridges
                WHERE source_skill_id = s1.skill_id AND target_skill_id = s2.skill_id
            )
        """)

        new_bridges = []
        for row in cur.fetchall():
            new_bridges.append(SkillBridge(
                source_skill_id=row[0],
                source_domain=row[2],
                target_skill_id=row[4],
                target_domain=row[6],
                bridge_type='same_principle',
                bridge_insight=f"Both use: {row[3]}",
                abstract_principle=row[3],
                confidence=0.9,
            ))

        return new_bridges
```

---

## 5. Knowledge Graph Analysis

### 5.1 Neo4j Sync

```python
# lib/db/neo4j_sync.py

class Neo4jSyncer:
    """Sync Postgres data to Neo4j graph."""

    def __init__(self, driver):
        self.driver = driver

    async def sync_paper(self, doc_id: str):
        """Sync a single paper and all its relationships to Neo4j."""

        # 1. Sync Paper node
        await self._sync_paper_node(doc_id)

        # 2. Sync Passages
        await self._sync_passages(doc_id)

        # 3. Sync Concepts and MENTIONS relationships
        await self._sync_concepts(doc_id)

        # 4. Sync Skills and TEACHES relationships
        await self._sync_skills(doc_id)

        # 5. Sync Code chunks if any
        await self._sync_code_chunks(doc_id)

        # 6. Sync HF models if any
        await self._sync_hf_models(doc_id)

        # 7. Sync citations
        await self._sync_citations(doc_id)

    async def sync_skill_bridges(self, bridges: List[SkillBridge]):
        """Sync skill bridges to Neo4j."""

        self.driver.execute_query("""
            UNWIND $bridges as b
            MATCH (s1:Skill {skill_id: b.source_skill_id})
            MATCH (s2:Skill {skill_id: b.target_skill_id})
            MERGE (s1)-[r:BRIDGES]->(s2)
            SET r.bridge_type = b.bridge_type,
                r.insight = b.bridge_insight,
                r.confidence = b.confidence,
                r.synced_at = datetime()
        """, bridges=[b.__dict__ for b in bridges])

    async def sync_domain_assignments(self):
        """Assign concepts and skills to domain nodes."""

        # Skills to domains
        self.driver.execute_query("""
            MATCH (s:Skill)
            WHERE s.original_domain IS NOT NULL
            MATCH (d:Domain {name: s.original_domain})
            MERGE (s)-[:APPLIES_TO]->(d)
        """)

        # Concepts to domains (based on co-occurrence with domain concepts)
        self.driver.execute_query("""
            MATCH (c:Concept)<-[:MENTIONS]-(p:Passage)-[:MENTIONS]->(dc:Concept)
            WHERE dc.type = 'domain'
            WITH c, dc.name as domain, count(*) as co_occurrences
            WHERE co_occurrences > 3
            MATCH (d:Domain {name: domain})
            MERGE (c)-[:BELONGS_TO {strength: co_occurrences}]->(d)
        """)
```

### 5.2 Graph Queries for Analysis

```cypher
// ============================================================
// CITATION NETWORK ANALYSIS
// ============================================================

// Find citation clusters (papers that cite each other)
CALL gds.louvain.stream('citation-graph', {
    nodeProjection: 'Paper',
    relationshipProjection: 'CITES'
})
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).title as paper, communityId
ORDER BY communityId;

// Find bridge papers (connect different clusters)
MATCH (p:Paper)-[:CITES]->(cited:Paper)
WITH p, collect(DISTINCT cited.communityId) as communities
WHERE size(communities) > 1
RETURN p.title, communities
ORDER BY size(communities) DESC;

// ============================================================
// CROSS-DOMAIN SKILL DISCOVERY
// ============================================================

// Find skills that bridge biology and AI/ML
MATCH (s1:Skill)-[:BRIDGES]->(s2:Skill)
MATCH (s1)-[:APPLIES_TO]->(d1:Domain)
MATCH (s2)-[:APPLIES_TO]->(d2:Domain)
WHERE (d1.name IN ['spatial_transcriptomics', 'single_cell', 'pathology', 'genomics']
       AND d2.name IN ['deep_learning', 'computer_vision', 'nlp', 'gnn'])
   OR (d2.name IN ['spatial_transcriptomics', 'single_cell', 'pathology', 'genomics']
       AND d1.name IN ['deep_learning', 'computer_vision', 'nlp', 'gnn'])
RETURN s1.skill_name, d1.name, s2.skill_name, d2.name, s1.bridge_insight;

// Find concepts that appear in both biology and AI papers
MATCH (c:Concept)<-[:MENTIONS]-(p1:Passage)-[:FROM_PAPER]->(paper1:Paper)
MATCH (c)<-[:MENTIONS]-(p2:Passage)-[:FROM_PAPER]->(paper2:Paper)
MATCH (paper1)-[:IN_DOMAIN]->(d1:Domain {name: 'biology'})
MATCH (paper2)-[:IN_DOMAIN]->(d2:Domain {name: 'ai_ml'})
WHERE paper1 <> paper2
RETURN c.name, c.type, count(DISTINCT paper1) as bio_papers, count(DISTINCT paper2) as ai_papers
ORDER BY bio_papers + ai_papers DESC
LIMIT 20;

// ============================================================
// SKILL-CODE-PAPER TRACEABILITY
// ============================================================

// For a skill, find: papers that teach it, code that implements it
MATCH (s:Skill {skill_name: 'neighborhood-enrichment-analysis'})
OPTIONAL MATCH (paper:Paper)-[:TEACHES]->(s)
OPTIONAL MATCH (code:CodeChunk)-[:IMPLEMENTS]->(s)
RETURN s.skill_name, s.description,
       collect(DISTINCT paper.title) as papers,
       collect(DISTINCT {repo: code.repo_name, file: code.file_path, func: code.name}) as implementations;

// ============================================================
// FIND HIGHLY CONNECTED PAPERS
// ============================================================

// Papers with most skill contributions
MATCH (p:Paper)-[:TEACHES]->(s:Skill)
RETURN p.title, p.doi, count(s) as skills_taught
ORDER BY skills_taught DESC
LIMIT 10;

// Papers that connect most domains
MATCH (p:Paper)-[:TEACHES]->(s:Skill)-[:APPLIES_TO]->(d:Domain)
WITH p, collect(DISTINCT d.name) as domains
WHERE size(domains) > 1
RETURN p.title, domains, size(domains) as domain_count
ORDER BY domain_count DESC;
```

---

## 6. Gap Detection & Recommendations

### 6.1 Gap Detection Queries

```python
# lib/analysis/gap_detector.py

class GapDetector:
    """Detect knowledge gaps and recommend papers to acquire."""

    def __init__(self, pg_conn, neo4j_driver):
        self.pg = pg_conn
        self.neo4j = neo4j_driver

    async def detect_all_gaps(self) -> List[KnowledgeGap]:
        """Run all gap detection analyses."""
        gaps = []

        gaps.extend(await self.detect_sparse_domains())
        gaps.extend(await self.detect_missing_bridges())
        gaps.extend(await self.detect_skills_without_code())
        gaps.extend(await self.detect_orphan_citations())
        gaps.extend(await self.detect_outdated_skills())

        return gaps

    async def detect_sparse_domains(self) -> List[KnowledgeGap]:
        """Find domains with few papers/skills."""

        gaps = []

        # Query domains with low coverage
        result = self.neo4j.execute_query("""
            MATCH (d:Domain)
            OPTIONAL MATCH (d)<-[:APPLIES_TO]-(s:Skill)
            OPTIONAL MATCH (d)<-[:IN_DOMAIN]-(p:Paper)
            WITH d, count(DISTINCT s) as skill_count, count(DISTINCT p) as paper_count
            WHERE skill_count < 5 OR paper_count < 10
            RETURN d.name as domain, skill_count, paper_count
        """)

        for row in result.records:
            gaps.append(KnowledgeGap(
                gap_type='sparse_domain',
                gap_description=f"Domain '{row['domain']}' has only {row['skill_count']} skills and {row['paper_count']} papers",
                related_domains=[row['domain']],
                recommended_searches=[
                    f"{row['domain']} review",
                    f"{row['domain']} methods",
                    f"{row['domain']} tutorial",
                ],
                priority=8 if row['domain'] in ['spatial_transcriptomics', 'pathology'] else 5
            ))

        return gaps

    async def detect_missing_bridges(self) -> List[KnowledgeGap]:
        """Find domain pairs that should have bridges but don't."""

        gaps = []

        # Check priority bridges
        for source, target in PRIORITY_BRIDGES:
            result = self.neo4j.execute_query("""
                MATCH (s1:Skill)-[:APPLIES_TO]->(d1:Domain {name: $source})
                MATCH (s2:Skill)-[:APPLIES_TO]->(d2:Domain {name: $target})
                OPTIONAL MATCH (s1)-[b:BRIDGES]-(s2)
                WITH count(DISTINCT s1) as source_skills,
                     count(DISTINCT s2) as target_skills,
                     count(DISTINCT b) as bridges
                RETURN source_skills, target_skills, bridges
            """, source=source, target=target)

            row = result.records[0]
            expected_bridges = min(row['source_skills'], row['target_skills']) * 0.1  # Expect 10% bridging

            if row['bridges'] < expected_bridges:
                gaps.append(KnowledgeGap(
                    gap_type='missing_bridge',
                    gap_description=f"Only {row['bridges']} bridges between {source} ({row['source_skills']} skills) and {target} ({row['target_skills']} skills)",
                    related_domains=[source, target],
                    recommended_searches=[
                        f"{source} {target}",
                        f"applying {target} to {source}",
                        f"{source} using {target} methods",
                    ],
                    priority=9  # High priority for our focus areas
                ))

        return gaps

    async def detect_skills_without_code(self) -> List[KnowledgeGap]:
        """Find skills that have no linked code implementation."""

        cur = self.pg.cursor()
        cur.execute("""
            SELECT s.skill_id, s.skill_name, s.original_domain
            FROM paper_skills s
            WHERE s.status = 'validated'
            AND (s.source_code_chunks IS NULL OR array_length(s.source_code_chunks, 1) = 0)
            AND s.skill_type IN ('method', 'algorithm')
        """)

        gaps = []
        for row in cur.fetchall():
            gaps.append(KnowledgeGap(
                gap_type='no_implementation',
                gap_description=f"Skill '{row[1]}' has no linked code implementation",
                related_skill_ids=[row[0]],
                related_domains=[row[2]] if row[2] else [],
                recommended_searches=[
                    f"{row[1]} github",
                    f"{row[1]} implementation",
                    f"{row[1]} code",
                ],
                priority=6
            ))

        return gaps

    async def detect_orphan_citations(self) -> List[KnowledgeGap]:
        """Find frequently cited papers not in our corpus."""

        cur = self.pg.cursor()
        cur.execute("""
            SELECT cited_doi, cited_title, count(*) as citation_count
            FROM citation_links
            WHERE in_corpus = FALSE
            AND cited_doi IS NOT NULL
            GROUP BY cited_doi, cited_title
            HAVING count(*) >= 3
            ORDER BY citation_count DESC
            LIMIT 50
        """)

        gaps = []
        for row in cur.fetchall():
            gaps.append(KnowledgeGap(
                gap_type='orphan_citation',
                gap_description=f"Paper '{row[1] or row[0]}' is cited {row[2]} times but not in corpus",
                recommended_dois=[row[0]],
                priority=min(10, 5 + row[2])  # Higher priority for more citations
            ))

        return gaps

    async def detect_outdated_skills(self) -> List[KnowledgeGap]:
        """Find skills based on old papers that may need updating."""

        cur = self.pg.cursor()
        cur.execute("""
            SELECT s.skill_id, s.skill_name, max(d.year) as latest_source_year
            FROM paper_skills s
            JOIN documents d ON d.doc_id = ANY(s.source_doc_ids)
            WHERE s.status = 'validated'
            GROUP BY s.skill_id, s.skill_name
            HAVING max(d.year) < 2023
        """)

        gaps = []
        for row in cur.fetchall():
            gaps.append(KnowledgeGap(
                gap_type='outdated',
                gap_description=f"Skill '{row[1]}' is based on papers from {row[2]} or earlier",
                related_skill_ids=[row[0]],
                recommended_searches=[
                    f"{row[1]} 2024",
                    f"{row[1]} latest",
                    f"{row[1]} state of the art",
                ],
                priority=4
            ))

        return gaps
```

### 6.2 Paper Recommendations

```python
# lib/analysis/recommender.py

class PaperRecommender:
    """Recommend papers to acquire based on gaps and interests."""

    async def get_recommendations(self, focus_domains: List[str] = None) -> List[dict]:
        """Get prioritized paper recommendations."""

        if focus_domains is None:
            focus_domains = ['spatial_transcriptomics', 'pathology', 'deep_learning', 'computer_vision']

        recommendations = []

        # 1. Orphan citations (frequently cited but not in corpus)
        orphans = await self._get_orphan_citations(focus_domains)
        recommendations.extend(orphans)

        # 2. Papers from highly-cited repos we don't have papers for
        repo_papers = await self._get_repo_source_papers()
        recommendations.extend(repo_papers)

        # 3. Recent papers in sparse domains
        recent = await self._get_recent_in_sparse_domains(focus_domains)
        recommendations.extend(recent)

        # 4. Papers that would create bridges
        bridge_papers = await self._get_bridge_papers(focus_domains)
        recommendations.extend(bridge_papers)

        # Deduplicate and sort by priority
        seen = set()
        unique = []
        for rec in sorted(recommendations, key=lambda x: x['priority'], reverse=True):
            key = rec.get('doi') or rec.get('title')
            if key and key not in seen:
                seen.add(key)
                unique.append(rec)

        return unique[:50]  # Top 50 recommendations
```

---

## 7. Query Patterns

### 7.1 Polymathic Search

```python
# lib/search/polymathic_search.py

class PolymathicSearch:
    """
    Search across papers, skills, code, and models with cross-domain awareness.
    """

    async def search(self, query: str, include_bridges: bool = True) -> SearchResult:
        """
        Multi-modal polymathic search.

        Returns:
        - Relevant passages (semantic search)
        - Matching skills (with cross-domain alternatives)
        - Code implementations
        - Related HF models
        - Cross-domain insights
        """

        # 1. Semantic passage search
        passages = await self._search_passages(query)

        # 2. Skill search (exact + semantic)
        skills = await self._search_skills(query)

        # 3. For each skill, find cross-domain alternatives
        if include_bridges:
            for skill in skills:
                bridges = await self._find_skill_bridges(skill)
                skill.cross_domain_alternatives = bridges

        # 4. Code search
        code_chunks = await self._search_code(query)

        # 5. Model search
        models = await self._search_models(query)

        # 6. Graph-based expansion (Neo4j)
        related = await self._graph_expand(passages, skills)

        return SearchResult(
            passages=passages,
            skills=skills,
            code=code_chunks,
            models=models,
            related_concepts=related['concepts'],
            related_papers=related['papers'],
            cross_domain_insights=related['bridges']
        )

    async def find_how_to(self, task: str) -> HowToResult:
        """
        Given a task description, find:
        1. Skills that accomplish it
        2. Code that implements those skills
        3. Papers that teach the methods
        4. Alternative approaches from other domains
        """

        # Extract intent
        intent = await self._extract_intent(task)

        # Find matching skills
        skills = await self._search_skills_by_intent(intent)

        results = []
        for skill in skills:
            result = HowToEntry(
                skill=skill,
                papers=await self._get_skill_papers(skill),
                code=await self._get_skill_code(skill),
                alternatives=await self._find_skill_bridges(skill),
            )
            results.append(result)

        return HowToResult(
            task=task,
            intent=intent,
            approaches=results
        )
```

### 7.2 Example Queries

```python
# Example: "How do I detect cell type colocalization?"

result = await polymath.find_how_to("detect cell type colocalization")

# Returns:
{
    "task": "detect cell type colocalization",
    "intent": {"goal": "spatial_colocalization", "entity": "cell_type"},
    "approaches": [
        {
            "skill": {
                "name": "neighborhood-enrichment-analysis",
                "description": "Permutation test for cell type co-occurrence",
                "steps": [...],
                "domain": "spatial_transcriptomics"
            },
            "papers": [
                {"title": "Squidpy: a scalable framework...", "doi": "10.1038/s41592-021-01358-2"}
            ],
            "code": [
                {"repo": "scverse/squidpy", "file": "squidpy/gr/_nhood.py", "function": "nhood_enrichment"}
            ],
            "alternatives": [
                {
                    "skill": "species-cooccurrence-analysis",
                    "domain": "ecology",
                    "insight": "Same permutation approach, different entity type",
                    "code": [{"repo": "ecopy", "function": "cooccurrence"}]
                },
                {
                    "skill": "point-pattern-analysis",
                    "domain": "geostatistics",
                    "insight": "Ripley's K function measures clustering at multiple scales"
                }
            ]
        }
    ]
}
```

---

## 8. Maintenance Cycles

### 8.1 Continuous (Per-Ingestion)

```python
# Runs automatically when a paper is ingested

async def on_paper_ingested(doc_id: str):
    """Post-ingestion hooks."""

    # 1. Sync to Neo4j
    await neo4j_syncer.sync_paper(doc_id)

    # 2. Analyze new skills for bridges
    skills = await get_skills_for_doc(doc_id)
    bridges = await bridge_analyzer.analyze_new_skills(skills)
    await store_bridges(bridges)

    # 3. Check if any knowledge gaps are addressed
    await gap_detector.check_addressed_gaps(doc_id)

    # 4. Update citation network
    await citation_analyzer.update_for_paper(doc_id)
```

### 8.2 Periodic (Daily/Weekly)

```python
# scripts/maintenance/daily_analysis.py

async def daily_maintenance():
    """Run daily maintenance tasks."""

    logger.info("Starting daily maintenance...")

    # 1. Process repo download queue
    await process_repo_queue(max_repos=10)

    # 2. Fetch HF model metadata for new models
    await refresh_hf_metadata()

    # 3. Run gap detection
    gaps = await gap_detector.detect_all_gaps()
    await store_gaps(gaps)

    # 4. Generate paper recommendations
    recommendations = await recommender.get_recommendations()
    await store_recommendations(recommendations)

    # 5. Periodic bridge analysis (whole graph)
    new_bridges = await bridge_analyzer.periodic_bridge_analysis()
    await neo4j_syncer.sync_skill_bridges(new_bridges)

    # 6. Promote validated skills to skill files
    await skill_promoter.promote_validated_skills()

    # 7. Generate status report
    report = await generate_status_report()
    await save_report(report)

    logger.info(f"Daily maintenance complete: {len(gaps)} gaps, {len(new_bridges)} new bridges")


# scripts/maintenance/weekly_analysis.py

async def weekly_maintenance():
    """Run weekly deep analysis."""

    logger.info("Starting weekly maintenance...")

    # 1. Full citation network analysis
    await citation_analyzer.full_network_analysis()

    # 2. Domain coverage report
    coverage = await generate_domain_coverage_report()

    # 3. Skill quality audit
    await audit_skill_quality()

    # 4. Cross-domain bridge discovery (exhaustive)
    await bridge_analyzer.exhaustive_bridge_search()

    # 5. Update skill catalog
    await update_skill_catalog()

    logger.info("Weekly maintenance complete")
```

### 8.3 Skill Promotion

```python
# lib/skills/promoter.py

class SkillPromoter:
    """Promote validated skills to ~/.claude/skills/ files."""

    SKILL_TEMPLATE = '''---
name: {skill_name}
description: {description}
source_papers: {dois}
domains: {domains}
---

# {title}

## Overview
{description}

## Prerequisites
{prerequisites}

## Quick Reference

```python
{quick_code}
```

## Detailed Steps

{steps}

## Parameters

| Parameter | Default | Range | Guidance |
|-----------|---------|-------|----------|
{parameters}

## Failure Modes
{failure_modes}

## Cross-Domain Applications
{cross_domain}

## Source Papers
{papers}
'''

    async def promote_validated_skills(self):
        """Promote high-confidence validated skills to skill files."""

        cur = self.pg.cursor()
        cur.execute("""
            SELECT * FROM paper_skills
            WHERE status = 'validated'
            AND confidence >= 0.8
            AND promoted_to_skill_file IS NULL
        """)

        for skill in cur.fetchall():
            skill_path = await self._create_skill_file(skill)

            # Update database
            cur.execute("""
                UPDATE paper_skills
                SET status = 'promoted', promoted_to_skill_file = %s
                WHERE skill_id = %s
            """, (str(skill_path), skill['skill_id']))

            # Update skill index
            await self._update_skill_index(skill, skill_path)

        self.pg.commit()
```

---

## 9. Implementation Checklist

### Phase 1: Schema & Core Infrastructure
- [ ] Create Postgres migration for new tables
- [ ] Create Neo4j constraints and indexes
- [ ] Implement `AssetDetector` (GitHub + HuggingFace)
- [ ] Implement `SkillExtractor` (LLM-based)
- [ ] Implement `Neo4jSyncer` (full sync)

### Phase 2: Cross-Domain Analysis
- [ ] Implement `BridgeAnalyzer`
- [ ] Define `DOMAIN_ABSTRACTIONS` mappings
- [ ] Implement skill similarity search
- [ ] Create bridge discovery queries

### Phase 3: Gap Detection & Recommendations
- [ ] Implement `GapDetector`
- [ ] Implement `PaperRecommender`
- [ ] Create gap visualization

### Phase 4: Search & Query
- [ ] Implement `PolymathicSearch`
- [ ] Create `find_how_to` interface
- [ ] Build search UI/CLI

### Phase 5: Maintenance & Lifecycle
- [ ] Implement daily maintenance script
- [ ] Implement weekly maintenance script
- [ ] Implement `SkillPromoter`
- [ ] Create monitoring/alerting

### Phase 6: Integration & Testing
- [ ] Integrate with paper ingestion pipeline
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Documentation

---

## Quick Reference Commands

```bash
# Check system status
python scripts/polymath_status.py

# Run paper ingestion with full extraction
python lib/ingest/pipeline.py /path/to/paper.pdf --full

# Process repo queue (background)
python scripts/process_repo_queue.py --max-repos 5

# Run gap detection
python scripts/detect_gaps.py

# Get paper recommendations
python scripts/recommend_papers.py --focus spatial_transcriptomics,deep_learning

# Run daily maintenance
python scripts/maintenance/daily_analysis.py

# Promote validated skills
python scripts/promote_skills.py

# Polymathic search
python polymath_cli.py search "cell type colocalization" --include-bridges
```

---

## Environment Variables

```bash
# Database
export POSTGRES_DSN="dbname=polymath user=polymath host=/var/run/postgresql"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_PASSWORD="polymathic2026"

# APIs
export GEMINI_API_KEY="..."
export HF_TOKEN="..."  # Optional, for private models

# Paths
export REPOS_DIR="/home/user/work/polymax/data/github_repos"
export SKILLS_DIR="/home/user/.claude/skills"

# Google Cloud (for batch API)
export GOOGLE_APPLICATION_CREDENTIALS="/home/user/.gcp/service-account.json"
export GCP_PROJECT_ID="fifth-branch-483806-m1"
```

---

*This document is the source of truth for the Polymath system. Update it as the system evolves.*
