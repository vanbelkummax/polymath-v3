-- Migration: 002_polymath_assets.sql
-- Purpose: Add tables for GitHub repos, HuggingFace models, skills, and gap detection
-- Date: 2026-01-17

BEGIN;

-- ============================================================
-- ASSET DETECTION: GitHub Repos
-- ============================================================

CREATE TABLE IF NOT EXISTS repo_queue (
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

CREATE TABLE IF NOT EXISTS paper_repos (
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

CREATE TABLE IF NOT EXISTS hf_models (
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

CREATE TABLE IF NOT EXISTS paper_hf_models (
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

CREATE TABLE IF NOT EXISTS paper_skills (
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

    -- Embedding for similarity search
    embedding vector(1024),

    -- Lifecycle
    confidence FLOAT,
    status TEXT DEFAULT 'draft',  -- draft, validated, promoted, deprecated
    promoted_to_skill_file TEXT,  -- Path if promoted to ~/.claude/skills/
    validation_notes TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS paper_skill_contributions (
    contribution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    skill_id UUID NOT NULL REFERENCES paper_skills(skill_id) ON DELETE CASCADE,

    contribution_type TEXT,  -- 'origin', 'enhancement', 'validation', 'cross_domain', 'contradiction'
    contribution_detail TEXT,

    -- Timestamps
    extracted_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(doc_id, skill_id)
);

CREATE TABLE IF NOT EXISTS skill_bridges (
    bridge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source
    source_skill_id UUID REFERENCES paper_skills(skill_id),
    source_domain TEXT NOT NULL,

    -- Target
    target_skill_id UUID REFERENCES paper_skills(skill_id),
    target_domain TEXT NOT NULL,

    -- The insight
    bridge_type TEXT,  -- 'same_math', 'analogous_structure', 'shared_abstraction', 'semantic_similarity'
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

CREATE TABLE IF NOT EXISTS citation_links (
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
    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(citing_doc_id, cited_doi)
);

-- ============================================================
-- GAP ANALYSIS
-- ============================================================

CREATE TABLE IF NOT EXISTS knowledge_gaps (
    gap_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- What's missing
    gap_type TEXT,  -- 'sparse_domain', 'missing_bridge', 'no_implementation', 'outdated', 'orphan_citation'
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

CREATE INDEX IF NOT EXISTS idx_repo_queue_status ON repo_queue(status);
CREATE INDEX IF NOT EXISTS idx_repo_queue_priority ON repo_queue(priority DESC);
CREATE INDEX IF NOT EXISTS idx_paper_repos_doc ON paper_repos(doc_id);
CREATE INDEX IF NOT EXISTS idx_paper_repos_url ON paper_repos(repo_url);

CREATE INDEX IF NOT EXISTS idx_hf_models_org ON hf_models(organization);
CREATE INDEX IF NOT EXISTS idx_hf_models_pipeline ON hf_models(pipeline_tag);
CREATE INDEX IF NOT EXISTS idx_paper_hf_doc ON paper_hf_models(doc_id);

CREATE INDEX IF NOT EXISTS idx_paper_skills_domain ON paper_skills(original_domain);
CREATE INDEX IF NOT EXISTS idx_paper_skills_status ON paper_skills(status);
CREATE INDEX IF NOT EXISTS idx_paper_skills_type ON paper_skills(skill_type);
CREATE INDEX IF NOT EXISTS idx_paper_skills_name ON paper_skills(skill_name);

CREATE INDEX IF NOT EXISTS idx_skill_bridges_domains ON skill_bridges(source_domain, target_domain);
CREATE INDEX IF NOT EXISTS idx_skill_bridges_source ON skill_bridges(source_skill_id);
CREATE INDEX IF NOT EXISTS idx_skill_bridges_target ON skill_bridges(target_skill_id);

CREATE INDEX IF NOT EXISTS idx_citation_links_citing ON citation_links(citing_doc_id);
CREATE INDEX IF NOT EXISTS idx_citation_links_cited ON citation_links(cited_doc_id);
CREATE INDEX IF NOT EXISTS idx_citation_links_doi ON citation_links(cited_doi);

CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_status ON knowledge_gaps(status, priority DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_type ON knowledge_gaps(gap_type);

-- Full-text search on skills
CREATE INDEX IF NOT EXISTS idx_paper_skills_fts ON paper_skills
    USING gin(to_tsvector('english', coalesce(skill_name, '') || ' ' || coalesce(description, '')));

-- Vector similarity on skill embeddings
CREATE INDEX IF NOT EXISTS idx_paper_skills_embedding ON paper_skills
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

COMMIT;

-- ============================================================
-- VERIFICATION
-- ============================================================

-- Check tables created
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('repo_queue', 'paper_repos', 'hf_models', 'paper_hf_models',
                   'paper_skills', 'skill_bridges', 'citation_links', 'knowledge_gaps');
