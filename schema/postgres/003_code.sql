-- Polymath v3 Code Schema
-- Code repository and chunk storage

-- =============================================================================
-- Code Repositories
-- =============================================================================

CREATE TABLE IF NOT EXISTS code_repos (
    repo_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_name       TEXT NOT NULL,
    org_name        TEXT,
    full_name       TEXT UNIQUE NOT NULL,  -- org/repo format

    -- Repository metadata
    description     TEXT,
    language        TEXT,  -- Primary language
    languages       JSONB,  -- All languages with byte counts
    topics          TEXT[],

    -- Stats
    stars           INTEGER,
    forks           INTEGER,
    open_issues     INTEGER,

    -- URLs
    html_url        TEXT,
    clone_url       TEXT,

    -- Paper linkage
    linked_doc_ids  UUID[],  -- Documents that reference this repo

    -- Processing state
    last_commit_sha TEXT,
    last_indexed_at TIMESTAMPTZ,
    ingest_status   TEXT DEFAULT 'pending',  -- pending, indexing, complete, failed
    error_message   TEXT,

    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_repo_name ON code_repos (repo_name);
CREATE INDEX IF NOT EXISTS idx_repo_org ON code_repos (org_name);
CREATE INDEX IF NOT EXISTS idx_repo_language ON code_repos (language);
CREATE INDEX IF NOT EXISTS idx_repo_status ON code_repos (ingest_status);

-- =============================================================================
-- Code Files
-- =============================================================================

CREATE TABLE IF NOT EXISTS code_files (
    file_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id         UUID NOT NULL REFERENCES code_repos(repo_id) ON DELETE CASCADE,

    -- File info
    file_path       TEXT NOT NULL,
    file_name       TEXT NOT NULL,
    language        TEXT,
    extension       TEXT,

    -- Content metadata
    lines_of_code   INTEGER,
    file_size       INTEGER,
    file_hash       TEXT,  -- For change detection

    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT now(),

    UNIQUE (repo_id, file_path)
);

CREATE INDEX IF NOT EXISTS idx_file_repo ON code_files (repo_id);
CREATE INDEX IF NOT EXISTS idx_file_language ON code_files (language);
CREATE INDEX IF NOT EXISTS idx_file_path ON code_files (file_path);

-- =============================================================================
-- Code Chunks
-- =============================================================================

CREATE TABLE IF NOT EXISTS code_chunks (
    chunk_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id         UUID NOT NULL REFERENCES code_files(file_id) ON DELETE CASCADE,

    -- Content
    chunk_text      TEXT NOT NULL,
    chunk_type      TEXT,  -- function, class, method, module, docstring, comment
    name            TEXT,  -- Function/class name if applicable

    -- Position
    line_start      INTEGER,
    line_end        INTEGER,

    -- Context
    parent_name     TEXT,  -- Parent class for methods
    signature       TEXT,  -- Function signature
    docstring       TEXT,  -- Extracted docstring

    -- pgvector embedding (1024-dim BGE-M3)
    embedding       vector(1024),

    -- Full-text search
    search_vector   tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,

    -- Processing metadata
    parser_version  TEXT,
    embedding_model TEXT,

    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunk_file ON code_chunks (file_id);
CREATE INDEX IF NOT EXISTS idx_chunk_type ON code_chunks (chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunk_name ON code_chunks (name);

-- HNSW index for code embeddings
CREATE INDEX IF NOT EXISTS idx_chunk_embedding ON code_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search for code
CREATE INDEX IF NOT EXISTS idx_chunk_fts ON code_chunks USING GIN (search_vector);

-- =============================================================================
-- Code Concepts
-- =============================================================================

CREATE TABLE IF NOT EXISTS chunk_concepts (
    id              SERIAL PRIMARY KEY,
    chunk_id        UUID NOT NULL REFERENCES code_chunks(chunk_id) ON DELETE CASCADE,
    concept_name    TEXT NOT NULL,
    concept_type    TEXT,

    -- Confidence
    confidence      REAL CHECK (confidence >= 0 AND confidence <= 1),

    -- Extraction metadata
    extractor_model TEXT NOT NULL,
    extractor_version TEXT NOT NULL,

    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT now(),

    UNIQUE (chunk_id, concept_name, extractor_version)
);

CREATE INDEX IF NOT EXISTS idx_cc_chunk ON chunk_concepts (chunk_id);
CREATE INDEX IF NOT EXISTS idx_cc_concept ON chunk_concepts (concept_name);
CREATE INDEX IF NOT EXISTS idx_cc_type ON chunk_concepts (concept_type);

-- =============================================================================
-- Paper-Repo Links
-- =============================================================================

CREATE TABLE IF NOT EXISTS paper_repo_links (
    id              SERIAL PRIMARY KEY,
    doc_id          UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    repo_id         UUID REFERENCES code_repos(repo_id) ON DELETE SET NULL,

    -- Link metadata
    repo_url        TEXT NOT NULL,
    detection_method TEXT,  -- url_pattern, github_api, manual
    confidence      REAL DEFAULT 1.0,
    verified        BOOLEAN DEFAULT FALSE,

    -- Position in paper
    mention_context TEXT,
    page_num        INTEGER,

    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT now(),

    UNIQUE (doc_id, repo_url)
);

CREATE INDEX IF NOT EXISTS idx_prl_doc ON paper_repo_links (doc_id);
CREATE INDEX IF NOT EXISTS idx_prl_repo ON paper_repo_links (repo_id);

-- =============================================================================
-- Unified Code Search Function
-- =============================================================================

CREATE OR REPLACE FUNCTION search_code(
    query_text TEXT,
    query_embedding vector(1024),
    result_limit INTEGER DEFAULT 20,
    repo_filter TEXT DEFAULT NULL,
    language_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    chunk_id UUID,
    file_path TEXT,
    repo_name TEXT,
    chunk_text TEXT,
    chunk_type TEXT,
    name TEXT,
    score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.chunk_id,
        f.file_path,
        r.repo_name,
        c.chunk_text,
        c.chunk_type,
        c.name,
        (1 - (c.embedding <=> query_embedding))::REAL AS score
    FROM code_chunks c
    JOIN code_files f ON c.file_id = f.file_id
    JOIN code_repos r ON f.repo_id = r.repo_id
    WHERE c.embedding IS NOT NULL
        AND (repo_filter IS NULL OR r.repo_name ILIKE '%' || repo_filter || '%')
        AND (language_filter IS NULL OR f.language = language_filter)
    ORDER BY c.embedding <=> query_embedding
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;
