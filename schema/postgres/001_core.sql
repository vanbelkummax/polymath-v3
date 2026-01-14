-- Polymath v3 Core Schema
-- Postgres with pgvector for unified vector + relational storage

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- Documents Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS documents (
    doc_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT NOT NULL,
    title_hash      TEXT UNIQUE NOT NULL,  -- SHA256 for deduplication

    -- Authors (stored as array for flexibility)
    authors         TEXT[],

    -- Publication metadata
    year            INTEGER,
    venue           TEXT,
    publication_type TEXT,  -- journal, conference, preprint, book, etc.

    -- Identifiers (priority: doi > pmid > arxiv_id)
    doi             TEXT UNIQUE,
    pmid            TEXT UNIQUE,
    arxiv_id        TEXT UNIQUE,
    openalex_id     TEXT,
    semantic_scholar_id TEXT,

    -- Zotero integration
    zotero_key      TEXT,
    zotero_synced_at TIMESTAMPTZ,

    -- Content metadata
    abstract        TEXT,
    keywords        TEXT[],

    -- Citation metrics
    cited_by_count  INTEGER,
    references_count INTEGER,

    -- OpenAlex concept tags
    openalex_concepts JSONB,

    -- Source tracking
    pdf_path        TEXT,
    pdf_hash        TEXT,  -- For change detection
    parser_version  TEXT,
    metadata_source TEXT,  -- zotero, pdf2doi, crossref, arxiv, filename
    metadata_confidence REAL CHECK (metadata_confidence >= 0 AND metadata_confidence <= 1),

    -- Batch processing
    ingest_batch    TEXT,

    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),

    -- Sync tracking
    vector_synced_at TIMESTAMPTZ,
    graph_synced_at  TIMESTAMPTZ
);

-- Indexes for documents
CREATE INDEX IF NOT EXISTS idx_doc_title_trgm ON documents USING GIN (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_doc_year ON documents (year);
CREATE INDEX IF NOT EXISTS idx_doc_ingest_batch ON documents (ingest_batch);
CREATE INDEX IF NOT EXISTS idx_doc_metadata_source ON documents (metadata_source);
CREATE INDEX IF NOT EXISTS idx_doc_created_at ON documents (created_at);

-- =============================================================================
-- Passages Table (with pgvector embeddings)
-- =============================================================================

CREATE TABLE IF NOT EXISTS passages (
    passage_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,

    -- Content
    passage_text    TEXT NOT NULL,

    -- Structure preservation (from markdown-aware chunking)
    section         TEXT,  -- Header context (e.g., "Methods", "Results")
    parent_section  TEXT,  -- Parent header for hierarchy
    section_level   INTEGER,  -- H1=1, H2=2, H3=3

    -- Position tracking
    page_num        INTEGER,
    char_start      INTEGER NOT NULL,
    char_end        INTEGER NOT NULL,

    -- Quality signals
    quality_score   REAL DEFAULT 1.0,
    citable         BOOLEAN DEFAULT TRUE,
    has_figures     BOOLEAN DEFAULT FALSE,
    has_tables      BOOLEAN DEFAULT FALSE,
    has_equations   BOOLEAN DEFAULT FALSE,

    -- pgvector embedding (1024-dim BGE-M3)
    embedding       vector(1024),

    -- Processing metadata
    parser_version  TEXT,
    embedding_model TEXT,

    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Indexes for passages
CREATE INDEX IF NOT EXISTS idx_passage_doc ON passages (doc_id);
CREATE INDEX IF NOT EXISTS idx_passage_section ON passages (section);
CREATE INDEX IF NOT EXISTS idx_passage_page ON passages (page_num);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_passage_embedding ON passages
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search index
ALTER TABLE passages ADD COLUMN IF NOT EXISTS search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', passage_text)) STORED;
CREATE INDEX IF NOT EXISTS idx_passage_fts ON passages USING GIN (search_vector);

-- =============================================================================
-- Hybrid Search Function
-- =============================================================================

CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(1024),
    keyword_weight REAL DEFAULT 0.3,
    vector_weight REAL DEFAULT 0.7,
    result_limit INTEGER DEFAULT 20
)
RETURNS TABLE (
    passage_id UUID,
    doc_id UUID,
    title TEXT,
    passage_text TEXT,
    section TEXT,
    vector_score REAL,
    keyword_score REAL,
    combined_score REAL
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT
            p.passage_id,
            p.doc_id,
            d.title,
            p.passage_text,
            p.section,
            (1 - (p.embedding <=> query_embedding))::REAL AS v_score
        FROM passages p
        JOIN documents d ON p.doc_id = d.doc_id
        WHERE p.embedding IS NOT NULL
        ORDER BY p.embedding <=> query_embedding
        LIMIT result_limit * 2
    ),
    keyword_results AS (
        SELECT
            p.passage_id,
            p.doc_id,
            d.title,
            p.passage_text,
            p.section,
            ts_rank(p.search_vector, plainto_tsquery('english', query_text))::REAL AS k_score
        FROM passages p
        JOIN documents d ON p.doc_id = d.doc_id
        WHERE p.search_vector @@ plainto_tsquery('english', query_text)
        ORDER BY k_score DESC
        LIMIT result_limit * 2
    ),
    combined AS (
        SELECT
            COALESCE(v.passage_id, k.passage_id) AS passage_id,
            COALESCE(v.doc_id, k.doc_id) AS doc_id,
            COALESCE(v.title, k.title) AS title,
            COALESCE(v.passage_text, k.passage_text) AS passage_text,
            COALESCE(v.section, k.section) AS section,
            COALESCE(v.v_score, 0)::REAL AS vector_score,
            COALESCE(k.k_score, 0)::REAL AS keyword_score,
            (vector_weight * COALESCE(v.v_score, 0) +
             keyword_weight * COALESCE(k.k_score, 0))::REAL AS combined_score
        FROM vector_results v
        FULL OUTER JOIN keyword_results k ON v.passage_id = k.passage_id
    )
    SELECT * FROM combined
    ORDER BY combined_score DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Trigger for updated_at
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
