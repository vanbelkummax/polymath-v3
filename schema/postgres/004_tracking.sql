-- Polymath v3 Tracking Schema
-- Job tracking, migrations, and audit tables

-- =============================================================================
-- Ingest Batches
-- =============================================================================

CREATE TABLE IF NOT EXISTS ingest_batches (
    batch_id        TEXT PRIMARY KEY,
    source          TEXT NOT NULL,  -- pdf_archive, zotero_import, manual, github
    description     TEXT,

    -- Progress
    total_items     INTEGER NOT NULL DEFAULT 0,
    processed       INTEGER NOT NULL DEFAULT 0,
    succeeded       INTEGER NOT NULL DEFAULT 0,
    failed          INTEGER NOT NULL DEFAULT 0,
    skipped         INTEGER NOT NULL DEFAULT 0,

    -- Status
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed, cancelled

    -- Timing
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,

    -- Error tracking
    last_error      TEXT,
    error_count     INTEGER DEFAULT 0,

    -- Metadata
    config          JSONB,  -- Batch configuration
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_batch_status ON ingest_batches (status);
CREATE INDEX IF NOT EXISTS idx_batch_created ON ingest_batches (created_at);

-- =============================================================================
-- Knowledge Base Migrations
-- =============================================================================

CREATE TABLE IF NOT EXISTS kb_migrations (
    job_name        TEXT PRIMARY KEY,
    job_type        TEXT,  -- backfill, sync, rebuild, extraction

    -- Progress tracking
    cursor_position TEXT,  -- Last processed ID or offset
    cursor_type     TEXT,  -- uuid, timestamp, integer

    -- Status
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed, paused

    -- Counts
    items_total     INTEGER,
    items_processed INTEGER DEFAULT 0,
    items_succeeded INTEGER DEFAULT 0,
    items_failed    INTEGER DEFAULT 0,

    -- Timing
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    last_activity   TIMESTAMPTZ,

    -- Error handling
    error_message   TEXT,
    retry_count     INTEGER DEFAULT 0,
    max_retries     INTEGER DEFAULT 3,

    -- Configuration
    config          JSONB,

    -- Worker tracking (for parallel jobs)
    worker_id       INTEGER,
    num_workers     INTEGER,

    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_migration_status ON kb_migrations (status);
CREATE INDEX IF NOT EXISTS idx_migration_type ON kb_migrations (job_type);

-- =============================================================================
-- GCP Batch Jobs
-- =============================================================================

CREATE TABLE IF NOT EXISTS gcp_batch_jobs (
    job_id          TEXT PRIMARY KEY,
    job_name        TEXT NOT NULL,
    job_type        TEXT NOT NULL,  -- concept_extraction, embedding, graph_sync

    -- GCP metadata
    gcp_job_name    TEXT,  -- Full GCP resource name
    gcp_project     TEXT,
    gcp_location    TEXT,

    -- Input/Output
    input_uri       TEXT,  -- GCS input location
    output_uri      TEXT,  -- GCS output location
    input_count     INTEGER,

    -- Status
    status          TEXT NOT NULL DEFAULT 'submitted',  -- submitted, running, succeeded, failed, cancelled

    -- Timing
    submitted_at    TIMESTAMPTZ DEFAULT now(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,

    -- Results
    output_count    INTEGER,
    error_count     INTEGER,
    error_message   TEXT,

    -- Cost tracking
    estimated_cost  REAL,
    actual_cost     REAL
);

CREATE INDEX IF NOT EXISTS idx_gcp_job_status ON gcp_batch_jobs (status);
CREATE INDEX IF NOT EXISTS idx_gcp_job_type ON gcp_batch_jobs (job_type);

-- =============================================================================
-- Ingest Queue
-- =============================================================================

CREATE TABLE IF NOT EXISTS ingest_queue (
    id              SERIAL PRIMARY KEY,
    source_type     TEXT NOT NULL,  -- pdf, repo, url
    source_path     TEXT NOT NULL,

    -- Priority (higher = process first)
    priority        INTEGER DEFAULT 0,

    -- Status
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending, processing, completed, failed, skipped

    -- Assignment
    batch_id        TEXT REFERENCES ingest_batches(batch_id),
    worker_id       TEXT,
    locked_at       TIMESTAMPTZ,

    -- Results
    doc_id          UUID REFERENCES documents(doc_id),
    error_message   TEXT,
    retry_count     INTEGER DEFAULT 0,

    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT now(),
    processed_at    TIMESTAMPTZ,

    UNIQUE (source_type, source_path)
);

CREATE INDEX IF NOT EXISTS idx_queue_status ON ingest_queue (status);
CREATE INDEX IF NOT EXISTS idx_queue_priority ON ingest_queue (priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_queue_batch ON ingest_queue (batch_id);

-- =============================================================================
-- Audit Log
-- =============================================================================

CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGSERIAL PRIMARY KEY,
    event_type      TEXT NOT NULL,  -- ingest, search, extraction, sync, error
    event_subtype   TEXT,

    -- Target
    target_type     TEXT,  -- document, passage, concept, repo
    target_id       TEXT,

    -- Details
    details         JSONB,
    error_message   TEXT,

    -- Context
    batch_id        TEXT,
    user_id         TEXT,
    session_id      TEXT,

    -- Timing
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_log (event_type);
CREATE INDEX IF NOT EXISTS idx_audit_target ON audit_log (target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log (created_at);

-- Partition by month for large-scale deployments
-- CREATE TABLE audit_log_2026_01 PARTITION OF audit_log
--     FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

-- =============================================================================
-- System Stats (for monitoring)
-- =============================================================================

CREATE TABLE IF NOT EXISTS system_stats (
    stat_name       TEXT PRIMARY KEY,
    stat_value      BIGINT,
    stat_metadata   JSONB,
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- Function to update stats
CREATE OR REPLACE FUNCTION update_system_stats()
RETURNS void AS $$
BEGIN
    INSERT INTO system_stats (stat_name, stat_value)
    VALUES
        ('documents_count', (SELECT COUNT(*) FROM documents)),
        ('passages_count', (SELECT COUNT(*) FROM passages)),
        ('passages_with_embeddings', (SELECT COUNT(*) FROM passages WHERE embedding IS NOT NULL)),
        ('concepts_count', (SELECT COUNT(*) FROM passage_concepts)),
        ('code_repos_count', (SELECT COUNT(*) FROM code_repos)),
        ('code_chunks_count', (SELECT COUNT(*) FROM code_chunks))
    ON CONFLICT (stat_name) DO UPDATE SET
        stat_value = EXCLUDED.stat_value,
        updated_at = now();
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Claim next item from queue
CREATE OR REPLACE FUNCTION claim_queue_item(
    p_worker_id TEXT,
    p_source_type TEXT DEFAULT NULL
)
RETURNS TABLE (id INTEGER, source_type TEXT, source_path TEXT) AS $$
DECLARE
    v_item RECORD;
BEGIN
    SELECT q.id, q.source_type, q.source_path INTO v_item
    FROM ingest_queue q
    WHERE q.status = 'pending'
        AND (p_source_type IS NULL OR q.source_type = p_source_type)
    ORDER BY q.priority DESC, q.created_at
    LIMIT 1
    FOR UPDATE SKIP LOCKED;

    IF v_item IS NOT NULL THEN
        UPDATE ingest_queue
        SET status = 'processing',
            worker_id = p_worker_id,
            locked_at = now()
        WHERE ingest_queue.id = v_item.id;

        RETURN QUERY SELECT v_item.id, v_item.source_type, v_item.source_path;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Mark queue item complete
CREATE OR REPLACE FUNCTION complete_queue_item(
    p_id INTEGER,
    p_doc_id UUID DEFAULT NULL,
    p_error TEXT DEFAULT NULL
)
RETURNS void AS $$
BEGIN
    UPDATE ingest_queue
    SET status = CASE WHEN p_error IS NULL THEN 'completed' ELSE 'failed' END,
        doc_id = p_doc_id,
        error_message = p_error,
        processed_at = now()
    WHERE id = p_id;
END;
$$ LANGUAGE plpgsql;
