-- Polymath v3 Schema: Soft Delete Support
-- Adds columns to preserve passages when re-ingesting (for annotation preservation)

-- =============================================================================
-- Soft Delete Columns for Passages
-- =============================================================================
-- When soft_delete=True in IngestPipeline, old passages are marked as
-- superseded instead of deleted. This preserves manual annotations.

ALTER TABLE passages
    ADD COLUMN IF NOT EXISTS is_superseded BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS superseded_by_batch VARCHAR(100);

-- Index for filtering active passages (most queries should use this)
CREATE INDEX IF NOT EXISTS idx_passages_active
    ON passages (doc_id)
    WHERE is_superseded = FALSE;

-- =============================================================================
-- User Annotations Table (Future Use)
-- =============================================================================
-- When you add manual annotation features, store them here.
-- The annotation links to passage_id, not doc_id, so soft delete preserves them.

CREATE TABLE IF NOT EXISTS passage_annotations (
    annotation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    passage_id UUID NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    user_id VARCHAR(100),  -- Future: user authentication
    annotation_type VARCHAR(50) NOT NULL,  -- 'highlight', 'note', 'tag', 'correction'
    content TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_annotations_passage
    ON passage_annotations (passage_id);

CREATE INDEX IF NOT EXISTS idx_annotations_type
    ON passage_annotations (annotation_type);

-- =============================================================================
-- View: Active Passages Only
-- =============================================================================
-- Use this view for search queries to exclude superseded passages

CREATE OR REPLACE VIEW active_passages AS
SELECT *
FROM passages
WHERE is_superseded = FALSE OR is_superseded IS NULL;

COMMENT ON VIEW active_passages IS
    'Use this view for searches. Excludes passages marked as superseded during re-ingestion.';
