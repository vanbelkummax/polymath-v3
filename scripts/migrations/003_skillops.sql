-- Migration: 003_skillops.sql
-- Purpose: Add SkillOps tables for skill usage tracking, canonical skills, and model mentions
-- Date: 2026-01-17

BEGIN;

-- ============================================================
-- SKILL USAGE LOGGING (Gate 4 Validation)
-- ============================================================

CREATE TABLE IF NOT EXISTS skill_usage_log (
    usage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill_name TEXT NOT NULL,

    -- Task context
    task_description TEXT,
    task_type TEXT,  -- 'analysis', 'visualization', 'pipeline', etc.

    -- Outcome tracking
    outcome TEXT NOT NULL CHECK (outcome IN ('success', 'failure', 'partial')),
    oracle_passed BOOLEAN,
    failure_notes TEXT,

    -- Performance metrics (optional)
    execution_time_ms INTEGER,
    error_type TEXT,  -- 'ImportError', 'ValueError', 'RuntimeError', etc.

    -- Session context
    session_id TEXT,
    user_context TEXT,  -- anonymized context like "spatial analysis task"

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_skill_usage_skill_name ON skill_usage_log(skill_name);
CREATE INDEX IF NOT EXISTS idx_skill_usage_outcome ON skill_usage_log(outcome);
CREATE INDEX IF NOT EXISTS idx_skill_usage_created ON skill_usage_log(created_at DESC);

-- ============================================================
-- HUGGINGFACE MODEL MENTIONS (Unresolved References)
-- ============================================================
-- Captures model IDs mentioned in papers before they're resolved to hf_models

CREATE TABLE IF NOT EXISTS hf_model_mentions (
    mention_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,

    -- Detected model reference
    model_id_raw TEXT NOT NULL,  -- Raw string like "bert-base-uncased" or "facebook/dinov2"
    model_id_normalized TEXT,     -- Normalized to "org/model" format

    -- Context
    passage_id UUID REFERENCES passages(passage_id),
    context TEXT,  -- Surrounding text
    confidence FLOAT DEFAULT 0.5,  -- Detection confidence

    -- Resolution status
    resolved BOOLEAN DEFAULT false,
    resolved_to_model_id TEXT REFERENCES hf_models(model_id),
    resolution_method TEXT,  -- 'exact_match', 'fuzzy_match', 'api_lookup', 'manual'

    -- Timestamps
    discovered_at TIMESTAMPTZ DEFAULT now(),
    resolved_at TIMESTAMPTZ,

    UNIQUE(doc_id, model_id_raw, passage_id)
);

CREATE INDEX IF NOT EXISTS idx_hf_mentions_doc ON hf_model_mentions(doc_id);
CREATE INDEX IF NOT EXISTS idx_hf_mentions_model ON hf_model_mentions(model_id_normalized);
CREATE INDEX IF NOT EXISTS idx_hf_mentions_unresolved ON hf_model_mentions(resolved) WHERE resolved = false;

-- ============================================================
-- CANONICAL SKILL EXTENSIONS
-- ============================================================
-- Extend paper_skills for deduplication and canonical skill tracking

-- Add canonical skill tracking columns
ALTER TABLE paper_skills
    ADD COLUMN IF NOT EXISTS canonical_skill_id UUID REFERENCES paper_skills(skill_id),
    ADD COLUMN IF NOT EXISTS is_canonical BOOLEAN DEFAULT false,
    ADD COLUMN IF NOT EXISTS merge_count INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_merge_at TIMESTAMPTZ;

-- Index for finding canonical skills
CREATE INDEX IF NOT EXISTS idx_paper_skills_canonical ON paper_skills(is_canonical) WHERE is_canonical = true;
CREATE INDEX IF NOT EXISTS idx_paper_skills_canonical_ref ON paper_skills(canonical_skill_id) WHERE canonical_skill_id IS NOT NULL;

-- ============================================================
-- SKILL BRIDGES EXTENSIONS
-- ============================================================
-- Extend skill_bridges for validated bridge tracking

ALTER TABLE skill_bridges
    ADD COLUMN IF NOT EXISTS validation_status TEXT DEFAULT 'proposed',  -- proposed, validated, rejected
    ADD COLUMN IF NOT EXISTS validated_by TEXT,  -- 'oracle', 'manual', 'usage'
    ADD COLUMN IF NOT EXISTS validation_notes TEXT,
    ADD COLUMN IF NOT EXISTS usage_count INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMPTZ;

-- ============================================================
-- VIEWS FOR SKILLOPS DASHBOARD
-- ============================================================

-- View: Skill promotion candidates (ready for Gate 4)
CREATE OR REPLACE VIEW v_promotion_candidates AS
SELECT
    ps.skill_id,
    ps.skill_name,
    ps.skill_type,
    ps.original_domain,
    ps.status,
    ps.confidence,
    COALESCE(jsonb_array_length(ps.source_passages::jsonb), 0) as passage_count,
    COALESCE(array_length(ps.source_code_chunks, 1), 0) as code_chunk_count,
    (SELECT COUNT(*) FROM skill_usage_log sul
     WHERE sul.skill_name = ps.skill_name AND sul.outcome = 'success') as success_count,
    ps.created_at
FROM paper_skills ps
WHERE ps.status = 'draft'
AND (
    COALESCE(jsonb_array_length(ps.source_passages::jsonb), 0) >= 2
    OR (
        COALESCE(jsonb_array_length(ps.source_passages::jsonb), 0) >= 1
        AND COALESCE(array_length(ps.source_code_chunks, 1), 0) >= 1
    )
);

-- View: Skill usage summary
CREATE OR REPLACE VIEW v_skill_usage_summary AS
SELECT
    skill_name,
    COUNT(*) as total_uses,
    SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successes,
    SUM(CASE WHEN outcome = 'failure' THEN 1 ELSE 0 END) as failures,
    SUM(CASE WHEN outcome = 'partial' THEN 1 ELSE 0 END) as partials,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) / COUNT(*), 1) as success_rate,
    ROUND(100.0 * SUM(CASE WHEN oracle_passed THEN 1 ELSE 0 END) / NULLIF(SUM(CASE WHEN oracle_passed IS NOT NULL THEN 1 ELSE 0 END), 0), 1) as oracle_pass_rate,
    MIN(created_at) as first_use,
    MAX(created_at) as last_use
FROM skill_usage_log
GROUP BY skill_name
ORDER BY total_uses DESC;

-- View: Unresolved HF models
CREATE OR REPLACE VIEW v_unresolved_hf_models AS
SELECT
    model_id_normalized,
    COUNT(DISTINCT doc_id) as paper_count,
    array_agg(DISTINCT model_id_raw) as raw_variants,
    MAX(confidence) as max_confidence,
    MIN(discovered_at) as first_seen
FROM hf_model_mentions
WHERE resolved = false
GROUP BY model_id_normalized
ORDER BY paper_count DESC;

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

-- Function: Log skill usage
CREATE OR REPLACE FUNCTION log_skill_usage(
    p_skill_name TEXT,
    p_outcome TEXT,
    p_oracle_passed BOOLEAN DEFAULT NULL,
    p_task_description TEXT DEFAULT NULL,
    p_failure_notes TEXT DEFAULT NULL,
    p_execution_time_ms INTEGER DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_usage_id UUID;
BEGIN
    INSERT INTO skill_usage_log (
        skill_name, outcome, oracle_passed,
        task_description, failure_notes, execution_time_ms
    ) VALUES (
        p_skill_name, p_outcome, p_oracle_passed,
        p_task_description, p_failure_notes, p_execution_time_ms
    ) RETURNING usage_id INTO v_usage_id;

    RETURN v_usage_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Mark skill as canonical
CREATE OR REPLACE FUNCTION mark_skill_canonical(
    p_skill_id UUID,
    p_merge_from_ids UUID[] DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    -- Set this skill as canonical
    UPDATE paper_skills
    SET is_canonical = true,
        merge_count = COALESCE(array_length(p_merge_from_ids, 1), 0),
        last_merge_at = CASE WHEN p_merge_from_ids IS NOT NULL THEN now() ELSE last_merge_at END
    WHERE skill_id = p_skill_id;

    -- Point merged skills to canonical
    IF p_merge_from_ids IS NOT NULL THEN
        UPDATE paper_skills
        SET canonical_skill_id = p_skill_id,
            status = 'merged'
        WHERE skill_id = ANY(p_merge_from_ids);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function: Resolve HF model mention
CREATE OR REPLACE FUNCTION resolve_hf_mention(
    p_mention_id UUID,
    p_model_id TEXT,
    p_method TEXT DEFAULT 'manual'
) RETURNS VOID AS $$
BEGIN
    UPDATE hf_model_mentions
    SET resolved = true,
        resolved_to_model_id = p_model_id,
        resolution_method = p_method,
        resolved_at = now()
    WHERE mention_id = p_mention_id;
END;
$$ LANGUAGE plpgsql;

COMMIT;

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================
-- Run these after migration to verify:

-- SELECT COUNT(*) FROM skill_usage_log;
-- SELECT COUNT(*) FROM hf_model_mentions;
-- SELECT column_name FROM information_schema.columns WHERE table_name = 'paper_skills' AND column_name IN ('canonical_skill_id', 'is_canonical');
-- SELECT * FROM v_skill_usage_summary LIMIT 5;
