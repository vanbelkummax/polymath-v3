-- Polymath v3 Concepts Schema
-- Extracted concepts from passages and code

-- =============================================================================
-- Passage Concepts
-- =============================================================================

CREATE TABLE IF NOT EXISTS passage_concepts (
    id              SERIAL PRIMARY KEY,
    passage_id      UUID NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    concept_name    TEXT NOT NULL,
    concept_type    TEXT,  -- method, problem, domain, dataset, metric, entity, etc.

    -- Normalization
    canonical_name  TEXT,  -- Normalized form (lowercase, stemmed)
    aliases         JSONB,  -- Alternative names

    -- Confidence and evidence
    confidence      REAL CHECK (confidence >= 0 AND confidence <= 1),
    evidence        JSONB,  -- {quote: "...", context: "...", position: {...}}

    -- Extraction metadata
    extractor_model TEXT NOT NULL,  -- e.g., "gemini-2.0-flash", "haiku"
    extractor_version TEXT NOT NULL,  -- e.g., "v3.0.0"

    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT now(),

    -- Prevent duplicate extractions
    UNIQUE (passage_id, concept_name, extractor_version)
);

-- Indexes for passage_concepts
CREATE INDEX IF NOT EXISTS idx_pc_passage ON passage_concepts (passage_id);
CREATE INDEX IF NOT EXISTS idx_pc_concept ON passage_concepts (concept_name);
CREATE INDEX IF NOT EXISTS idx_pc_canonical ON passage_concepts (canonical_name);
CREATE INDEX IF NOT EXISTS idx_pc_type ON passage_concepts (concept_type);
CREATE INDEX IF NOT EXISTS idx_pc_confidence ON passage_concepts (confidence) WHERE confidence >= 0.7;
CREATE INDEX IF NOT EXISTS idx_pc_extractor ON passage_concepts (extractor_model, extractor_version);

-- Trigram index for fuzzy concept search
CREATE INDEX IF NOT EXISTS idx_pc_concept_trgm ON passage_concepts USING GIN (concept_name gin_trgm_ops);

-- =============================================================================
-- Concept Statistics (Materialized View)
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS concept_stats AS
SELECT
    concept_name,
    concept_type,
    COUNT(DISTINCT passage_id) AS passage_count,
    COUNT(DISTINCT p.doc_id) AS document_count,
    AVG(confidence) AS avg_confidence,
    array_agg(DISTINCT concept_type) FILTER (WHERE concept_type IS NOT NULL) AS types
FROM passage_concepts pc
JOIN passages p ON pc.passage_id = p.passage_id
WHERE confidence >= 0.5
GROUP BY concept_name, concept_type
ORDER BY passage_count DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_concept_stats_name ON concept_stats (concept_name, concept_type);

-- Refresh function
CREATE OR REPLACE FUNCTION refresh_concept_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY concept_stats;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Concept Types Enum (for reference)
-- =============================================================================

COMMENT ON TABLE passage_concepts IS '
Concept types:
- method: Techniques, algorithms, tools (e.g., "spatial transcriptomics", "optimal transport")
- problem: Challenges being addressed (e.g., "cell type deconvolution", "batch effects")
- domain: Research fields (e.g., "computational pathology", "drug discovery")
- dataset: Specific datasets (e.g., "Visium HD", "10x Xenium")
- metric: Evaluation metrics (e.g., "PCC", "SSIM", "AUC")
- entity: Genes, proteins, diseases (e.g., "TP53", "EGFR", "Alzheimer")
- mechanism: Underlying mechanisms (e.g., "attention", "diffusion")
- data_structure: Data representations (e.g., "point cloud", "graph")
';

-- =============================================================================
-- Concept Cooccurrence (for BridgeMine)
-- =============================================================================

CREATE TABLE IF NOT EXISTS concept_cooccurrence (
    concept_a       TEXT NOT NULL,
    concept_b       TEXT NOT NULL,
    cooccurrence_count INTEGER NOT NULL DEFAULT 1,
    document_count  INTEGER NOT NULL DEFAULT 1,
    avg_distance    REAL,  -- Average character distance in passages
    updated_at      TIMESTAMPTZ DEFAULT now(),

    PRIMARY KEY (concept_a, concept_b),
    CHECK (concept_a < concept_b)  -- Ensure consistent ordering
);

CREATE INDEX IF NOT EXISTS idx_cooccur_a ON concept_cooccurrence (concept_a);
CREATE INDEX IF NOT EXISTS idx_cooccur_b ON concept_cooccurrence (concept_b);
CREATE INDEX IF NOT EXISTS idx_cooccur_count ON concept_cooccurrence (cooccurrence_count DESC);

-- Function to update cooccurrence
CREATE OR REPLACE FUNCTION update_concept_cooccurrence()
RETURNS void AS $$
BEGIN
    INSERT INTO concept_cooccurrence (concept_a, concept_b, cooccurrence_count, document_count)
    SELECT
        LEAST(pc1.concept_name, pc2.concept_name) AS concept_a,
        GREATEST(pc1.concept_name, pc2.concept_name) AS concept_b,
        COUNT(*) AS cooccurrence_count,
        COUNT(DISTINCT p.doc_id) AS document_count
    FROM passage_concepts pc1
    JOIN passage_concepts pc2 ON pc1.passage_id = pc2.passage_id
        AND pc1.concept_name < pc2.concept_name
    JOIN passages p ON pc1.passage_id = p.passage_id
    WHERE pc1.confidence >= 0.7 AND pc2.confidence >= 0.7
    GROUP BY LEAST(pc1.concept_name, pc2.concept_name),
             GREATEST(pc1.concept_name, pc2.concept_name)
    ON CONFLICT (concept_a, concept_b) DO UPDATE SET
        cooccurrence_count = EXCLUDED.cooccurrence_count,
        document_count = EXCLUDED.document_count,
        updated_at = now();
END;
$$ LANGUAGE plpgsql;
