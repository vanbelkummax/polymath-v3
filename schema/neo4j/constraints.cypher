// Polymath v3 Neo4j Schema
// Constraints, indexes, and initial setup

// =============================================================================
// Core Entity Constraints
// =============================================================================

// Passages - linked to Postgres
CREATE CONSTRAINT passage_id_unique IF NOT EXISTS
    FOR (p:Passage) REQUIRE p.passage_id IS UNIQUE;

// Papers - document metadata
CREATE CONSTRAINT paper_doc_id_unique IF NOT EXISTS
    FOR (p:Paper) REQUIRE p.doc_id IS UNIQUE;

CREATE CONSTRAINT paper_doi_unique IF NOT EXISTS
    FOR (p:Paper) REQUIRE p.doi IS UNIQUE;

// Code chunks - linked to Postgres
CREATE CONSTRAINT code_chunk_id_unique IF NOT EXISTS
    FOR (c:Code) REQUIRE c.chunk_id IS UNIQUE;

// Repositories
CREATE CONSTRAINT repo_full_name_unique IF NOT EXISTS
    FOR (r:Repo) REQUIRE r.full_name IS UNIQUE;

// =============================================================================
// Concept Type Constraints
// Each concept type gets its own label for efficient querying
// =============================================================================

CREATE CONSTRAINT method_name_unique IF NOT EXISTS
    FOR (m:METHOD) REQUIRE m.name IS UNIQUE;

CREATE CONSTRAINT problem_name_unique IF NOT EXISTS
    FOR (p:PROBLEM) REQUIRE p.name IS UNIQUE;

CREATE CONSTRAINT domain_name_unique IF NOT EXISTS
    FOR (d:DOMAIN) REQUIRE d.name IS UNIQUE;

CREATE CONSTRAINT dataset_name_unique IF NOT EXISTS
    FOR (d:DATASET) REQUIRE d.name IS UNIQUE;

CREATE CONSTRAINT metric_name_unique IF NOT EXISTS
    FOR (m:METRIC) REQUIRE m.name IS UNIQUE;

CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
    FOR (e:ENTITY) REQUIRE e.name IS UNIQUE;

CREATE CONSTRAINT mechanism_name_unique IF NOT EXISTS
    FOR (m:MECHANISM) REQUIRE m.name IS UNIQUE;

CREATE CONSTRAINT data_structure_name_unique IF NOT EXISTS
    FOR (d:DATA_STRUCTURE) REQUIRE d.name IS UNIQUE;

// Generic CONCEPT label (for cross-type queries)
CREATE CONSTRAINT concept_name_unique IF NOT EXISTS
    FOR (c:CONCEPT) REQUIRE c.name IS UNIQUE;

// =============================================================================
// Fulltext Indexes for Search
// =============================================================================

// Search across all concept types
CREATE FULLTEXT INDEX concept_name_fulltext IF NOT EXISTS
    FOR (n:METHOD|PROBLEM|DOMAIN|DATASET|METRIC|ENTITY|MECHANISM|DATA_STRUCTURE|CONCEPT)
    ON EACH [n.name, n.aliases];

// Paper title search
CREATE FULLTEXT INDEX paper_title_fulltext IF NOT EXISTS
    FOR (p:Paper) ON EACH [p.title];

// Passage content search
CREATE FULLTEXT INDEX passage_content_fulltext IF NOT EXISTS
    FOR (p:Passage) ON EACH [p.content_preview];

// =============================================================================
// Regular Indexes for Performance
// =============================================================================

// Paper metadata
CREATE INDEX paper_year_idx IF NOT EXISTS FOR (p:Paper) ON (p.year);
CREATE INDEX paper_venue_idx IF NOT EXISTS FOR (p:Paper) ON (p.venue);

// Passage linkage
CREATE INDEX passage_doc_idx IF NOT EXISTS FOR (p:Passage) ON (p.doc_id);
CREATE INDEX passage_section_idx IF NOT EXISTS FOR (p:Passage) ON (p.section);

// Concept stats
CREATE INDEX concept_mention_count_idx IF NOT EXISTS FOR (c:CONCEPT) ON (c.mention_count);

// Code
CREATE INDEX code_repo_idx IF NOT EXISTS FOR (c:Code) ON (c.repo_name);
CREATE INDEX code_language_idx IF NOT EXISTS FOR (c:Code) ON (c.language);

// =============================================================================
// Relationship Types (Documentation)
// =============================================================================

// Passage -> Concept
// (p:Passage)-[:MENTIONS {confidence: 0.9, extractor: "gemini"}]->(c:METHOD)

// Paper -> Concept (aggregated from passages)
// (paper:Paper)-[:DISCUSSES {count: 5, avg_confidence: 0.85}]->(c:METHOD)

// Concept -> Concept relationships
// IMPORTANT: SOLVES includes context property for domain-specific validity
// A method might solve a problem in histology but fail in cytology
// (m:METHOD)-[:SOLVES {context: "spatial", organism: "human", cell_type: null}]->(p:PROBLEM)
//
// Context properties:
//   - context: Domain/modality where this works (e.g., "spatial", "bulk", "single-cell")
//   - organism: Species context (e.g., "human", "mouse", "in vitro")
//   - cell_type: Cell type context (e.g., "tumor", "immune", null for general)
//   - experimental_condition: Specific conditions (e.g., "FFPE", "fresh frozen")
//   - confidence: Extraction confidence score (0-1)
//   - source_count: Number of papers supporting this relationship
//
// Examples:
// (m:METHOD {name: "Cell2location"})-[:SOLVES {context: "spatial", organism: "human"}]->(p:PROBLEM {name: "deconvolution"})
// (m:METHOD {name: "RCTD"})-[:SOLVES {context: "spatial", organism: "mouse"}]->(p:PROBLEM {name: "deconvolution"})

// (m:METHOD)-[:APPLIES_TO]->(d:DATASET)
// (m:METHOD)-[:IMPLEMENTS]->(mech:MECHANISM)
// (m:METHOD)-[:OPERATES_ON]->(ds:DATA_STRUCTURE)
// (m1:METHOD)-[:OUTPERFORMS {metric: "PCC", value: 0.05}]->(m2:METHOD)
// (m1:METHOD)-[:SIMILAR_TO {similarity: 0.85}]->(m2:METHOD)

// Paper relationships
// (p1:Paper)-[:CITES]->(p2:Paper)
// (p:Paper)-[:HAS_CODE]->(r:Repo)

// =============================================================================
// Utility Queries (for reference)
// =============================================================================

// Find methods that solve a problem
// MATCH (m:METHOD)-[:SOLVES]->(p:PROBLEM {name: 'cell type deconvolution'})
// RETURN m.name, m.mention_count ORDER BY m.mention_count DESC

// Find cross-domain method applications
// MATCH (m:METHOD)-[:APPLIES_TO]->(d1:DOMAIN)
// MATCH (m)-[:APPLIES_TO]->(d2:DOMAIN)
// WHERE d1 <> d2
// RETURN m.name, collect(DISTINCT d1.name) + collect(DISTINCT d2.name) AS domains

// Find similar methods via shared mechanisms
// MATCH (m1:METHOD)-[:IMPLEMENTS]->(mech:MECHANISM)<-[:IMPLEMENTS]-(m2:METHOD)
// WHERE m1 <> m2
// RETURN m1.name, m2.name, mech.name

// BridgeMine gap detection
// MATCH (m:METHOD)-[:SOLVES]->(p:PROBLEM)
// WHERE NOT EXISTS((m)-[:APPLIES_TO]->(:DOMAIN {name: 'spatial_transcriptomics'}))
// AND EXISTS((m)-[:APPLIES_TO]->(:DOMAIN))
// RETURN m.name, collect(p.name) AS problems_solved
// ORDER BY size(collect(p.name)) DESC

// =============================================================================
// Initial Data (run after constraint creation)
// =============================================================================

// Create root domain node
MERGE (d:DOMAIN:CONCEPT {name: 'spatial_transcriptomics'})
SET d.description = 'Spatially resolved gene expression analysis',
    d.aliases = ['spatial tx', 'spatial omics', 'ST'];

// Create common method categories
MERGE (m:METHOD:CONCEPT {name: 'deep_learning'})
SET m.description = 'Neural network-based approaches',
    m.is_category = true;

MERGE (m:METHOD:CONCEPT {name: 'optimal_transport'})
SET m.description = 'Mathematical framework for distribution matching',
    m.aliases = ['OT', 'Wasserstein'];
