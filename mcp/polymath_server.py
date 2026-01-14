#!/usr/bin/env python3
"""
Polymath v3 MCP Server.

Provides tools for:
- Semantic search across the knowledge base
- Paper ingestion
- Hallucination detection
- Research gap discovery
- JIT retrieval with synthesis
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("polymath-mcp")

# Initialize MCP server
server = Server("polymath-v3")


# ============================================================================
# Tool Definitions
# ============================================================================

TOOLS = [
    Tool(
        name="semantic_search",
        description="""Search the Polymath knowledge base using hybrid semantic + keyword search.

Returns relevant passages from scientific papers with citation information.

Args:
    query: Search query (natural language)
    n: Number of results (default: 10, max: 50)
    rerank: Whether to apply neural reranking (default: true)
    year_min: Filter by minimum year (optional)
    year_max: Filter by maximum year (optional)""",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "n": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                "rerank": {"type": "boolean", "default": True},
                "year_min": {"type": "integer"},
                "year_max": {"type": "integer"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="retrieve_and_synthesize",
        description="""Retrieve relevant passages and synthesize an answer.

Uses JIT (Just-In-Time) retrieval to find relevant passages and
generate a cited answer to the query.

Args:
    query: Question to answer
    n_passages: Number of passages to retrieve (default: 10)
    max_context: Maximum context length for synthesis (default: 8000)""",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question to answer"},
                "n_passages": {"type": "integer", "default": 10},
                "max_context": {"type": "integer", "default": 8000},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="verify_claim",
        description="""Verify a factual claim against the knowledge base.

Checks if a claim is SUPPORTED, CONTRADICTED, or UNVERIFIABLE
based on evidence in the knowledge base.

Args:
    claim: The factual claim to verify""",
        inputSchema={
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "Claim to verify"},
            },
            "required": ["claim"],
        },
    ),
    Tool(
        name="detect_hallucinations",
        description="""Detect potential hallucinations in generated text.

Extracts verifiable claims from the text and checks each against
the knowledge base. Returns a hallucination score and details.

Args:
    text: Text to check for hallucinations""",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to check"},
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="find_research_gaps",
        description="""Find under-explored research opportunities in a domain.

Uses BridgeMine to identify methods from other domains that could
be applied to problems in the target domain.

Args:
    domain: Target domain (e.g., "spatial_transcriptomics")
    limit: Maximum number of gaps to return (default: 20)""",
        inputSchema={
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Target domain"},
                "limit": {"type": "integer", "default": 20, "maximum": 100},
            },
            "required": ["domain"],
        },
    ),
    Tool(
        name="find_similar_passages",
        description="""Find passages similar to a given passage.

Uses vector similarity to find related content in the knowledge base.

Args:
    passage_id: UUID of the source passage
    n: Number of similar passages to return (default: 10)""",
        inputSchema={
            "type": "object",
            "properties": {
                "passage_id": {"type": "string", "description": "Source passage UUID"},
                "n": {"type": "integer", "default": 10},
            },
            "required": ["passage_id"],
        },
    ),
    Tool(
        name="ingest_pdf",
        description="""Ingest a PDF file into the knowledge base.

Parses the PDF, resolves metadata, chunks text, generates embeddings,
and stores everything in the database.

Args:
    pdf_path: Path to the PDF file
    extract_concepts: Whether to extract concepts using Gemini (default: true)""",
        inputSchema={
            "type": "object",
            "properties": {
                "pdf_path": {"type": "string", "description": "Path to PDF file"},
                "extract_concepts": {"type": "boolean", "default": True},
            },
            "required": ["pdf_path"],
        },
    ),
    Tool(
        name="get_stats",
        description="""Get statistics about the knowledge base.

Returns counts of documents, passages, concepts, and other metrics.""",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="graph_query",
        description="""Execute a Cypher query against the Neo4j concept graph.

For advanced users who want to explore the knowledge graph directly.

Args:
    cypher: Cypher query to execute
    params: Optional query parameters (JSON object)""",
        inputSchema={
            "type": "object",
            "properties": {
                "cypher": {"type": "string", "description": "Cypher query"},
                "params": {"type": "object", "description": "Query parameters"},
            },
            "required": ["cypher"],
        },
    ),
]


# ============================================================================
# Tool Handlers
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "semantic_search":
            return await handle_semantic_search(arguments)
        elif name == "retrieve_and_synthesize":
            return await handle_retrieve_synthesize(arguments)
        elif name == "verify_claim":
            return await handle_verify_claim(arguments)
        elif name == "detect_hallucinations":
            return await handle_detect_hallucinations(arguments)
        elif name == "find_research_gaps":
            return await handle_find_gaps(arguments)
        elif name == "find_similar_passages":
            return await handle_find_similar(arguments)
        elif name == "ingest_pdf":
            return await handle_ingest_pdf(arguments)
        elif name == "get_stats":
            return await handle_get_stats(arguments)
        elif name == "graph_query":
            return await handle_graph_query(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_semantic_search(args: dict) -> list[TextContent]:
    """Handle semantic_search tool."""
    from lib.search.hybrid_search import HybridSearcher

    query = args["query"]
    n = args.get("n", 10)
    rerank = args.get("rerank", True)

    filters = {}
    if "year_min" in args:
        filters["year_min"] = args["year_min"]
    if "year_max" in args:
        filters["year_max"] = args["year_max"]

    searcher = HybridSearcher()
    response = searcher.search(
        query,
        n=n,
        rerank=rerank,
        filters=filters if filters else None,
    )

    # Format results
    output = f"Found {len(response.results)} results for: {query}\n\n"

    for i, r in enumerate(response.results):
        output += f"## [{i+1}] {r.title}"
        if r.year:
            output += f" ({r.year})"
        output += "\n"

        if r.doi:
            output += f"DOI: {r.doi}\n"
        if r.section:
            output += f"Section: {r.section}\n"

        output += f"Score: {r.score:.4f}\n\n"
        output += f"{r.passage_text[:500]}...\n\n"
        output += "---\n\n"

    return [TextContent(type="text", text=output)]


async def handle_retrieve_synthesize(args: dict) -> list[TextContent]:
    """Handle retrieve_and_synthesize tool."""
    from lib.search.jit_retrieval import JITRetriever

    query = args["query"]
    n_passages = args.get("n_passages", 10)
    max_context = args.get("max_context", 8000)

    retriever = JITRetriever()
    result = retriever.retrieve(
        query,
        n_passages=n_passages,
        max_context_length=max_context,
    )

    output = f"# Answer to: {query}\n\n"

    if result.synthesis:
        output += f"{result.synthesis}\n\n"
    else:
        output += "Could not generate synthesis.\n\n"

    output += f"## Sources ({result.sources_used} passages)\n\n"

    for i, p in enumerate(result.passages[:5]):
        output += f"[{i+1}] {p.title}"
        if p.year:
            output += f" ({p.year})"
        output += "\n"

    return [TextContent(type="text", text=output)]


async def handle_verify_claim(args: dict) -> list[TextContent]:
    """Handle verify_claim tool."""
    from lib.validation.hallucination import verify_claim

    claim = args["claim"]
    result = verify_claim(claim)

    output = f"# Claim Verification\n\n"
    output += f"**Claim:** {claim}\n\n"
    output += f"**Verdict:** {result.status.value.upper()}\n"
    output += f"**Confidence:** {result.confidence:.2f}\n\n"

    if result.reasoning:
        output += f"**Reasoning:** {result.reasoning}\n\n"

    if result.evidence:
        output += "## Evidence\n\n"
        for i, e in enumerate(result.evidence[:3]):
            output += f"[{i+1}] {e.title}: {e.passage_text[:200]}...\n\n"

    return [TextContent(type="text", text=output)]


async def handle_detect_hallucinations(args: dict) -> list[TextContent]:
    """Handle detect_hallucinations tool."""
    from lib.validation.hallucination import detect_hallucinations

    text = args["text"]
    report = detect_hallucinations(text)

    output = f"# Hallucination Detection Report\n\n"
    output += f"**Hallucination Score:** {report.hallucination_score:.2f}\n"
    output += f"**Reliable:** {'Yes' if report.is_reliable else 'No'}\n\n"

    output += f"## Claim Analysis\n"
    output += f"- Supported: {report.supported_count}\n"
    output += f"- Contradicted: {report.contradicted_count}\n"
    output += f"- Unverifiable: {report.unverifiable_count}\n\n"

    if report.results:
        output += "## Details\n\n"
        for r in report.results:
            status_emoji = {
                "supported": "✓",
                "contradicted": "✗",
                "unverifiable": "?",
            }.get(r.status.value, "?")

            output += f"{status_emoji} **{r.status.value}**: {r.claim.text}\n"
            if r.reasoning:
                output += f"   {r.reasoning}\n"
            output += "\n"

    return [TextContent(type="text", text=output)]


async def handle_find_gaps(args: dict) -> list[TextContent]:
    """Handle find_research_gaps tool."""
    from lib.bridgemine.gap_detection import GapDetector
    from lib.bridgemine.novelty_check import NoveltyChecker

    domain = args["domain"]
    limit = args.get("limit", 20)

    detector = GapDetector()
    result = detector.find_gaps(domain, limit=limit)

    if not result.candidates:
        return [TextContent(
            type="text",
            text=f"No research gaps found for domain: {domain}"
        )]

    # Check novelty for top candidates
    checker = NoveltyChecker()
    novelty_results = checker.check_batch(result.candidates[:10], delay=0.3)

    output = f"# Research Gaps in {domain}\n\n"
    output += f"Analyzed {result.methods_analyzed} methods, "
    output += f"{result.problems_analyzed} problems\n\n"

    for i, nr in enumerate(novelty_results):
        c = nr.candidate
        output += f"## {i+1}. {c.method_name}\n\n"
        output += f"- **Transfer:** {c.source_problem} → {c.target_problem}\n"
        output += f"- **Problem Similarity:** {c.problem_similarity:.2f}\n"
        output += f"- **Domain Penetration:** {c.domain_penetration:.1%}\n"
        output += f"- **Novelty Score:** {nr.novelty_score:.2f}\n"
        output += f"- **Prior Art:** {nr.pubmed_hits} PubMed, {nr.semantic_scholar_hits} S2\n"
        output += f"- **Assessment:** {'NOVEL' if nr.is_novel else 'WELL-STUDIED'}\n\n"

    return [TextContent(type="text", text=output)]


async def handle_find_similar(args: dict) -> list[TextContent]:
    """Handle find_similar_passages tool."""
    from lib.search.hybrid_search import HybridSearcher

    passage_id = args["passage_id"]
    n = args.get("n", 10)

    searcher = HybridSearcher()
    results = searcher.search_similar(passage_id, n=n)

    if not results:
        return [TextContent(
            type="text",
            text=f"No similar passages found for: {passage_id}"
        )]

    output = f"# Similar Passages\n\n"

    for i, r in enumerate(results):
        output += f"## [{i+1}] {r.title}"
        if r.year:
            output += f" ({r.year})"
        output += f" (similarity: {r.score:.3f})\n\n"
        output += f"{r.passage_text[:300]}...\n\n"

    return [TextContent(type="text", text=output)]


async def handle_ingest_pdf(args: dict) -> list[TextContent]:
    """Handle ingest_pdf tool."""
    from lib.ingest.pipeline import IngestPipeline

    pdf_path = Path(args["pdf_path"])
    extract_concepts = args.get("extract_concepts", True)

    if not pdf_path.exists():
        return [TextContent(type="text", text=f"File not found: {pdf_path}")]

    pipeline = IngestPipeline(extract_concepts=extract_concepts)
    result = pipeline.ingest_pdf(pdf_path)

    if result.success:
        output = f"# Ingestion Successful\n\n"
        output += f"- **Title:** {result.title}\n"
        output += f"- **Doc ID:** {result.doc_id}\n"
        output += f"- **Passages:** {result.passage_count}\n"
        output += f"- **Concepts:** {result.concept_count}\n"
        output += f"- **Metadata Source:** {result.metadata_source}\n"
        output += f"- **Time:** {result.elapsed_seconds:.1f}s\n"
    else:
        output = f"# Ingestion Failed\n\n"
        output += f"**Error:** {result.error}\n"

    return [TextContent(type="text", text=output)]


async def handle_get_stats(args: dict) -> list[TextContent]:
    """Handle get_stats tool."""
    from lib.db.postgres import get_pg_pool

    pool = get_pg_pool()

    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Get counts
            cur.execute("SELECT COUNT(*) as count FROM documents")
            doc_count = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(*) as count FROM passages")
            passage_count = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(*) as count FROM passage_concepts")
            concept_count = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(*) as count FROM passages WHERE embedding IS NOT NULL")
            embedded_count = cur.fetchone()["count"]

    output = f"# Polymath v3 Statistics\n\n"
    output += f"- **Documents:** {doc_count:,}\n"
    output += f"- **Passages:** {passage_count:,}\n"
    output += f"- **Concepts:** {concept_count:,}\n"
    output += f"- **Embedded Passages:** {embedded_count:,}\n"

    return [TextContent(type="text", text=output)]


async def handle_graph_query(args: dict) -> list[TextContent]:
    """Handle graph_query tool."""
    from lib.db.neo4j import get_neo4j_driver

    cypher = args["cypher"]
    params = args.get("params", {})

    driver = get_neo4j_driver()
    records, summary, _ = driver.execute_query(cypher, **params)

    output = f"# Graph Query Results\n\n"
    output += f"**Query:** `{cypher}`\n\n"
    output += f"**Results:** {len(records)} records\n\n"

    if records:
        output += "```json\n"
        # Convert records to JSON-serializable format
        results = []
        for record in records[:20]:
            row = {}
            for key in record.keys():
                value = record[key]
                # Handle Neo4j node/relationship objects
                if hasattr(value, "_properties"):
                    row[key] = dict(value._properties)
                else:
                    row[key] = value
            results.append(row)

        output += json.dumps(results, indent=2, default=str)
        output += "\n```\n"

    return [TextContent(type="text", text=output)]


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run the MCP server."""
    logger.info("Starting Polymath v3 MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
