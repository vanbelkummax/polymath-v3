#!/usr/bin/env python3
"""
Polymath v3 CLI.

Usage:
    polymath search "query"
    polymath ingest /path/to/paper.pdf
    polymath stats
    polymath gaps "domain"
    polymath verify "claim"
"""

import json
import sys
from pathlib import Path

import click

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@click.group()
@click.version_option(version="3.0.0")
def cli():
    """Polymath v3 - Scientific Knowledge Base CLI."""
    pass


# ============================================================================
# Search Commands
# ============================================================================

@cli.command()
@click.argument("query")
@click.option("-n", "--num-results", default=10, help="Number of results")
@click.option("--rerank/--no-rerank", default=True, help="Apply neural reranking")
@click.option("--year-min", type=int, help="Minimum publication year")
@click.option("--year-max", type=int, help="Maximum publication year")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def search(query: str, num_results: int, rerank: bool, year_min: int, year_max: int, output_json: bool):
    """Search the knowledge base."""
    from lib.search.hybrid_search import HybridSearcher

    filters = {}
    if year_min:
        filters["year_min"] = year_min
    if year_max:
        filters["year_max"] = year_max

    searcher = HybridSearcher()
    response = searcher.search(
        query,
        n=num_results,
        rerank=rerank,
        filters=filters if filters else None,
    )

    if output_json:
        results = []
        for r in response.results:
            results.append({
                "passage_id": r.passage_id,
                "doc_id": r.doc_id,
                "title": r.title,
                "year": r.year,
                "doi": r.doi,
                "section": r.section,
                "score": r.score,
                "text": r.passage_text[:500],
            })
        click.echo(json.dumps(results, indent=2))
        return

    click.echo(f"\nFound {len(response.results)} results for: {query}")
    click.echo(f"Search time: {response.search_time_ms:.1f}ms\n")

    for i, r in enumerate(response.results):
        click.echo(click.style(f"[{i+1}] {r.title}", fg="green", bold=True))
        if r.year:
            click.echo(f"    Year: {r.year}")
        if r.doi:
            click.echo(f"    DOI: {r.doi}")
        if r.section:
            click.echo(f"    Section: {r.section}")
        click.echo(f"    Score: {r.score:.4f}")
        click.echo(f"    {r.passage_text[:300]}...")
        click.echo()


@cli.command()
@click.argument("query")
@click.option("-n", "--num-passages", default=10, help="Number of passages")
def ask(query: str, num_passages: int):
    """Ask a question and get a synthesized answer."""
    from lib.search.jit_retrieval import JITRetriever

    click.echo(f"\nSearching for: {query}\n")

    retriever = JITRetriever()
    result = retriever.retrieve(query, n_passages=num_passages)

    if result.synthesis:
        click.echo(click.style("Answer:", fg="green", bold=True))
        click.echo(result.synthesis)
        click.echo()

    click.echo(click.style(f"Sources ({result.sources_used} passages):", fg="blue"))
    for i, p in enumerate(result.passages[:5]):
        click.echo(f"  [{i+1}] {p.title}")


# ============================================================================
# Ingestion Commands
# ============================================================================

@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--no-concepts", is_flag=True, help="Skip concept extraction")
@click.option("--no-embeddings", is_flag=True, help="Skip embedding generation")
def ingest(pdf_path: str, no_concepts: bool, no_embeddings: bool):
    """Ingest a PDF into the knowledge base."""
    from lib.ingest.pipeline import IngestPipeline

    path = Path(pdf_path)
    click.echo(f"\nIngesting: {path.name}")

    pipeline = IngestPipeline(
        extract_concepts=not no_concepts,
        compute_embeddings=not no_embeddings,
    )

    result = pipeline.ingest_pdf(path)

    if result.success:
        click.echo(click.style("✓ Ingestion successful!", fg="green"))
        click.echo(f"  Title: {result.title}")
        click.echo(f"  Doc ID: {result.doc_id}")
        click.echo(f"  Passages: {result.passage_count}")
        click.echo(f"  Concepts: {result.concept_count}")
        click.echo(f"  Metadata source: {result.metadata_source}")
        click.echo(f"  Time: {result.elapsed_seconds:.1f}s")
    else:
        click.echo(click.style("✗ Ingestion failed!", fg="red"))
        click.echo(f"  Error: {result.error}")
        sys.exit(1)


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--workers", default=4, help="Number of parallel workers")
@click.option("--dry-run", is_flag=True, help="List files without ingesting")
def ingest_batch(directory: str, workers: int, dry_run: bool):
    """Ingest all PDFs in a directory."""
    from lib.ingest.pipeline import IngestPipeline

    dir_path = Path(directory)
    pdf_files = list(dir_path.glob("**/*.pdf"))

    click.echo(f"\nFound {len(pdf_files)} PDF files in {directory}")

    if dry_run:
        for f in pdf_files[:20]:
            click.echo(f"  {f.name}")
        if len(pdf_files) > 20:
            click.echo(f"  ... and {len(pdf_files) - 20} more")
        return

    pipeline = IngestPipeline()

    def progress_callback(current, total, result):
        status = "✓" if result.success else "✗"
        click.echo(f"[{current}/{total}] {status} {result.title or 'Unknown'}")

    result = pipeline.ingest_batch(
        pdf_files,
        max_workers=workers,
        progress_callback=progress_callback,
    )

    click.echo(f"\n{'='*50}")
    click.echo(f"Total: {result.total}")
    click.echo(f"Succeeded: {result.succeeded}")
    click.echo(f"Failed: {result.failed}")
    click.echo(f"Time: {result.elapsed_seconds:.1f}s")


# ============================================================================
# Validation Commands
# ============================================================================

@cli.command()
@click.argument("claim")
def verify(claim: str):
    """Verify a factual claim against the knowledge base."""
    from lib.validation.hallucination import verify_claim

    click.echo(f"\nVerifying: {claim}\n")

    result = verify_claim(claim)

    status_color = {
        "supported": "green",
        "contradicted": "red",
        "unverifiable": "yellow",
    }.get(result.status.value, "white")

    click.echo(click.style(f"Verdict: {result.status.value.upper()}", fg=status_color, bold=True))
    click.echo(f"Confidence: {result.confidence:.2f}")

    if result.reasoning:
        click.echo(f"\nReasoning: {result.reasoning}")

    if result.evidence:
        click.echo("\nEvidence:")
        for i, e in enumerate(result.evidence[:3]):
            click.echo(f"  [{i+1}] {e.title}")


@cli.command()
@click.argument("text")
def check_hallucinations(text: str):
    """Check text for potential hallucinations."""
    from lib.validation.hallucination import detect_hallucinations

    click.echo(f"\nChecking text for hallucinations...\n")

    report = detect_hallucinations(text)

    score_color = "green" if report.is_reliable else "red"
    click.echo(click.style(f"Hallucination Score: {report.hallucination_score:.2f}", fg=score_color, bold=True))
    click.echo(f"Reliable: {'Yes' if report.is_reliable else 'No'}")

    click.echo(f"\nClaims analyzed: {len(report.claims)}")
    click.echo(f"  Supported: {report.supported_count}")
    click.echo(f"  Contradicted: {report.contradicted_count}")
    click.echo(f"  Unverifiable: {report.unverifiable_count}")

    if report.contradicted_count > 0:
        click.echo("\nContradicted claims:")
        for r in report.results:
            if r.status.value == "contradicted":
                click.echo(click.style(f"  ✗ {r.claim.text}", fg="red"))


# ============================================================================
# BridgeMine Commands
# ============================================================================

@cli.command()
@click.argument("domain")
@click.option("-n", "--limit", default=20, help="Maximum gaps to return")
@click.option("--check-novelty/--no-check-novelty", default=True, help="Check novelty via PubMed/S2")
def gaps(domain: str, limit: int, check_novelty: bool):
    """Find research gaps in a domain."""
    from lib.bridgemine.gap_detection import GapDetector
    from lib.bridgemine.novelty_check import NoveltyChecker

    click.echo(f"\nFinding research gaps in: {domain}\n")

    detector = GapDetector()
    result = detector.find_gaps(domain, limit=limit)

    if not result.candidates:
        click.echo("No gaps found.")
        return

    click.echo(f"Analyzed {result.methods_analyzed} methods, {result.problems_analyzed} problems")
    click.echo(f"Found {len(result.candidates)} candidates\n")

    candidates = result.candidates

    if check_novelty:
        click.echo("Checking novelty...")
        checker = NoveltyChecker()
        novelty_results = checker.check_batch(candidates[:10], delay=0.3)

        for i, nr in enumerate(novelty_results):
            c = nr.candidate
            novel_indicator = click.style("NOVEL", fg="green") if nr.is_novel else click.style("KNOWN", fg="yellow")

            click.echo(f"\n{i+1}. {click.style(c.method_name, bold=True)} [{novel_indicator}]")
            click.echo(f"   Transfer: {c.source_problem} → {c.target_problem}")
            click.echo(f"   Similarity: {c.problem_similarity:.2f}, Penetration: {c.domain_penetration:.1%}")
            click.echo(f"   Novelty: {nr.novelty_score:.2f}, Prior art: {nr.pubmed_hits}+{nr.semantic_scholar_hits}")
    else:
        for i, c in enumerate(candidates[:20]):
            click.echo(f"\n{i+1}. {click.style(c.method_name, bold=True)}")
            click.echo(f"   Transfer: {c.source_problem} → {c.target_problem}")
            click.echo(f"   Similarity: {c.problem_similarity:.2f}, Penetration: {c.domain_penetration:.1%}")


# ============================================================================
# Stats & Admin Commands
# ============================================================================

@cli.command()
def stats():
    """Show knowledge base statistics."""
    from lib.db.postgres import get_pg_pool

    pool = get_pg_pool()

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM documents")
            doc_count = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(*) as count FROM passages")
            passage_count = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(*) as count FROM passage_concepts")
            concept_count = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(*) as count FROM passages WHERE embedding IS NOT NULL")
            embedded_count = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(DISTINCT concept_name) as count FROM passage_concepts")
            unique_concepts = cur.fetchone()["count"]

    click.echo("\nPolymath v3 Statistics")
    click.echo("=" * 40)
    click.echo(f"Documents:         {doc_count:>15,}")
    click.echo(f"Passages:          {passage_count:>15,}")
    click.echo(f"Embedded:          {embedded_count:>15,}")
    click.echo(f"Concept mentions:  {concept_count:>15,}")
    click.echo(f"Unique concepts:   {unique_concepts:>15,}")


@cli.command()
@click.argument("cypher")
@click.option("--params", type=str, help="Query parameters as JSON")
def graph(cypher: str, params: str):
    """Execute a Cypher query against Neo4j."""
    from lib.db.neo4j import get_neo4j_driver

    query_params = json.loads(params) if params else {}

    driver = get_neo4j_driver()
    records, summary, _ = driver.execute_query(cypher, **query_params)

    click.echo(f"\nQuery: {cypher}")
    click.echo(f"Results: {len(records)}\n")

    for record in records[:20]:
        row = {}
        for key in record.keys():
            value = record[key]
            if hasattr(value, "_properties"):
                row[key] = dict(value._properties)
            else:
                row[key] = value
        click.echo(json.dumps(row, indent=2, default=str))


@cli.command()
def init_db():
    """Initialize database schemas."""
    from pathlib import Path
    from lib.db.postgres import get_pg_pool

    schema_dir = Path(__file__).parent.parent / "schema" / "postgres"
    pool = get_pg_pool()

    click.echo("\nInitializing Postgres schema...")

    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Execute schema files in order
            for sql_file in sorted(schema_dir.glob("*.sql")):
                click.echo(f"  Executing {sql_file.name}...")
                cur.execute(sql_file.read_text())

        conn.commit()

    click.echo(click.style("✓ Postgres schema initialized", fg="green"))

    # Neo4j
    click.echo("\nInitializing Neo4j schema...")

    try:
        from lib.db.neo4j import get_neo4j_driver

        driver = get_neo4j_driver()
        neo4j_schema = Path(__file__).parent.parent / "schema" / "neo4j" / "constraints.cypher"

        for statement in neo4j_schema.read_text().split(";"):
            statement = statement.strip()
            if statement:
                driver.execute_query(statement)

        click.echo(click.style("✓ Neo4j schema initialized", fg="green"))

    except Exception as e:
        click.echo(click.style(f"✗ Neo4j initialization failed: {e}", fg="red"))


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
