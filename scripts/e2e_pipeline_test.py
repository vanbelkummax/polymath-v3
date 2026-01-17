#!/usr/bin/env python3
"""
End-to-End Pipeline Test for Polymath System.

Tests the complete pipeline from PDF ingestion to skill extraction on a batch of papers.

Usage:
    # Full pipeline test
    python scripts/e2e_pipeline_test.py --run-all

    # Individual steps
    python scripts/e2e_pipeline_test.py --step ingest
    python scripts/e2e_pipeline_test.py --step concepts
    python scripts/e2e_pipeline_test.py --step assets
    python scripts/e2e_pipeline_test.py --step skills
    python scripts/e2e_pipeline_test.py --step citations
    python scripts/e2e_pipeline_test.py --step neo4j
    python scripts/e2e_pipeline_test.py --step registry

    # Generate report only
    python scripts/e2e_pipeline_test.py --report
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg2

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directories
TEST_BATCH_DIR = Path("/home/user/work/e2e_test_batch")
PDF_SOURCE_DIRS = [
    Path("/home/user/work/polymax/ingest_staging"),
    Path("/home/user/pdfs"),
    Path("/home/user/ken_lau_papers"),
]
SKILLS_DRAFTS_DIR = Path.home() / ".claude" / "skills_drafts"
REPO_ROOT = Path(__file__).parent.parent


class E2EPipelineTest:
    """Orchestrates end-to-end pipeline testing."""

    def __init__(self, conn=None, batch_name: str = None):
        self.batch_name = batch_name or f"e2e_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._conn = conn
        self.results = {
            "batch_name": self.batch_name,
            "started_at": datetime.now().isoformat(),
            "steps": {},
            "metrics": {},
            "errors": []
        }

    @property
    def conn(self):
        if self._conn is None:
            dsn = os.environ.get('POSTGRES_DSN', 'dbname=polymath user=polymath host=/var/run/postgresql')
            self._conn = psycopg2.connect(dsn)
        return self._conn

    # =========================================================================
    # Paper Selection
    # =========================================================================

    def select_test_papers(self, count: int = 20) -> List[Path]:
        """
        Select papers for testing based on criteria.

        Prioritizes:
        1. Papers with "spatial", "transcriptomics", "single-cell" in filename
        2. Papers with "transformer", "attention", "neural" in filename
        3. Papers with "segmentation", "detection" in filename
        4. Random selection to fill remaining slots
        """
        logger.info(f"Selecting {count} papers for testing...")

        all_pdfs = []
        for source_dir in PDF_SOURCE_DIRS:
            if source_dir.exists():
                all_pdfs.extend(source_dir.glob("*.pdf"))

        logger.info(f"Found {len(all_pdfs)} total PDFs")

        # Categorize by keywords
        spatial_papers = []
        dl_papers = []
        seg_papers = []
        other_papers = []

        spatial_keywords = ['spatial', 'transcriptom', 'single-cell', 'scanpy', 'squidpy', 'visium']
        dl_keywords = ['transformer', 'attention', 'neural', 'deep', 'learning', 'vision', 'dino', 'clip']
        seg_keywords = ['segment', 'detection', 'cellpose', 'stardist', 'nuclei']

        for pdf in all_pdfs:
            name_lower = pdf.name.lower()
            if any(kw in name_lower for kw in spatial_keywords):
                spatial_papers.append(pdf)
            elif any(kw in name_lower for kw in dl_keywords):
                dl_papers.append(pdf)
            elif any(kw in name_lower for kw in seg_keywords):
                seg_papers.append(pdf)
            else:
                other_papers.append(pdf)

        # Select balanced mix
        selected = []
        target_per_category = count // 4

        for category, papers in [
            ("spatial", spatial_papers),
            ("deep_learning", dl_papers),
            ("segmentation", seg_papers),
            ("other", other_papers)
        ]:
            available = min(len(papers), target_per_category)
            if papers:
                import random
                random.seed(42)  # Reproducible
                selected.extend(random.sample(papers, available))
                logger.info(f"  {category}: selected {available} papers")

        # Fill remaining with random
        remaining = count - len(selected)
        if remaining > 0:
            pool = [p for p in all_pdfs if p not in selected]
            if pool:
                import random
                selected.extend(random.sample(pool, min(remaining, len(pool))))

        logger.info(f"Selected {len(selected)} papers total")
        return selected[:count]

    def stage_papers(self, papers: List[Path]) -> Path:
        """Copy selected papers to test batch directory."""
        batch_dir = TEST_BATCH_DIR / self.batch_name / "pdfs"
        batch_dir.mkdir(parents=True, exist_ok=True)

        for pdf in papers:
            dest = batch_dir / pdf.name
            if not dest.exists():
                shutil.copy2(pdf, dest)
                logger.info(f"  Staged: {pdf.name}")

        self.results["steps"]["staging"] = {
            "status": "complete",
            "papers_staged": len(papers),
            "batch_dir": str(batch_dir)
        }
        return batch_dir

    # =========================================================================
    # Step 1: Ingestion
    # =========================================================================

    def run_ingestion(self, pdf_dir: Path) -> List[str]:
        """Run unified ingest on all PDFs."""
        logger.info("Step 1: Ingesting PDFs...")

        doc_ids = []
        errors = []

        for pdf in sorted(pdf_dir.glob("*.pdf")):
            try:
                # Use unified_ingest.py
                result = subprocess.run(
                    [sys.executable, "lib/ingest/unified_ingest.py", str(pdf), "--enhanced-parser"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(REPO_ROOT)
                )

                if result.returncode == 0:
                    # Extract doc_id from output
                    for line in result.stdout.split('\n'):
                        if 'doc_id' in line.lower():
                            # Try to extract UUID
                            import re
                            match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', line)
                            if match:
                                doc_ids.append(match.group())
                    logger.info(f"  ✓ Ingested: {pdf.name}")
                else:
                    errors.append(f"{pdf.name}: {result.stderr[:200]}")
                    logger.warning(f"  ✗ Failed: {pdf.name}")

            except subprocess.TimeoutExpired:
                errors.append(f"{pdf.name}: Timeout")
            except Exception as e:
                errors.append(f"{pdf.name}: {str(e)}")

        self.results["steps"]["ingestion"] = {
            "status": "complete" if len(doc_ids) > 0 else "failed",
            "docs_ingested": len(doc_ids),
            "errors": errors[:10]  # First 10 errors
        }

        if errors:
            self.results["errors"].extend(errors)

        return doc_ids

    # =========================================================================
    # Step 2: Concept Extraction (Batch API)
    # =========================================================================

    def submit_concept_batch(self, doc_ids: List[str] = None) -> Optional[str]:
        """Submit batch job for concept extraction."""
        logger.info("Step 2: Submitting concept extraction batch job...")

        try:
            # Get passages without concepts from recent docs
            cur = self.conn.cursor()

            if doc_ids:
                placeholders = ','.join(['%s'] * len(doc_ids))
                cur.execute(f"""
                    SELECT p.passage_id
                    FROM passages p
                    LEFT JOIN passage_concepts pc ON p.passage_id = pc.passage_id
                    WHERE p.doc_id IN ({placeholders})
                    AND pc.passage_id IS NULL
                    LIMIT 2000
                """, doc_ids)
            else:
                cur.execute("""
                    SELECT p.passage_id
                    FROM passages p
                    JOIN documents d ON p.doc_id = d.doc_id
                    LEFT JOIN passage_concepts pc ON p.passage_id = pc.passage_id
                    WHERE d.created_at > NOW() - INTERVAL '1 day'
                    AND pc.passage_id IS NULL
                    LIMIT 2000
                """)

            passage_ids = [row[0] for row in cur.fetchall()]
            logger.info(f"  Found {len(passage_ids)} passages needing concepts")

            if not passage_ids:
                self.results["steps"]["concepts"] = {
                    "status": "skipped",
                    "reason": "No passages need concept extraction"
                }
                return None

            # Submit batch job
            result = subprocess.run(
                [sys.executable, "/home/user/polymath-repo/scripts/batch_concept_extraction_async.py",
                 "--limit", str(len(passage_ids))],
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "POSTGRES_DSN": "dbname=polymath user=polymath host=/var/run/postgresql"}
            )

            # Extract job ID
            job_id = None
            for line in result.stdout.split('\n'):
                if 'job' in line.lower() and any(c.isdigit() for c in line):
                    import re
                    match = re.search(r'\d{10,}', line)
                    if match:
                        job_id = match.group()

            self.results["steps"]["concepts"] = {
                "status": "submitted",
                "job_id": job_id,
                "passages_submitted": len(passage_ids)
            }

            return job_id

        except Exception as e:
            self.results["steps"]["concepts"] = {
                "status": "error",
                "error": str(e)
            }
            self.results["errors"].append(f"Concept extraction: {str(e)}")
            return None

    # =========================================================================
    # Step 3: Asset Detection
    # =========================================================================

    def run_asset_detection(self, doc_ids: List[str] = None):
        """Detect GitHub repos and HuggingFace models in papers."""
        logger.info("Step 3: Running asset detection...")

        try:
            from lib.ingest.asset_detector import AssetDetector

            detector = AssetDetector(self.conn)
            cur = self.conn.cursor()

            # Get recent documents
            if doc_ids:
                docs = [(did,) for did in doc_ids]
            else:
                cur.execute("""
                    SELECT doc_id FROM documents
                    WHERE created_at > NOW() - INTERVAL '1 day'
                """)
                docs = cur.fetchall()

            repos_found = 0
            models_found = 0

            for (doc_id,) in docs:
                result = detector.detect_and_store(doc_id)
                repos_found += result.get('repos', 0)
                models_found += result.get('models', 0)

            self.results["steps"]["assets"] = {
                "status": "complete",
                "docs_scanned": len(docs),
                "repos_found": repos_found,
                "models_found": models_found
            }

        except Exception as e:
            self.results["steps"]["assets"] = {
                "status": "error",
                "error": str(e)
            }
            self.results["errors"].append(f"Asset detection: {str(e)}")

    # =========================================================================
    # Step 4: Skill Extraction
    # =========================================================================

    async def run_skill_extraction(self, doc_ids: List[str] = None):
        """Extract skills from method-heavy passages."""
        logger.info("Step 4: Running skill extraction...")

        try:
            from lib.ingest.skill_extractor import SkillExtractor

            extractor = SkillExtractor(self.conn)
            cur = self.conn.cursor()

            # Get documents with method concepts
            if doc_ids:
                placeholders = ','.join(['%s'] * len(doc_ids))
                cur.execute(f"""
                    SELECT DISTINCT d.doc_id, d.title
                    FROM documents d
                    JOIN passages p ON d.doc_id = p.doc_id
                    JOIN passage_concepts pc ON p.passage_id = pc.passage_id
                    WHERE d.doc_id IN ({placeholders})
                    AND pc.concept_type IN ('method', 'algorithm', 'workflow', 'mechanism')
                """, doc_ids)
            else:
                cur.execute("""
                    SELECT DISTINCT d.doc_id, d.title
                    FROM documents d
                    JOIN passages p ON d.doc_id = p.doc_id
                    JOIN passage_concepts pc ON p.passage_id = pc.passage_id
                    WHERE d.created_at > NOW() - INTERVAL '1 day'
                    AND pc.concept_type IN ('method', 'algorithm', 'workflow', 'mechanism')
                """)

            docs_with_methods = cur.fetchall()
            skills_created = 0

            for doc_id, title in docs_with_methods:
                # Get passages
                cur.execute("SELECT passage_id, passage_text FROM passages WHERE doc_id = %s", (doc_id,))
                passages = [{'passage_id': str(r[0]), 'passage_text': r[1]} for r in cur.fetchall()]

                # Get concepts
                cur.execute("""
                    SELECT pc.passage_id, pc.concept_name, pc.concept_type
                    FROM passage_concepts pc
                    JOIN passages p ON pc.passage_id = p.passage_id
                    WHERE p.doc_id = %s
                """, (doc_id,))
                concepts = [{'passage_id': str(r[0]), 'concept_name': r[1], 'concept_type': r[2]}
                           for r in cur.fetchall()]

                # Extract and write to drafts
                try:
                    paths = await extractor.extract_and_write_drafts(
                        str(doc_id), passages, concepts, title or "Unknown"
                    )
                    skills_created += len(paths)
                    logger.info(f"  Extracted {len(paths)} skills from: {title[:50]}...")
                except Exception as e:
                    logger.warning(f"  Skill extraction failed for {doc_id}: {e}")

            self.results["steps"]["skills"] = {
                "status": "complete",
                "docs_processed": len(docs_with_methods),
                "skills_created": skills_created
            }

        except Exception as e:
            self.results["steps"]["skills"] = {
                "status": "error",
                "error": str(e)
            }
            self.results["errors"].append(f"Skill extraction: {str(e)}")

    # =========================================================================
    # Step 5: Citation Network
    # =========================================================================

    def build_citation_network(self, doc_ids: List[str] = None):
        """Extract and link citations between papers."""
        logger.info("Step 5: Building citation network...")

        try:
            cur = self.conn.cursor()

            # Check if citation_links table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'citation_links'
                )
            """)
            table_exists = cur.fetchone()[0]

            if not table_exists:
                logger.info("  Creating citation_links table...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS citation_links (
                        link_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        citing_doc_id UUID REFERENCES documents(doc_id),
                        cited_doi TEXT,
                        cited_pmid TEXT,
                        cited_doc_id UUID REFERENCES documents(doc_id),
                        context TEXT,
                        created_at TIMESTAMPTZ DEFAULT now()
                    )
                """)
                self.conn.commit()

            # Extract DOIs from passages
            if doc_ids:
                placeholders = ','.join(['%s'] * len(doc_ids))
                cur.execute(f"""
                    SELECT p.doc_id, p.passage_text
                    FROM passages p
                    WHERE p.doc_id IN ({placeholders})
                    AND p.passage_text ~ '10\\.\\d{{4,}}/[^\\s]+'
                """, doc_ids)
            else:
                cur.execute("""
                    SELECT p.doc_id, p.passage_text
                    FROM passages p
                    JOIN documents d ON p.doc_id = d.doc_id
                    WHERE d.created_at > NOW() - INTERVAL '1 day'
                    AND p.passage_text ~ '10\\.\\d{4,}/[^\\s]+'
                """)

            passages_with_dois = cur.fetchall()
            import re
            doi_pattern = r'10\.\d{4,}/[^\s\])\'">,]+'

            citations_found = 0
            for doc_id, text in passages_with_dois:
                dois = re.findall(doi_pattern, text)
                for doi in set(dois):
                    doi = doi.rstrip('.')  # Clean trailing periods
                    try:
                        cur.execute("""
                            INSERT INTO citation_links (citing_doc_id, cited_doi, context)
                            VALUES (%s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (doc_id, doi, text[:500]))
                        citations_found += 1
                    except:
                        pass

            self.conn.commit()

            self.results["steps"]["citations"] = {
                "status": "complete",
                "passages_scanned": len(passages_with_dois),
                "citations_found": citations_found
            }

        except Exception as e:
            self.results["steps"]["citations"] = {
                "status": "error",
                "error": str(e)
            }
            self.results["errors"].append(f"Citation network: {str(e)}")

    # =========================================================================
    # Step 6: Neo4j Sync
    # =========================================================================

    def sync_to_neo4j(self):
        """Sync all data to Neo4j knowledge graph."""
        logger.info("Step 6: Syncing to Neo4j...")

        try:
            result = subprocess.run(
                [sys.executable, "scripts/sync_neo4j.py", "--full"],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(REPO_ROOT)
            )

            if result.returncode == 0:
                self.results["steps"]["neo4j"] = {
                    "status": "complete",
                    "output": result.stdout[-500:] if result.stdout else ""
                }
            else:
                self.results["steps"]["neo4j"] = {
                    "status": "partial",
                    "error": result.stderr[:500] if result.stderr else "Unknown error"
                }

        except subprocess.TimeoutExpired:
            self.results["steps"]["neo4j"] = {
                "status": "timeout",
                "error": "Neo4j sync timed out (>10 min)"
            }
        except Exception as e:
            self.results["steps"]["neo4j"] = {
                "status": "error",
                "error": str(e)
            }

    # =========================================================================
    # Step 7: Generate Asset Registry
    # =========================================================================

    def generate_asset_registry(self):
        """Generate central asset registry document."""
        logger.info("Step 7: Generating asset registry...")

        try:
            cur = self.conn.cursor()

            # Get GitHub repos
            cur.execute("""
                SELECT repo_url, repo_owner, repo_name, source_doc_count, status
                FROM repo_queue
                ORDER BY source_doc_count DESC
                LIMIT 50
            """)
            repos = cur.fetchall()

            # Get HuggingFace models
            cur.execute("""
                SELECT model_id, organization, model_name, pipeline_tag, downloads_30d, citation_count
                FROM hf_models
                ORDER BY citation_count DESC NULLS LAST
                LIMIT 50
            """)
            models = cur.fetchall()

            # Get extracted skills
            cur.execute("""
                SELECT skill_name, skill_type, original_domain, status, confidence
                FROM paper_skills
                WHERE created_at > NOW() - INTERVAL '7 days'
                ORDER BY confidence DESC NULLS LAST
                LIMIT 30
            """)
            skills = cur.fetchall()

            # Generate markdown
            registry_content = f"""# Polymath Asset Registry

> Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## GitHub Repositories ({len(repos)} tracked)

| Repository | Papers | Status |
|------------|--------|--------|
"""
            for repo_url, owner, name, count, status in repos[:20]:
                registry_content += f"| [{owner}/{name}]({repo_url}) | {count} | {status} |\n"

            registry_content += f"""

## HuggingFace Models ({len(models)} tracked)

| Model | Type | Downloads | Citations |
|-------|------|-----------|-----------|
"""
            for model_id, org, name, pipeline, downloads, citations in models[:20]:
                registry_content += f"| `{model_id}` | {pipeline or 'unknown'} | {downloads or 'N/A'} | {citations or 0} |\n"

            registry_content += f"""

## Extracted Skills ({len(skills)} recent)

| Skill | Type | Domain | Status | Confidence |
|-------|------|--------|--------|------------|
"""
            for skill_name, skill_type, domain, status, conf in skills:
                registry_content += f"| `{skill_name}` | {skill_type} | {domain or 'unknown'} | {status} | {conf:.2f if conf else 'N/A'} |\n"

            registry_content += """

## Quick Access

### Promoting Skills
```bash
python scripts/promote_skill.py --list
python scripts/promote_skill.py <skill-name> --bootstrap
```

### Checking Status
```bash
psql -U polymath -d polymath -c "SELECT * FROM v_skill_usage_summary;"
psql -U polymath -d polymath -c "SELECT * FROM v_unresolved_hf_models;"
```
"""

            # Write registry
            registry_path = REPO_ROOT / "ASSET_REGISTRY.md"
            registry_path.write_text(registry_content)

            self.results["steps"]["registry"] = {
                "status": "complete",
                "repos_listed": len(repos),
                "models_listed": len(models),
                "skills_listed": len(skills),
                "path": str(registry_path)
            }

        except Exception as e:
            self.results["steps"]["registry"] = {
                "status": "error",
                "error": str(e)
            }
            self.results["errors"].append(f"Asset registry: {str(e)}")

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_report(self) -> str:
        """Generate test report."""
        self.results["completed_at"] = datetime.now().isoformat()

        # Collect metrics
        cur = self.conn.cursor()

        cur.execute("SELECT COUNT(*) FROM documents WHERE created_at > NOW() - INTERVAL '1 day'")
        self.results["metrics"]["docs_today"] = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM passages WHERE doc_id IN (SELECT doc_id FROM documents WHERE created_at > NOW() - INTERVAL '1 day')")
        self.results["metrics"]["passages_today"] = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM passage_concepts WHERE passage_id IN (SELECT passage_id FROM passages WHERE doc_id IN (SELECT doc_id FROM documents WHERE created_at > NOW() - INTERVAL '1 day'))")
        self.results["metrics"]["concepts_today"] = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM repo_queue")
        self.results["metrics"]["total_repos"] = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM hf_models")
        self.results["metrics"]["total_models"] = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM paper_skills WHERE status = 'draft'")
        self.results["metrics"]["draft_skills"] = cur.fetchone()[0]

        # Count skill drafts on filesystem
        if SKILLS_DRAFTS_DIR.exists():
            self.results["metrics"]["skills_in_drafts_dir"] = len(list(SKILLS_DRAFTS_DIR.iterdir()))

        # Generate report
        report = f"""# E2E Pipeline Test Report

**Batch:** {self.results['batch_name']}
**Started:** {self.results['started_at']}
**Completed:** {self.results.get('completed_at', 'In progress')}

## Summary Metrics

| Metric | Value |
|--------|-------|
| Documents (today) | {self.results['metrics'].get('docs_today', 'N/A')} |
| Passages (today) | {self.results['metrics'].get('passages_today', 'N/A')} |
| Concepts (today) | {self.results['metrics'].get('concepts_today', 'N/A')} |
| Total Repos | {self.results['metrics'].get('total_repos', 'N/A')} |
| Total HF Models | {self.results['metrics'].get('total_models', 'N/A')} |
| Draft Skills (DB) | {self.results['metrics'].get('draft_skills', 'N/A')} |
| Skills in Drafts Dir | {self.results['metrics'].get('skills_in_drafts_dir', 'N/A')} |

## Step Results

"""
        for step_name, step_result in self.results.get("steps", {}).items():
            status = step_result.get("status", "unknown")
            icon = "✓" if status == "complete" else "⚠" if status in ["partial", "submitted"] else "✗"
            report += f"### {icon} {step_name.title()}\n\n"
            for key, value in step_result.items():
                if key != "status":
                    report += f"- **{key}:** {value}\n"
            report += "\n"

        if self.results.get("errors"):
            report += "## Errors\n\n"
            for err in self.results["errors"][:10]:
                report += f"- {err}\n"

        report += """
## Next Steps

1. Check batch API job status: `python scripts/batch_concept_extraction_async.py --status`
2. Review extracted skills: `python scripts/promote_skill.py --check-all`
3. Verify Neo4j graph: `docker exec polymax-neo4j cypher-shell -u neo4j -p polymathic2026 "MATCH (n) RETURN labels(n), count(n)"`
4. Check asset registry: `cat ASSET_REGISTRY.md`
"""

        # Save report
        report_path = TEST_BATCH_DIR / self.batch_name / "E2E_TEST_REPORT.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)

        # Also save JSON results
        json_path = TEST_BATCH_DIR / self.batch_name / "results.json"
        json_path.write_text(json.dumps(self.results, indent=2, default=str))

        logger.info(f"Report saved to: {report_path}")
        return report

    # =========================================================================
    # Main Runner
    # =========================================================================

    async def run_all(self, paper_count: int = 20):
        """Run complete pipeline test."""
        logger.info(f"Starting E2E Pipeline Test: {self.batch_name}")
        logger.info("=" * 60)

        # Select and stage papers
        papers = self.select_test_papers(paper_count)
        pdf_dir = self.stage_papers(papers)

        # Run pipeline steps
        doc_ids = self.run_ingestion(pdf_dir)

        if doc_ids:
            self.submit_concept_batch(doc_ids)
            self.run_asset_detection(doc_ids)
            await self.run_skill_extraction(doc_ids)
            self.build_citation_network(doc_ids)

        self.sync_to_neo4j()
        self.generate_asset_registry()

        # Generate report
        report = self.generate_report()
        print("\n" + "=" * 60)
        print(report)

        return self.results


def main():
    parser = argparse.ArgumentParser(description='E2E Pipeline Test for Polymath')
    parser.add_argument('--run-all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--step', choices=['ingest', 'concepts', 'assets', 'skills', 'citations', 'neo4j', 'registry'],
                       help='Run specific step')
    parser.add_argument('--report', action='store_true', help='Generate report only')
    parser.add_argument('--papers', type=int, default=20, help='Number of papers to test')
    parser.add_argument('--batch-name', type=str, help='Custom batch name')

    args = parser.parse_args()

    tester = E2EPipelineTest(batch_name=args.batch_name)

    if args.run_all:
        asyncio.run(tester.run_all(args.papers))
    elif args.step:
        if args.step == 'registry':
            tester.generate_asset_registry()
        elif args.step == 'neo4j':
            tester.sync_to_neo4j()
        elif args.step == 'citations':
            tester.build_citation_network()
        elif args.step == 'skills':
            asyncio.run(tester.run_skill_extraction())
        elif args.step == 'assets':
            tester.run_asset_detection()
        elif args.step == 'concepts':
            tester.submit_concept_batch()
        elif args.step == 'ingest':
            papers = tester.select_test_papers(args.papers)
            pdf_dir = tester.stage_papers(papers)
            tester.run_ingestion(pdf_dir)
        tester.generate_report()
    elif args.report:
        print(tester.generate_report())
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
