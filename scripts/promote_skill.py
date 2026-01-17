#!/usr/bin/env python3
"""
Skill Promotion Script for Polymath System.

Promotes skills from ~/.claude/skills_drafts/ to ~/.claude/skills/
after validating through 4 gates:

    Gate 1: Evidence    - ≥2 source passages OR 1 passage + 1 code link
    Gate 2: Oracle      - Runnable test exists and passes
    Gate 3: Dedup       - Not similar to existing promoted skill (cosine < 0.85)
    Gate 4: Usage       - At least 1 logged successful use (bootstrap-skippable)

Usage:
    python scripts/promote_skill.py <skill-name>
    python scripts/promote_skill.py <skill-name> --bootstrap  # Skip usage gate
    python scripts/promote_skill.py --list                    # List draft skills
    python scripts/promote_skill.py --check-all               # Check all drafts

Author: Polymath System
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging

import psycopg2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths
DRAFTS_DIR = Path.home() / ".claude" / "skills_drafts"
SKILLS_DIR = Path.home() / ".claude" / "skills"
TEMPLATE_PATH = Path.home() / ".claude" / "SKILL_TEMPLATE.md"

# Gate thresholds
MIN_EVIDENCE_PASSAGES = 2
SIMILARITY_THRESHOLD = 0.85


@dataclass
class GateResult:
    """Result of a gate check."""
    passed: bool
    gate_name: str
    message: str
    details: Optional[Dict] = None


@dataclass
class PromotionResult:
    """Result of skill promotion attempt."""
    success: bool
    skill_name: str
    gates: List[GateResult]
    promoted_path: Optional[Path] = None
    error: Optional[str] = None


class SkillPromoter:
    """Promotes skills from drafts to production with gate validation."""

    def __init__(self, conn=None, bootstrap: bool = False):
        """
        Initialize promoter.

        Args:
            conn: Postgres connection (will create if None)
            bootstrap: If True, skip Gate 4 (usage requirement)
        """
        self.bootstrap = bootstrap
        self._conn = conn
        self._embedder = None

    @property
    def conn(self):
        """Lazy database connection."""
        if self._conn is None:
            dsn = os.environ.get('POSTGRES_DSN', 'dbname=polymath user=polymath host=/var/run/postgresql')
            self._conn = psycopg2.connect(dsn)
        return self._conn

    @property
    def embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from lib.embeddings.bge_m3 import Embedder
                self._embedder = Embedder()
            except ImportError:
                logger.warning("Could not load Embedder - dedup gate will be disabled")
        return self._embedder

    def list_drafts(self) -> List[str]:
        """List all draft skills."""
        if not DRAFTS_DIR.exists():
            return []
        drafts = []
        for d in DRAFTS_DIR.iterdir():
            if d.is_dir():
                # Check for SKILL.md or CANDIDATE.md
                if (d / "SKILL.md").exists() or (d / "CANDIDATE.md").exists():
                    drafts.append(d.name)
        return drafts

    def get_draft_path(self, skill_name: str) -> Optional[Path]:
        """Get path to draft skill directory."""
        draft_path = DRAFTS_DIR / skill_name
        if draft_path.exists() and ((draft_path / "SKILL.md").exists() or (draft_path / "CANDIDATE.md").exists()):
            return draft_path
        # Try kebab-case conversion
        kebab_name = skill_name.lower().replace(' ', '-').replace('_', '-')
        draft_path = DRAFTS_DIR / kebab_name
        if draft_path.exists() and ((draft_path / "SKILL.md").exists() or (draft_path / "CANDIDATE.md").exists()):
            return draft_path
        return None

    def get_skill_file(self, skill_path: Path) -> Optional[Path]:
        """Get the skill definition file (SKILL.md or CANDIDATE.md)."""
        skill_md = skill_path / "SKILL.md"
        if skill_md.exists():
            return skill_md
        candidate_md = skill_path / "CANDIDATE.md"
        if candidate_md.exists():
            return candidate_md
        return None

    def get_skill_from_db(self, skill_name: str) -> Optional[Dict]:
        """Get skill record from database by name."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT skill_id, skill_name, description, source_passages,
                   source_code_chunks, embedding, status
            FROM paper_skills
            WHERE skill_name = %s OR skill_name = %s
            LIMIT 1
        """, (skill_name, skill_name.replace('-', '_')))
        row = cur.fetchone()
        if row:
            return {
                'skill_id': str(row[0]),
                'skill_name': row[1],
                'description': row[2],
                'source_passages': row[3],
                'source_code_chunks': row[4],
                'embedding': row[5],
                'status': row[6]
            }
        return None

    def parse_skill_metadata(self, skill_path: Path) -> Dict:
        """Parse SKILL.md or CANDIDATE.md frontmatter and extract metadata."""
        skill_file = self.get_skill_file(skill_path)
        if not skill_file or not skill_file.exists():
            return {}

        content = skill_file.read_text()

        # Parse YAML frontmatter
        metadata = {}
        if content.startswith('---'):
            end = content.find('---', 3)
            if end > 0:
                frontmatter = content[3:end].strip()
                for line in frontmatter.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        # Parse arrays
                        if value.startswith('[') and value.endswith(']'):
                            value = [v.strip().strip('"\'') for v in value[1:-1].split(',')]
                        metadata[key] = value

        # Extract Oracle section
        oracle_match = re.search(r'## Oracle.*?```python\n(.*?)```', content, re.DOTALL)
        if oracle_match:
            metadata['oracle_code'] = oracle_match.group(1)

        # Extract Provenance/Passages section
        passages_match = re.search(r'### Passages\n(.*?)(?=\n##|\n---|\Z)', content, re.DOTALL)
        if passages_match:
            passage_ids = re.findall(r'passage_id[:\s]+(\S+)', passages_match.group(1))
            metadata['passage_ids'] = passage_ids

        return metadata

    # =========================================================================
    # GATE 1: Evidence
    # =========================================================================

    def check_evidence(self, skill_name: str, skill_path: Path) -> GateResult:
        """
        Gate 1: Check evidence requirements.

        Requires: ≥2 source passages OR 1 passage + 1 code link
        """
        # Check database record
        db_skill = self.get_skill_from_db(skill_name)
        metadata = self.parse_skill_metadata(skill_path)

        passage_count = 0
        code_link_count = 0

        # Count passages from database
        if db_skill and db_skill.get('source_passages'):
            passages = db_skill['source_passages']
            if isinstance(passages, str):
                try:
                    passages = json.loads(passages)
                except:
                    passages = []
            if isinstance(passages, list):
                passage_count = len(passages)

        # Count passages from SKILL.md
        if 'passage_ids' in metadata:
            passage_count = max(passage_count, len(metadata['passage_ids']))

        # Check evidence.json if exists
        evidence_file = skill_path / "evidence.json"
        if evidence_file.exists():
            try:
                evidence = json.loads(evidence_file.read_text())
                passage_count = max(passage_count, len(evidence.get('passages', [])))
                code_link_count = len(evidence.get('code_links', []))
            except:
                pass

        # Count code links from database
        if db_skill and db_skill.get('source_code_chunks'):
            code_link_count = max(code_link_count, len(db_skill['source_code_chunks']))

        # Check Provenance/Code section in skill file
        skill_file = self.get_skill_file(skill_path)
        skill_content = skill_file.read_text() if skill_file else ""
        code_section_match = re.search(r'### Code\n(.*?)(?=\n###|\n##|\Z)', skill_content, re.DOTALL)
        if code_section_match:
            code_refs = re.findall(r'`[^`]+\.py:[^`]+`|github\.com/\S+', code_section_match.group(1))
            code_link_count = max(code_link_count, len(code_refs))

        # Apply gate logic
        passed = (passage_count >= MIN_EVIDENCE_PASSAGES) or (passage_count >= 1 and code_link_count >= 1)

        return GateResult(
            passed=passed,
            gate_name="Evidence",
            message=f"Found {passage_count} passage(s), {code_link_count} code link(s). "
                    f"Need ≥{MIN_EVIDENCE_PASSAGES} passages OR 1 passage + 1 code link.",
            details={
                'passage_count': passage_count,
                'code_link_count': code_link_count,
                'threshold': MIN_EVIDENCE_PASSAGES
            }
        )

    # =========================================================================
    # GATE 2: Oracle
    # =========================================================================

    def check_oracle(self, skill_name: str, skill_path: Path) -> GateResult:
        """
        Gate 2: Check that oracle test exists and passes.

        Requires: test_skill() function exists and passes when executed.
        """
        metadata = self.parse_skill_metadata(skill_path)

        # Check for oracle code
        oracle_code = metadata.get('oracle_code', '')

        # Also check for separate oracle.py file
        oracle_file = skill_path / "oracle.py"
        if oracle_file.exists():
            oracle_code = oracle_file.read_text()

        if not oracle_code or 'def test_' not in oracle_code:
            return GateResult(
                passed=False,
                gate_name="Oracle",
                message="No oracle test found. Skill must have a `def test_*()` function in Oracle section.",
                details={'has_oracle': False}
            )

        # Run the oracle test
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Add imports and test runner
                test_code = f"""
import sys
sys.path.insert(0, '{Path(__file__).parent.parent}')

{oracle_code}

if __name__ == "__main__":
    import re
    # Find and run all test functions
    test_funcs = [name for name in dir() if name.startswith('test_')]
    for func_name in test_funcs:
        func = globals()[func_name]
        if callable(func):
            try:
                func()
                print(f"✓ {{func_name}} passed")
            except Exception as e:
                print(f"✗ {{func_name}} failed: {{e}}")
                sys.exit(1)
    print("All oracle tests passed")
"""
                f.write(test_code)
                f.flush()
                temp_path = f.name

            # Run with timeout
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(skill_path)
            )

            os.unlink(temp_path)

            if result.returncode == 0:
                return GateResult(
                    passed=True,
                    gate_name="Oracle",
                    message="Oracle test passed.",
                    details={'output': result.stdout[:500]}
                )
            else:
                return GateResult(
                    passed=False,
                    gate_name="Oracle",
                    message=f"Oracle test failed: {result.stderr[:200]}",
                    details={'stdout': result.stdout[:500], 'stderr': result.stderr[:500]}
                )

        except subprocess.TimeoutExpired:
            return GateResult(
                passed=False,
                gate_name="Oracle",
                message="Oracle test timed out (>60s)",
                details={'timeout': True}
            )
        except Exception as e:
            return GateResult(
                passed=False,
                gate_name="Oracle",
                message=f"Oracle test error: {str(e)[:200]}",
                details={'error': str(e)}
            )

    # =========================================================================
    # GATE 3: Dedup
    # =========================================================================

    def check_dedup(self, skill_name: str, skill_path: Path) -> GateResult:
        """
        Gate 3: Check skill is not a near-duplicate of existing promoted skill.

        Requires: Cosine similarity < 0.85 to all promoted skills.
        """
        if self.embedder is None:
            return GateResult(
                passed=True,
                gate_name="Dedup",
                message="Embedder not available - skipping dedup check.",
                details={'skipped': True}
            )

        # Get skill description for embedding
        skill_file = self.get_skill_file(skill_path)
        if not skill_file:
            return GateResult(
                passed=True,
                gate_name="Dedup",
                message="No skill file found - skipping dedup check.",
                details={'skipped': True}
            )
        content = skill_file.read_text()

        # Extract description (after first ## heading)
        desc_match = re.search(r'## When to Use\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        description = desc_match.group(1)[:1000] if desc_match else content[:1000]

        # Also include skill name
        description = f"{skill_name}: {description}"

        # Generate embedding (use encode_query for 1D output)
        try:
            if hasattr(self.embedder, 'encode_query'):
                draft_embedding = self.embedder.encode_query(description)
            else:
                # Fallback: encode and flatten
                emb = self.embedder.encode(description)
                draft_embedding = emb[0] if len(emb.shape) > 1 else emb
        except Exception as e:
            return GateResult(
                passed=True,
                gate_name="Dedup",
                message=f"Could not generate embedding: {e}",
                details={'error': str(e)}
            )

        # Check against promoted skills in database
        cur = self.conn.cursor()
        cur.execute("""
            SELECT skill_id, skill_name, description,
                   1 - (embedding <=> %s::vector) as similarity
            FROM paper_skills
            WHERE status = 'promoted'
            AND embedding IS NOT NULL
            AND skill_name != %s
            ORDER BY similarity DESC
            LIMIT 5
        """, (draft_embedding.tolist(), skill_name))

        similar_skills = []
        max_similarity = 0.0
        most_similar = None

        for row in cur.fetchall():
            similarity = float(row[3]) if row[3] else 0.0
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar = row[1]
            if similarity > SIMILARITY_THRESHOLD:
                similar_skills.append({
                    'skill_id': str(row[0]),
                    'skill_name': row[1],
                    'similarity': similarity
                })

        # Also check against promoted skill files
        if SKILLS_DIR.exists():
            for promoted_dir in SKILLS_DIR.iterdir():
                if promoted_dir.is_dir() and promoted_dir.name != skill_name:
                    promoted_skill = promoted_dir / "SKILL.md"
                    if promoted_skill.exists():
                        promoted_content = promoted_skill.read_text()
                        p_desc_match = re.search(r'## When to Use\n(.*?)(?=\n##|\Z)', promoted_content, re.DOTALL)
                        p_description = p_desc_match.group(1)[:1000] if p_desc_match else promoted_content[:1000]
                        p_description = f"{promoted_dir.name}: {p_description}"

                        try:
                            if hasattr(self.embedder, 'encode_query'):
                                promoted_embedding = self.embedder.encode_query(p_description)
                            else:
                                emb = self.embedder.encode(p_description)
                                promoted_embedding = emb[0] if len(emb.shape) > 1 else emb
                            similarity = float(np.dot(draft_embedding, promoted_embedding) /
                                             (np.linalg.norm(draft_embedding) * np.linalg.norm(promoted_embedding)))

                            if similarity > max_similarity:
                                max_similarity = similarity
                                most_similar = promoted_dir.name

                            if similarity > SIMILARITY_THRESHOLD:
                                similar_skills.append({
                                    'skill_name': promoted_dir.name,
                                    'similarity': similarity,
                                    'source': 'filesystem'
                                })
                        except:
                            pass

        passed = len(similar_skills) == 0

        return GateResult(
            passed=passed,
            gate_name="Dedup",
            message=f"Max similarity: {max_similarity:.3f} (to '{most_similar}'). "
                    f"Threshold: <{SIMILARITY_THRESHOLD}. "
                    f"{'No duplicates found.' if passed else f'Found {len(similar_skills)} similar skill(s).'}",
            details={
                'max_similarity': max_similarity,
                'most_similar': most_similar,
                'similar_skills': similar_skills,
                'threshold': SIMILARITY_THRESHOLD
            }
        )

    # =========================================================================
    # GATE 4: Usage
    # =========================================================================

    def check_usage(self, skill_name: str, skill_path: Path) -> GateResult:
        """
        Gate 4: Check that skill has at least 1 successful usage logged.

        Can be skipped with --bootstrap flag for initial promotion.
        """
        if self.bootstrap:
            return GateResult(
                passed=True,
                gate_name="Usage",
                message="Skipped (bootstrap mode).",
                details={'bootstrap': True}
            )

        # Check if skill_usage_log table exists
        cur = self.conn.cursor()
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'skill_usage_log'
            )
        """)
        table_exists = cur.fetchone()[0]

        if not table_exists:
            # No usage table yet - pass with warning
            return GateResult(
                passed=True,
                gate_name="Usage",
                message="skill_usage_log table not found. Skipping usage check.",
                details={'table_missing': True}
            )

        # Count successful uses
        cur.execute("""
            SELECT COUNT(*)
            FROM skill_usage_log
            WHERE skill_name = %s
            AND outcome = 'success'
        """, (skill_name,))

        success_count = cur.fetchone()[0]

        passed = success_count >= 1

        return GateResult(
            passed=passed,
            gate_name="Usage",
            message=f"Found {success_count} successful use(s). Need ≥1.",
            details={'success_count': success_count}
        )

    # =========================================================================
    # Promotion Flow
    # =========================================================================

    def promote(self, skill_name: str) -> PromotionResult:
        """
        Attempt to promote a skill from drafts to production.

        Runs all 4 gates and promotes if all pass.
        """
        gates: List[GateResult] = []

        # Find draft
        draft_path = self.get_draft_path(skill_name)
        if draft_path is None:
            return PromotionResult(
                success=False,
                skill_name=skill_name,
                gates=[],
                error=f"Draft skill '{skill_name}' not found in {DRAFTS_DIR}"
            )

        logger.info(f"Checking skill '{skill_name}' for promotion...")

        # Gate 1: Evidence
        g1 = self.check_evidence(skill_name, draft_path)
        gates.append(g1)
        logger.info(f"  Gate 1 (Evidence): {'✓' if g1.passed else '✗'} {g1.message}")

        # Gate 2: Oracle
        g2 = self.check_oracle(skill_name, draft_path)
        gates.append(g2)
        logger.info(f"  Gate 2 (Oracle): {'✓' if g2.passed else '✗'} {g2.message}")

        # Gate 3: Dedup
        g3 = self.check_dedup(skill_name, draft_path)
        gates.append(g3)
        logger.info(f"  Gate 3 (Dedup): {'✓' if g3.passed else '✗'} {g3.message}")

        # Gate 4: Usage
        g4 = self.check_usage(skill_name, draft_path)
        gates.append(g4)
        logger.info(f"  Gate 4 (Usage): {'✓' if g4.passed else '✗'} {g4.message}")

        # Check if all gates passed
        all_passed = all(g.passed for g in gates)

        if not all_passed:
            failed_gates = [g.gate_name for g in gates if not g.passed]
            return PromotionResult(
                success=False,
                skill_name=skill_name,
                gates=gates,
                error=f"Failed gates: {', '.join(failed_gates)}"
            )

        # Promote: copy to skills directory
        promoted_path = SKILLS_DIR / skill_name

        try:
            # Ensure skills dir exists
            SKILLS_DIR.mkdir(parents=True, exist_ok=True)

            # Copy draft to promoted
            if promoted_path.exists():
                shutil.rmtree(promoted_path)
            shutil.copytree(draft_path, promoted_path)

            # Rename CANDIDATE.md to SKILL.md if needed
            candidate_file = promoted_path / "CANDIDATE.md"
            skill_file = promoted_path / "SKILL.md"
            if candidate_file.exists() and not skill_file.exists():
                candidate_file.rename(skill_file)
                logger.info(f"  Renamed CANDIDATE.md -> SKILL.md")

            # Update database status
            db_skill = self.get_skill_from_db(skill_name)
            if db_skill:
                cur = self.conn.cursor()
                cur.execute("""
                    UPDATE paper_skills
                    SET status = 'promoted',
                        promoted_to_skill_file = %s,
                        updated_at = NOW()
                    WHERE skill_id = %s
                """, (str(promoted_path / "SKILL.md"), db_skill['skill_id']))
                self.conn.commit()

            logger.info(f"✓ Promoted '{skill_name}' to {promoted_path}")

            return PromotionResult(
                success=True,
                skill_name=skill_name,
                gates=gates,
                promoted_path=promoted_path
            )

        except Exception as e:
            return PromotionResult(
                success=False,
                skill_name=skill_name,
                gates=gates,
                error=f"Promotion failed: {str(e)}"
            )

    def check_all_drafts(self) -> Dict[str, List[GateResult]]:
        """Check all draft skills and return gate results."""
        drafts = self.list_drafts()
        results = {}

        for skill_name in drafts:
            draft_path = self.get_draft_path(skill_name)
            if draft_path:
                gates = [
                    self.check_evidence(skill_name, draft_path),
                    self.check_oracle(skill_name, draft_path),
                    self.check_dedup(skill_name, draft_path),
                    self.check_usage(skill_name, draft_path),
                ]
                results[skill_name] = gates

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Promote skills from drafts to production with gate validation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Gates:
  1. Evidence  - ≥2 source passages OR 1 passage + 1 code link
  2. Oracle    - Runnable test exists and passes
  3. Dedup     - Not similar to existing promoted skill (cosine < 0.85)
  4. Usage     - At least 1 logged successful use (skip with --bootstrap)

Examples:
  %(prog)s spatial-autocorrelation           # Promote specific skill
  %(prog)s spatial-autocorrelation --bootstrap   # Skip usage check
  %(prog)s --list                            # List all draft skills
  %(prog)s --check-all                       # Check all drafts
"""
    )

    parser.add_argument('skill_name', nargs='?', help='Name of skill to promote')
    parser.add_argument('--bootstrap', action='store_true',
                       help='Skip usage gate (for initial skill promotion)')
    parser.add_argument('--list', action='store_true',
                       help='List all draft skills')
    parser.add_argument('--check-all', action='store_true',
                       help='Check all drafts without promoting')
    parser.add_argument('--dry-run', action='store_true',
                       help='Check gates but do not actually promote')

    args = parser.parse_args()

    promoter = SkillPromoter(bootstrap=args.bootstrap)

    if args.list:
        drafts = promoter.list_drafts()
        if drafts:
            print(f"Draft skills in {DRAFTS_DIR}:")
            for d in drafts:
                print(f"  - {d}")
        else:
            print(f"No draft skills found in {DRAFTS_DIR}")
        return 0

    if args.check_all:
        results = promoter.check_all_drafts()
        if not results:
            print("No draft skills found.")
            return 0

        print(f"\nGate check results for {len(results)} draft(s):\n")
        for skill_name, gates in results.items():
            passed = sum(1 for g in gates if g.passed)
            status = "✓ Ready" if passed == 4 else f"✗ {passed}/4"
            print(f"  {skill_name}: {status}")
            for g in gates:
                mark = "✓" if g.passed else "✗"
                print(f"    {mark} {g.gate_name}: {g.message[:60]}")
        return 0

    if not args.skill_name:
        parser.print_help()
        return 1

    # Promote skill
    result = promoter.promote(args.skill_name)

    if result.success:
        print(f"\n✓ Successfully promoted '{result.skill_name}' to {result.promoted_path}")
        return 0
    else:
        print(f"\n✗ Promotion failed for '{result.skill_name}'")
        if result.error:
            print(f"  Error: {result.error}")
        print("\nGate results:")
        for g in result.gates:
            mark = "✓" if g.passed else "✗"
            print(f"  {mark} {g.gate_name}: {g.message}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
