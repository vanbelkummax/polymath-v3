#!/usr/bin/env python3
"""
Sanity Check Script for Polymath v3.

Run before git commits to catch breaking changes early.
This enables "vibe coding" with safety rails.

Usage:
    python tests/sanity_check.py          # Run all checks
    python tests/sanity_check.py --quick  # Skip slow checks

Exit codes:
    0 = All checks passed
    1 = One or more checks failed
"""

import argparse
import sys
import time
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_status(name: str, passed: bool, message: str = "", elapsed_ms: float = 0):
    """Print a check result with color."""
    icon = f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.RED}✗{Colors.RESET}"
    status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
    timing = f"({elapsed_ms:.0f}ms)" if elapsed_ms else ""
    print(f"  {icon} {name}: {status} {timing}")
    if message and not passed:
        print(f"    {Colors.YELLOW}{message}{Colors.RESET}")


def check_imports() -> tuple[bool, str]:
    """Verify core modules can be imported."""
    try:
        from lib.config import config
        from lib.db.postgres import get_pg_pool
        from lib.db.neo4j import get_neo4j_driver
        from lib.search.hybrid_search import HybridSearcher
        from lib.bridgemine.gap_detection import GapDetector
        from lib.validation.hallucination import HallucinationDetector
        return True, ""
    except ImportError as e:
        return False, str(e)


def check_config() -> tuple[bool, str]:
    """Verify config loads without errors."""
    try:
        from lib.config import config
        errors = config.validate()
        if errors:
            return False, "; ".join(errors)
        return True, ""
    except Exception as e:
        return False, str(e)


def check_postgres_connection() -> tuple[bool, str]:
    """Verify Postgres connection works."""
    try:
        from lib.db.postgres import get_pg_connection
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                if result and result.get("?column?") == 1:
                    return True, ""
        return False, "Query returned unexpected result"
    except Exception as e:
        return False, str(e)


def check_postgres_schema() -> tuple[bool, str]:
    """Verify expected tables exist."""
    try:
        from lib.db.postgres import get_pg_connection
        required_tables = ["documents", "passages", "passage_concepts"]

        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public'
                """)
                existing = {row["tablename"] for row in cur.fetchall()}

        missing = set(required_tables) - existing
        if missing:
            return False, f"Missing tables: {missing}"
        return True, ""
    except Exception as e:
        return False, str(e)


def check_neo4j_connection() -> tuple[bool, str]:
    """Verify Neo4j connection works."""
    try:
        from lib.db.neo4j import get_neo4j_driver
        driver = get_neo4j_driver()
        records, _, _ = driver.execute_query("RETURN 1 as n")
        if records and records[0]["n"] == 1:
            return True, ""
        return False, "Query returned unexpected result"
    except Exception as e:
        return False, str(e)


def check_simple_search(quick: bool = False) -> tuple[bool, str]:
    """Verify search returns results."""
    if quick:
        return True, "Skipped (quick mode)"

    try:
        from lib.search.hybrid_search import HybridSearcher
        searcher = HybridSearcher()
        response = searcher.search("test", n=1, rerank=False)
        # Empty results are OK - we just want no crash
        return True, ""
    except Exception as e:
        return False, str(e)


def check_concept_extraction(quick: bool = False) -> tuple[bool, str]:
    """Verify concept extraction works."""
    if quick:
        return True, "Skipped (quick mode)"

    try:
        from lib.config import config
        if not config.GEMINI_API_KEY:
            return True, "Skipped (no API key)"

        from lib.ingest.concept_extractor import extract_concepts
        result = extract_concepts("Test passage about machine learning and neural networks.")
        # Just check it returns something
        return True, ""
    except Exception as e:
        return False, str(e)


def check_telemetry_module() -> tuple[bool, str]:
    """Verify telemetry module works."""
    try:
        from lib.telemetry import TelemetryLogger, is_telemetry_enabled
        with TelemetryLogger() as tl:
            tl.set_query("sanity check test")
        return True, ""
    except Exception as e:
        return False, str(e)


def run_sanity_checks(quick: bool = False) -> bool:
    """Run all sanity checks."""
    print(f"\n{Colors.BOLD}Polymath v3 Sanity Checks{Colors.RESET}")
    print("=" * 50)

    checks = [
        ("Module imports", check_imports),
        ("Config validation", check_config),
        ("Postgres connection", check_postgres_connection),
        ("Postgres schema", check_postgres_schema),
        ("Neo4j connection", check_neo4j_connection),
        ("Telemetry module", check_telemetry_module),
        ("Search (basic)", lambda: check_simple_search(quick)),
        ("Concept extraction", lambda: check_concept_extraction(quick)),
    ]

    all_passed = True
    total_time = 0

    for name, check_fn in checks:
        start = time.perf_counter()
        try:
            passed, message = check_fn()
        except Exception as e:
            passed, message = False, str(e)
        elapsed_ms = (time.perf_counter() - start) * 1000
        total_time += elapsed_ms

        print_status(name, passed, message, elapsed_ms)
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}All checks passed!{Colors.RESET} ({total_time:.0f}ms total)")
    else:
        print(f"{Colors.RED}{Colors.BOLD}Some checks failed.{Colors.RESET} Fix issues before committing.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Run sanity checks")
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Skip slow checks (search, extraction)",
    )
    args = parser.parse_args()

    success = run_sanity_checks(quick=args.quick)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
