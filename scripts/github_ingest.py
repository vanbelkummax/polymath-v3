#!/usr/bin/env python3
"""
GitHub Repository Ingestion for Polymath

Modes:
1. Single repo: python scripts/github_ingest.py https://github.com/owner/repo
2. Username/org: python scripts/github_ingest.py --user owner
3. From queue: python scripts/github_ingest.py --queue
4. Discover from papers: python scripts/github_ingest.py --discover

Features:
- Clones repos to local staging
- Extracts semantic chunks (functions, classes, modules)
- Stores in Postgres with full provenance
- Tracks in repo_queue for management
"""

import os
import re
import sys
import json
import shutil
import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values
import requests

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Config
REPOS_DIR = Path("/home/user/work/polymax/data/github_repos")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("PAT_2")

# Try importing from polymath-repo if available
try:
    sys.path.insert(0, '/home/user/polymath-repo')
    from lib.code_ingest import ingest_repo, scan_repo
    HAS_CODE_INGEST = True
except ImportError:
    HAS_CODE_INGEST = False
    logger.warning("code_ingest not available, will use basic ingestion")


def get_db_connection():
    """Get database connection."""
    dsn = os.environ.get('POSTGRES_DSN', 'dbname=polymath user=polymath host=/var/run/postgresql')
    return psycopg2.connect(dsn)


def get_github_headers():
    """Get headers for GitHub API."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def parse_github_url(url: str) -> Optional[Tuple[str, str]]:
    """Parse GitHub URL into (owner, repo)."""
    patterns = [
        r'github\.com/([^/]+)/([^/\s\.]+)',
        r'^([^/]+)/([^/\s]+)$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2).rstrip('.git')
    return None


def clone_repo(owner: str, repo: str) -> Optional[Path]:
    """Clone a repository to staging."""
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    repo_path = REPOS_DIR / f"{owner}__{repo}"

    if repo_path.exists():
        logger.info(f"Repository already exists: {repo_path}")
        # Pull latest
        try:
            subprocess.run(
                ['git', '-C', str(repo_path), 'pull', '--ff-only'],
                capture_output=True, timeout=60
            )
        except Exception as e:
            logger.debug(f"Pull failed: {e}")
        return repo_path

    # Clone
    clone_url = f"https://github.com/{owner}/{repo}.git"
    logger.info(f"Cloning {clone_url}")

    try:
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', clone_url, str(repo_path)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            logger.error(f"Clone failed: {result.stderr}")
            return None
        return repo_path
    except subprocess.TimeoutExpired:
        logger.error(f"Clone timed out for {owner}/{repo}")
        return None
    except Exception as e:
        logger.error(f"Clone error: {e}")
        return None


def fetch_user_repos(username: str, include_forks: bool = False) -> List[Dict]:
    """Fetch all repositories for a user/organization."""
    repos = []
    page = 1

    while True:
        # Try user repos first
        url = f"https://api.github.com/users/{username}/repos?per_page=100&page={page}"
        response = requests.get(url, headers=get_github_headers())

        if response.status_code == 404:
            # Try org repos
            url = f"https://api.github.com/orgs/{username}/repos?per_page=100&page={page}"
            response = requests.get(url, headers=get_github_headers())

        if response.status_code != 200:
            if page == 1:
                logger.error(f"Failed to fetch repos for {username}: {response.status_code}")
            break

        data = response.json()
        if not data:
            break

        for repo in data:
            if not include_forks and repo.get('fork'):
                continue
            repos.append({
                'owner': username,
                'name': repo['name'],
                'full_name': repo['full_name'],
                'description': repo.get('description'),
                'url': repo['html_url'],
                'stars': repo.get('stargazers_count', 0),
                'language': repo.get('language'),
                'updated_at': repo.get('updated_at'),
            })

        page += 1
        if len(data) < 100:
            break

    return repos


def add_to_queue(conn, owner: str, repo: str, source: str, context: str = None,
                 priority: int = 5) -> bool:
    """Add repository to ingestion queue."""
    cur = conn.cursor()

    # Check if already in queue
    cur.execute("""
        SELECT repo_id, status FROM repo_queue
        WHERE repo_url LIKE %s OR (owner = %s AND repo_name = %s)
    """, (f"%{owner}/{repo}%", owner, repo))

    existing = cur.fetchone()
    if existing:
        logger.debug(f"Already in queue: {owner}/{repo}")
        return False

    # Add to queue
    cur.execute("""
        INSERT INTO repo_queue (owner, repo_name, repo_url, source, context, priority, status)
        VALUES (%s, %s, %s, %s, %s, %s, 'pending')
        ON CONFLICT DO NOTHING
        RETURNING repo_id
    """, (owner, repo, f"https://github.com/{owner}/{repo}", source, context, priority))

    result = cur.fetchone()
    conn.commit()

    if result:
        logger.info(f"Added to queue: {owner}/{repo}")
        return True
    return False


def update_queue_status(conn, owner: str, repo: str, status: str,
                        stats: Dict = None):
    """Update queue entry status."""
    cur = conn.cursor()

    extra_sql = ""
    params = [status, owner, repo]

    if stats:
        extra_sql = ", files_count = %s, chunks_count = %s, ingested_at = NOW()"
        params = [status, stats.get('files', 0), stats.get('chunks', 0), owner, repo]

    cur.execute(f"""
        UPDATE repo_queue
        SET status = %s{extra_sql}
        WHERE owner = %s AND repo_name = %s
    """, params)
    conn.commit()


def ingest_single_repo(owner: str, repo: str, conn=None) -> Dict:
    """Ingest a single repository."""
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    stats = {'files': 0, 'chunks': 0, 'errors': 0, 'status': 'pending'}

    try:
        # Clone repo
        repo_path = clone_repo(owner, repo)
        if not repo_path:
            stats['status'] = 'clone_failed'
            return stats

        # Update queue
        update_queue_status(conn, owner, repo, 'ingesting')

        # Ingest using polymath-repo code_ingest if available
        if HAS_CODE_INGEST:
            result = ingest_repo(repo_path, conn)
            stats['files'] = result.get('files', 0)
            stats['chunks'] = result.get('chunks', 0)
            stats['errors'] = result.get('errors', 0)
        else:
            # Basic fallback ingestion
            stats = basic_ingest(repo_path, f"{owner}/{repo}", conn)

        stats['status'] = 'completed' if stats['errors'] == 0 else 'partial'
        update_queue_status(conn, owner, repo, stats['status'], stats)

        logger.info(f"Ingested {owner}/{repo}: {stats['files']} files, {stats['chunks']} chunks")

    except Exception as e:
        logger.error(f"Error ingesting {owner}/{repo}: {e}")
        stats['status'] = 'error'
        stats['error'] = str(e)
        update_queue_status(conn, owner, repo, 'error')

    finally:
        if close_conn:
            conn.close()

    return stats


def basic_ingest(repo_path: Path, repo_name: str, conn) -> Dict:
    """Basic code ingestion when polymath-repo not available."""
    import ast
    import hashlib
    import uuid

    stats = {'files': 0, 'chunks': 0, 'errors': 0}
    cur = conn.cursor()

    # Get commit info
    commit_sha = 'unknown'
    try:
        result = subprocess.run(
            ['git', '-C', str(repo_path), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            commit_sha = result.stdout.strip()
    except:
        pass

    # Scan Python files
    for py_file in repo_path.rglob('*.py'):
        rel_path = str(py_file.relative_to(repo_path))

        # Skip test files, etc
        if any(skip in rel_path for skip in ['test', '__pycache__', 'setup.py', '.git']):
            continue

        try:
            content = py_file.read_text(encoding='utf-8', errors='ignore')
            if len(content) < 50 or len(content) > 500000:
                continue

            # Insert file record
            file_id = str(uuid.uuid4())
            file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            cur.execute("""
                INSERT INTO code_files
                (file_id, repo_name, file_path, head_commit_sha, language, file_hash,
                 file_size_bytes, loc)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (repo_name, file_path, head_commit_sha) DO NOTHING
                RETURNING file_id
            """, (file_id, repo_name, rel_path, commit_sha, 'python', file_hash,
                  len(content), content.count('\n')))

            result = cur.fetchone()
            if not result:
                continue

            stats['files'] += 1

            # Parse and extract chunks
            try:
                tree = ast.parse(content)
                lines = content.split('\n')

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        start = node.lineno - 1
                        end = getattr(node, 'end_lineno', start + 20)
                        chunk_content = '\n'.join(lines[start:end])

                        chunk_type = 'class' if isinstance(node, ast.ClassDef) else 'function'
                        docstring = ast.get_docstring(node)

                        cur.execute("""
                            INSERT INTO code_chunks
                            (chunk_id, file_id, chunk_type, name, start_line, end_line,
                             content, docstring)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (str(uuid.uuid4()), file_id, chunk_type, node.name,
                              start + 1, end, chunk_content[:5000],
                              docstring[:1000] if docstring else None))
                        stats['chunks'] += 1

            except SyntaxError:
                pass

        except Exception as e:
            logger.debug(f"Error processing {py_file}: {e}")
            stats['errors'] += 1

    conn.commit()
    return stats


def discover_from_papers(conn) -> List[Tuple[str, str]]:
    """Discover GitHub repos mentioned in papers."""
    cur = conn.cursor()

    # Get repos from repo_queue that came from paper detection
    cur.execute("""
        SELECT DISTINCT owner, repo_name FROM repo_queue
        WHERE source = 'paper_detection' AND status = 'pending'
        ORDER BY priority DESC, created_at DESC
        LIMIT 50
    """)

    repos = [(row[0], row[1]) for row in cur.fetchall()]

    if not repos:
        # Also search passages for GitHub URLs not yet in queue
        cur.execute("""
            SELECT DISTINCT
                substring(passage_text from 'github\.com/([^/\s]+)/([^/\s\.]+)') as match
            FROM passages
            WHERE passage_text ~* 'github\.com/[a-z0-9_-]+/[a-z0-9_-]+'
            LIMIT 100
        """)

        pattern = re.compile(r'github\.com/([^/\s]+)/([^/\s\.]+)')
        for row in cur.fetchall():
            if row[0]:
                match = pattern.search(row[0])
                if match:
                    repos.append((match.group(1), match.group(2)))

    return repos[:50]


def process_queue(conn, limit: int = 10):
    """Process pending repos from queue."""
    cur = conn.cursor()

    cur.execute("""
        SELECT owner, repo_name FROM repo_queue
        WHERE status = 'pending'
        ORDER BY priority DESC, created_at ASC
        LIMIT %s
    """, (limit,))

    repos = cur.fetchall()

    total_stats = {'repos': 0, 'files': 0, 'chunks': 0}

    for owner, repo in repos:
        stats = ingest_single_repo(owner, repo, conn)
        total_stats['repos'] += 1
        total_stats['files'] += stats.get('files', 0)
        total_stats['chunks'] += stats.get('chunks', 0)

    return total_stats


def list_queue(conn, status: str = None):
    """List repos in queue."""
    cur = conn.cursor()

    sql = """
        SELECT owner, repo_name, status, priority, source,
               files_count, chunks_count, created_at
        FROM repo_queue
    """
    params = []

    if status:
        sql += " WHERE status = %s"
        params.append(status)

    sql += " ORDER BY priority DESC, created_at DESC LIMIT 50"

    cur.execute(sql, params)

    print("\n" + "=" * 80)
    print("GITHUB REPOSITORY QUEUE")
    print("=" * 80)
    print(f"{'Repo':<40} {'Status':<12} {'Pri':<4} {'Files':<6} {'Chunks':<8} {'Source':<15}")
    print("-" * 80)

    for row in cur.fetchall():
        owner, repo, status, priority, source, files, chunks, created = row
        print(f"{owner}/{repo:<38} {status:<12} {priority:<4} {files or '-':<6} {chunks or '-':<8} {source or '-':<15}")

    # Summary
    cur.execute("""
        SELECT status, COUNT(*) FROM repo_queue GROUP BY status ORDER BY status
    """)
    print("\n" + "-" * 40)
    print("Summary:")
    for status, count in cur.fetchall():
        print(f"  {status}: {count}")


def main():
    parser = argparse.ArgumentParser(description='GitHub Repository Ingestion')
    parser.add_argument('url', nargs='?', help='GitHub URL or owner/repo')
    parser.add_argument('--user', '-u', help='Fetch all repos from user/org')
    parser.add_argument('--queue', '-q', action='store_true', help='Process pending queue')
    parser.add_argument('--discover', '-d', action='store_true', help='Discover repos from papers')
    parser.add_argument('--list', '-l', action='store_true', help='List queue')
    parser.add_argument('--status', '-s', help='Filter by status for --list')
    parser.add_argument('--limit', type=int, default=10, help='Limit for --queue')
    parser.add_argument('--add-only', action='store_true', help='Only add to queue, do not ingest')
    parser.add_argument('--priority', type=int, default=5, help='Priority for queue (1-10)')
    args = parser.parse_args()

    conn = get_db_connection()

    if args.list:
        list_queue(conn, args.status)
        return

    if args.user:
        # Fetch all repos from user/org
        logger.info(f"Fetching repos for {args.user}")
        repos = fetch_user_repos(args.user)
        logger.info(f"Found {len(repos)} repos")

        for repo in repos:
            add_to_queue(conn, repo['owner'], repo['name'],
                        f'user:{args.user}', repo.get('description'),
                        args.priority)

        if not args.add_only:
            process_queue(conn, args.limit)

    elif args.discover:
        # Discover from papers
        repos = discover_from_papers(conn)
        logger.info(f"Discovered {len(repos)} repos from papers")

        for owner, repo in repos:
            add_to_queue(conn, owner, repo, 'paper_discovery', priority=args.priority)

        if not args.add_only:
            process_queue(conn, args.limit)

    elif args.queue:
        # Process queue
        stats = process_queue(conn, args.limit)
        logger.info(f"Processed {stats['repos']} repos: {stats['files']} files, {stats['chunks']} chunks")

    elif args.url:
        # Single repo
        parsed = parse_github_url(args.url)
        if not parsed:
            logger.error(f"Could not parse URL: {args.url}")
            return

        owner, repo = parsed

        if args.add_only:
            add_to_queue(conn, owner, repo, 'manual', priority=args.priority)
        else:
            add_to_queue(conn, owner, repo, 'manual', priority=args.priority)
            stats = ingest_single_repo(owner, repo, conn)
            print(f"\nResult: {stats}")

    else:
        parser.print_help()

    conn.close()


if __name__ == '__main__':
    main()
