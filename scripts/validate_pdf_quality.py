#!/usr/bin/env python3
"""
Validate PDF text extraction quality.

Tests a sample of PDFs to ensure PyMuPDF extracts clean text (not OCR garbage).

Usage:
    python scripts/validate_pdf_quality.py --sample 20
    python scripts/validate_pdf_quality.py --csv /path/to/zotero.csv --sample 20
"""

import argparse
import csv
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct import to avoid cascading DB imports
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pdf_parser",
    Path(__file__).parent.parent / "lib" / "ingest" / "pdf_parser.py"
)
pdf_parser_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdf_parser_module)
PDFParser = pdf_parser_module.PDFParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class PDFQualityResult:
    """Quality metrics for a single PDF."""

    path: Path
    pages: int = 0
    chars: int = 0
    has_text: bool = False
    is_scanned: bool = False
    extraction_method: str = ""
    alphanumeric_ratio: float = 0.0
    avg_chars_per_page: float = 0.0
    has_headers: bool = False
    sample_text: str = ""
    error: Optional[str] = None


@dataclass
class QualityReport:
    """Aggregate quality report."""

    total_pdfs: int = 0
    successful: int = 0
    failed: int = 0
    scanned_count: int = 0
    clean_extraction_count: int = 0
    avg_alphanumeric_ratio: float = 0.0
    avg_chars_per_page: float = 0.0
    results: list[PDFQualityResult] = field(default_factory=list)


def calculate_alphanumeric_ratio(text: str) -> float:
    """Calculate ratio of alphanumeric characters to total."""
    if not text:
        return 0.0
    alphanumeric = sum(1 for c in text if c.isalnum() or c.isspace())
    return alphanumeric / len(text)


def has_section_headers(text: str) -> bool:
    """Check if text has recognizable section headers."""
    patterns = [
        r"^#{1,3}\s+\w+",  # Markdown headers
        r"^[A-Z][A-Z\s]{2,20}$",  # ALL CAPS headers
        r"^\d+\.\s+[A-Z]",  # Numbered sections
        r"^(Abstract|Introduction|Methods|Results|Discussion|Conclusion)",
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False


def validate_pdf(pdf_path: Path, parser: PDFParser) -> PDFQualityResult:
    """Validate a single PDF's extraction quality."""
    result = PDFQualityResult(path=pdf_path)

    try:
        parse_result = parser.parse(pdf_path)

        result.pages = parse_result.page_count
        result.chars = len(parse_result.text)
        result.has_text = parse_result.has_text
        result.is_scanned = parse_result.is_scanned
        result.extraction_method = parse_result.extraction_method

        if result.chars > 0:
            result.alphanumeric_ratio = calculate_alphanumeric_ratio(parse_result.text)
            result.avg_chars_per_page = result.chars / max(1, result.pages)
            result.has_headers = has_section_headers(parse_result.text)
            result.sample_text = parse_result.text[:300].replace("\n", " ")

    except Exception as e:
        result.error = str(e)
        logger.error(f"Failed to parse {pdf_path}: {e}")

    return result


def get_pdfs_from_csv(csv_path: Path, limit: int = 100) -> list[Path]:
    """Extract PDF paths from Zotero CSV."""
    pdfs = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            attachments = row.get("File Attachments", "")
            for attachment in attachments.split(";"):
                attachment = attachment.strip()
                if attachment.lower().endswith(".pdf"):
                    # Convert Windows path to WSL
                    match = re.match(r"^([A-Za-z]):\\(.+)$", attachment)
                    if match:
                        drive = match.group(1).lower()
                        rest = match.group(2).replace("\\", "/")
                        wsl_path = Path(f"/mnt/{drive}/{rest}")
                        if wsl_path.exists():
                            pdfs.append(wsl_path)
                    elif attachment.startswith("/"):
                        path = Path(attachment)
                        if path.exists():
                            pdfs.append(path)

            if len(pdfs) >= limit:
                break

    return pdfs


def get_pdfs_from_directory(directory: Path, limit: int = 100) -> list[Path]:
    """Get PDFs from a directory recursively."""
    pdfs = list(directory.rglob("*.pdf"))[:limit]
    return [p for p in pdfs if p.exists()]


def run_validation(
    pdfs: list[Path],
    sample_size: int = 20,
) -> QualityReport:
    """Run validation on a sample of PDFs."""
    if len(pdfs) > sample_size:
        pdfs = random.sample(pdfs, sample_size)

    parser = PDFParser(strip_nul=True, detect_tables=True)
    report = QualityReport(total_pdfs=len(pdfs))

    logger.info(f"Validating {len(pdfs)} PDFs...")

    for i, pdf_path in enumerate(pdfs):
        logger.info(f"[{i + 1}/{len(pdfs)}] {pdf_path.name}")
        result = validate_pdf(pdf_path, parser)
        report.results.append(result)

        if result.error:
            report.failed += 1
        else:
            report.successful += 1
            if result.is_scanned:
                report.scanned_count += 1
            if result.has_text and result.alphanumeric_ratio > 0.7:
                report.clean_extraction_count += 1

    # Calculate averages
    valid_results = [r for r in report.results if not r.error and r.chars > 0]
    if valid_results:
        report.avg_alphanumeric_ratio = sum(r.alphanumeric_ratio for r in valid_results) / len(valid_results)
        report.avg_chars_per_page = sum(r.avg_chars_per_page for r in valid_results) / len(valid_results)

    return report


def print_report(report: QualityReport):
    """Print the quality report."""
    print("\n" + "=" * 70)
    print("PDF QUALITY VALIDATION REPORT")
    print("=" * 70)

    print(f"\nSummary:")
    print(f"  Total PDFs tested:     {report.total_pdfs}")
    print(f"  Successful:            {report.successful}")
    print(f"  Failed:                {report.failed}")
    print(f"  Scanned (need OCR):    {report.scanned_count}")
    print(f"  Clean extraction:      {report.clean_extraction_count}")

    clean_pct = (report.clean_extraction_count / max(1, report.successful)) * 100
    print(f"\n  Clean extraction rate: {clean_pct:.1f}%")
    print(f"  Avg alphanumeric ratio: {report.avg_alphanumeric_ratio:.2f}")
    print(f"  Avg chars/page:        {report.avg_chars_per_page:.0f}")

    # Pass/Fail
    print("\n" + "-" * 70)
    if clean_pct >= 90:
        print("RESULT: PASS (>90% clean extraction)")
    else:
        print(f"RESULT: FAIL ({clean_pct:.1f}% < 90% threshold)")
    print("-" * 70)

    # Sample details
    print("\nSample Results:")
    print("-" * 70)
    for r in report.results[:10]:
        status = "OK" if r.has_text and r.alphanumeric_ratio > 0.7 else "WARN"
        if r.error:
            status = "ERR"
        elif r.is_scanned:
            status = "SCAN"

        print(f"\n[{status}] {r.path.name}")
        print(f"     Pages: {r.pages}, Chars: {r.chars}, Method: {r.extraction_method}")
        print(f"     Alphanumeric: {r.alphanumeric_ratio:.2f}, Headers: {r.has_headers}")
        if r.sample_text:
            print(f"     Sample: {r.sample_text[:100]}...")
        if r.error:
            print(f"     Error: {r.error}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Validate PDF extraction quality")
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        help="Zotero CSV file(s) to get PDF paths from",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        help="Directory to scan for PDFs",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=20,
        help="Number of PDFs to sample (default: 20)",
    )

    args = parser.parse_args()

    # Get PDFs
    pdfs = []
    if args.csv:
        for csv_path in args.csv:
            logger.info(f"Loading PDFs from {csv_path}...")
            pdfs.extend(get_pdfs_from_csv(csv_path, limit=200))
    elif args.directory:
        logger.info(f"Scanning {args.directory} for PDFs...")
        pdfs = get_pdfs_from_directory(args.directory, limit=200)
    else:
        # Default: Zotero storage
        default_path = Path("/mnt/c/Users/User/Zotero/storage/")
        if default_path.exists():
            logger.info(f"Scanning default Zotero path: {default_path}")
            pdfs = get_pdfs_from_directory(default_path, limit=200)
        else:
            logger.error("No PDF source specified and default path not found")
            sys.exit(1)

    logger.info(f"Found {len(pdfs)} PDFs")

    if not pdfs:
        logger.error("No PDFs found")
        sys.exit(1)

    # Run validation
    report = run_validation(pdfs, sample_size=args.sample)

    # Print report
    print_report(report)

    # Exit code based on pass/fail
    clean_pct = (report.clean_extraction_count / max(1, report.successful)) * 100
    sys.exit(0 if clean_pct >= 90 else 1)


if __name__ == "__main__":
    main()
