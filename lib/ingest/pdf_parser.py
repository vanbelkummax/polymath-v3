"""
PDF parsing for Polymath v3.

Uses PyMuPDF (fitz) as primary parser with fallback options.
Handles:
- Text extraction with page tracking
- Table detection
- Figure detection
- OCR fallback for scanned PDFs
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of PDF parsing."""

    text: str
    page_count: int
    pages: list[dict] = field(default_factory=list)  # {page_num, text, char_start, char_end}

    # Quality indicators
    has_text: bool = True
    is_scanned: bool = False
    extraction_method: str = "fitz"

    # Detected elements
    has_tables: bool = False
    has_figures: bool = False
    has_equations: bool = False

    # Errors
    errors: list[str] = field(default_factory=list)


class PDFParser:
    """
    PDF text extraction with quality handling.

    Usage:
        parser = PDFParser()
        result = parser.parse("/path/to/paper.pdf")
        print(result.text)
    """

    def __init__(
        self,
        strip_nul: bool = True,
        detect_tables: bool = True,
        ocr_fallback: bool = False,
    ):
        """
        Initialize parser.

        Args:
            strip_nul: Remove NUL characters (common PDF artifact)
            detect_tables: Try to detect tables in content
            ocr_fallback: Use OCR for scanned PDFs (requires pytesseract)
        """
        self.strip_nul = strip_nul
        self.detect_tables = detect_tables
        self.ocr_fallback = ocr_fallback

    def parse(self, pdf_path: Path) -> ParseResult:
        """
        Parse a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ParseResult with extracted text and metadata
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            return ParseResult(
                text="",
                page_count=0,
                has_text=False,
                errors=[f"File not found: {pdf_path}"],
            )

        try:
            return self._parse_with_fitz(pdf_path)
        except Exception as e:
            logger.error(f"PDF parsing failed for {pdf_path}: {e}")
            return ParseResult(
                text="",
                page_count=0,
                has_text=False,
                errors=[str(e)],
            )

    def _parse_with_fitz(self, pdf_path: Path) -> ParseResult:
        """Parse PDF using PyMuPDF (fitz)."""
        import fitz

        doc = fitz.open(pdf_path)
        pages = []
        all_text = []
        char_offset = 0

        has_tables = False
        has_figures = False
        has_equations = False

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")

                # Clean text
                if self.strip_nul:
                    text = text.replace("\x00", "")

                # Track page boundaries
                page_start = char_offset
                page_end = char_offset + len(text)

                pages.append({
                    "page_num": page_num + 1,  # 1-indexed
                    "text": text,
                    "char_start": page_start,
                    "char_end": page_end,
                })

                all_text.append(text)
                char_offset = page_end + 1  # +1 for page separator

                # Detect elements
                if self.detect_tables:
                    if self._looks_like_table(text):
                        has_tables = True

                if self._has_figures(page):
                    has_figures = True

                if self._has_equations(text):
                    has_equations = True

            full_text = "\n".join(all_text)

            # Check if PDF might be scanned (very little text)
            is_scanned = len(full_text.strip()) < 100 and len(doc) > 0

            if is_scanned and self.ocr_fallback:
                logger.info(f"PDF appears scanned, attempting OCR: {pdf_path}")
                ocr_result = self._ocr_pdf(pdf_path)
                if ocr_result:
                    return ocr_result

            return ParseResult(
                text=full_text,
                page_count=len(doc),
                pages=pages,
                has_text=len(full_text.strip()) > 100,
                is_scanned=is_scanned,
                extraction_method="fitz",
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
            )

        finally:
            doc.close()

    def _looks_like_table(self, text: str) -> bool:
        """Heuristic detection of table content."""
        lines = text.split("\n")

        # Count lines with multiple tab/space-separated columns
        tabular_lines = 0
        for line in lines:
            # Check for consistent spacing pattern
            parts = re.split(r"\s{2,}|\t", line.strip())
            if len(parts) >= 3:
                tabular_lines += 1

        return tabular_lines >= 5

    def _has_figures(self, page) -> bool:
        """Check if page has images/figures."""
        try:
            images = page.get_images()
            return len(images) > 0
        except Exception:
            return False

    def _has_equations(self, text: str) -> bool:
        """Heuristic detection of mathematical equations."""
        # Look for common equation patterns
        equation_patterns = [
            r"\\frac\{",  # LaTeX fractions
            r"\\sum",  # Summation
            r"\\int",  # Integral
            r"\$.*?\$",  # Inline math
            r"=\s*\\",  # Equals followed by LaTeX
            r"∑|∫|∏|√",  # Unicode math symbols
        ]

        for pattern in equation_patterns:
            if re.search(pattern, text):
                return True

        return False

    def _ocr_pdf(self, pdf_path: Path) -> Optional[ParseResult]:
        """OCR fallback for scanned PDFs."""
        try:
            import pytesseract
            from PIL import Image
            import fitz

            doc = fitz.open(pdf_path)
            pages = []
            all_text = []
            char_offset = 0

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Render page as image
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # OCR
                text = pytesseract.image_to_string(img)

                page_start = char_offset
                page_end = char_offset + len(text)

                pages.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "char_start": page_start,
                    "char_end": page_end,
                })

                all_text.append(text)
                char_offset = page_end + 1

            doc.close()

            return ParseResult(
                text="\n".join(all_text),
                page_count=len(pages),
                pages=pages,
                has_text=True,
                is_scanned=True,
                extraction_method="ocr",
            )

        except ImportError:
            logger.warning("pytesseract not installed, skipping OCR")
            return None
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None


def extract_text(pdf_path: Path) -> tuple[str, str]:
    """
    Convenience function to extract text from PDF.

    Returns:
        Tuple of (text, extraction_method)
    """
    parser = PDFParser()
    result = parser.parse(pdf_path)
    return result.text, result.extraction_method
