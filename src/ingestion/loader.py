# src/ingestion/loader.py
"""
Document loader — reads raw files and returns plain text.

Supports: PDF (via pdfplumber), plain text (.txt), markdown (.md).

For PDFs: extracts text page by page. If a page has no extractable
text (scanned image), logs a warning and skips that page rather than
returning empty content that would fail the chunk contract downstream.

Single responsibility: file path in, text string out.
All validation of what to do with that text happens in chunker.py.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from src.config.logging_config import get_logger

logger = get_logger(__name__)


def load_document(file_path: str | Path) -> tuple[str, str]:
    """
    Load a document and return its text content and detected file type.

    Args:
        file_path: Path to the document file.

    Returns:
        Tuple of (text_content, file_type).
        file_type is one of: 'pdf', 'txt', 'md'.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = path.suffix.lower().lstrip(".")

    if suffix == "pdf":
        return _load_pdf(path), "pdf"
    elif suffix in ("txt", "text"):
        return _load_text(path), "txt"
    elif suffix == "md":
        return _load_text(path), "md"
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported types: pdf, txt, md")


def _load_pdf(path: Path) -> str:
    """
    Extract text from a PDF using pdfplumber.

    Processes page by page. Skips pages with no extractable text
    (these are typically scanned images — OCR is out of scope).
    Logs a warning for each skipped page so failures are visible.
    """
    pages_text: list[str] = []
    skipped_pages = 0

    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(text.strip())
            else:
                skipped_pages += 1
                logger.warning(
                    f"{path.name}: page {page_num}/{total_pages} "
                    f"has no extractable text (possibly scanned) — skipping"
                )

    if not pages_text:
        logger.error(
            f"{path.name}: no extractable text found in any of "
            f"{total_pages} pages — document will produce no chunks"
        )
        return ""

    full_text = "\n\n".join(pages_text)
    logger.info(
        f"Loaded PDF: {path.name} — "
        f"{total_pages} pages, {skipped_pages} skipped, "
        f"{len(full_text):,} characters extracted"
    )
    return full_text


def _load_text(path: Path) -> str:
    """Load a plain text or markdown file."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back to latin-1 for older legal documents
        text = path.read_text(encoding="latin-1")
        logger.warning(f"{path.name}: UTF-8 decode failed, used latin-1 fallback")

    logger.info(f"Loaded text: {path.name} — {len(text):,} characters")
    return text
