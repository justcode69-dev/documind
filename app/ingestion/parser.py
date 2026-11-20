"""PDF -> page-tagged text using PyMuPDF."""

import re
from dataclasses import dataclass

import pymupdf


class PDFError(ValueError):
    """The upload is not a usable PDF (corrupt, encrypted, scanned-only, too long)."""


@dataclass(frozen=True)
class Page:
    page_number: int  # 1-based, matches what a human sees in a PDF viewer
    text: str


PDF_MAGIC = b"%PDF-"


def looks_like_pdf(data: bytes) -> bool:
    # Content-Type headers can lie; check the file signature too.
    return data[:1024].lstrip().startswith(PDF_MAGIC)


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(data: bytes, max_pages: int = 500) -> list[Page]:
    if not looks_like_pdf(data):
        raise PDFError("File is not a PDF.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:  # pymupdf raises several types for corrupt files
        raise PDFError("Could not open PDF (file may be corrupt).") from e
    with doc:
        if doc.needs_pass:
            raise PDFError("PDF is password-protected.")
        if doc.page_count > max_pages:
            raise PDFError(f"PDF has {doc.page_count} pages; limit is {max_pages}.")
        pages = [Page(i + 1, _clean(p.get_text("text"))) for i, p in enumerate(doc)]
    pages = [p for p in pages if p.text]
    if not pages:
        raise PDFError("No extractable text found (scanned PDF? OCR is not supported).")
    return pages
