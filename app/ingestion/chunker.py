"""Chunking strategies. Chunks never cross a page boundary, so every chunk has one exact page
number for its citation."""

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.parser import Page


@dataclass
class Chunk:
    text: str
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def payload(self) -> dict:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page": self.page,
            "chunk_index": self.chunk_index,
            **self.metadata,
        }


def _fixed_split(text: str, size: int, overlap: int) -> list[str]:
    """Baseline: fixed-size character windows, blind to sentence/paragraph boundaries."""
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            out.append(piece)
        if start + size >= len(text):
            break
    return out


def chunk_pages(pages: list[Page], *, doc_id: str, filename: str, strategy: str = "recursive",
                chunk_size: int = 800, chunk_overlap: int = 100) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "। ", " ", ""],  # includes Hindi danda
            keep_separator="end",  # sentences keep their full stop instead of the next chunk starting with ". "
        )
        split = splitter.split_text
    elif strategy == "fixed":
        split = lambda t: _fixed_split(t, chunk_size, chunk_overlap)  # noqa: E731
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    chunks: list[Chunk] = []
    for page in pages:
        for piece in split(page.text):
            chunks.append(Chunk(piece, doc_id, filename, page.page_number, len(chunks)))
    return chunks
