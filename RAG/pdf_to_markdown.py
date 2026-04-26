"""
RAG/pdf_to_markdown.py
──────────────────────
Convert a PDF to clean Markdown, then split it into overlapping chunks
ready for vector ingestion.

Usage:
  python RAG/pdf_to_markdown.py path/to/file.pdf
"""

import pathlib
import re
from dataclasses import dataclass

import pymupdf4llm


@dataclass
class ConversionResult:
    source_pdf: str
    output_md:  str
    char_count: int
    success:    bool
    error:      str | None = None


def pdf_to_markdown(
    pdf_path: str,
    output_path: str | None = None,
    pages: list[int] | None = None,
) -> ConversionResult:
    """
    Convert a PDF to Markdown using pymupdf4llm.

    Args:
        pdf_path:    Path to the input PDF.
        output_path: Destination .md file. Defaults to same dir as PDF.
        pages:       0-based page list to convert. None = all pages.

    Returns:
        ConversionResult with output path, char count, and success flag.
    """
    pdf_path = pathlib.Path(pdf_path)
    if not pdf_path.exists():
        return ConversionResult(
            source_pdf=str(pdf_path),
            output_md="",
            char_count=0,
            success=False,
            error=f"File not found: {pdf_path}",
        )

    if output_path is None:
        output_path = pdf_path.with_suffix(".md")
    output_path = pathlib.Path(output_path)

    try:
        md_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            pages=pages,
            show_progress=True,
        )
        md_text = re.sub(r"\n{3,}", "\n\n", md_text)
        output_path.write_bytes(md_text.encode("utf-8"))
        return ConversionResult(
            source_pdf=str(pdf_path),
            output_md=str(output_path),
            char_count=len(md_text),
            success=True,
        )
    except Exception as e:
        return ConversionResult(
            source_pdf=str(pdf_path),
            output_md="",
            char_count=0,
            success=False,
            error=str(e),
        )


def chunk_markdown(
    md_path: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[dict]:
    """
    Split a Markdown file into overlapping chunks for vector ingestion.

    Splits on paragraph boundaries first, then falls back to
    character-level splitting to respect chunk_size.

    Args:
        md_path:    Path to the .md file.
        chunk_size: Target characters per chunk.
        overlap:    Character overlap between consecutive chunks.

    Returns:
        List of dicts: [{text, chunk_index, source}, ...]
    """
    text       = pathlib.Path(md_path).read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    source     = pathlib.Path(md_path).name

    chunks  = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current)
            while len(para) > chunk_size:
                chunks.append(para[:chunk_size])
                para = para[chunk_size - overlap:]
            current = para

    if current:
        chunks.append(current)

    return [
        {"text": chunk, "chunk_index": i, "source": source}
        for i, chunk in enumerate(chunks)
    ]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python RAG/pdf_to_markdown.py <path/to/file.pdf>")
        sys.exit(1)

    result = pdf_to_markdown(sys.argv[1])
    if result.success:
        print(f"Converted → {result.output_md}  ({result.char_count:,} chars)")
        chunks = chunk_markdown(result.output_md)
        print(f"Chunked into {len(chunks)} pieces")
    else:
        print(f"Error: {result.error}")
        sys.exit(1)
