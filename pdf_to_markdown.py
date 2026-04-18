#!/usr/bin/env python3
"""Convert a PDF file to Markdown.

Usage:
    python pdf_to_markdown.py input.pdf
    python pdf_to_markdown.py input.pdf -o output.md
    python pdf_to_markdown.py input.pdf --pages 1-3,5
    python pdf_to_markdown.py --interactive
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Iterable


def find_pdfs(base_dir: Path, recursive: bool) -> list[Path]:
    """Return sorted PDF files in the target directory."""
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted([p for p in base_dir.glob(pattern) if p.is_file()])


def choose_pdfs_interactively(base_dir: Path, recursive: bool) -> list[Path]:
    """Open a terminal checklist to choose one or more PDF files."""
    pdf_files = find_pdfs(base_dir, recursive)
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {base_dir}")

    try:
        inquirer_module = importlib.import_module("InquirerPy.inquirer")
    except ImportError as exc:
        raise RuntimeError(
            "Interactive mode requires InquirerPy. Install it with: pip install InquirerPy"
        ) from exc

    choices = [{"name": str(path.relative_to(base_dir)),
                "value": path} for path in pdf_files]
    selected = inquirer_module.checkbox(
        message="Select PDF file(s) to convert (SPACE to toggle, ENTER to confirm):",
        choices=choices,
        cycle=True,
        instruction="Use arrow keys to move.",
    ).execute()

    if not selected:
        raise ValueError("No PDF files selected.")

    return list(selected)


def parse_page_spec(page_spec: str, total_pages: int) -> list[int]:
    """Parse page specs like '1-3,5,8-10' into zero-based page indices."""
    selected: set[int] = set()

    for chunk in page_spec.split(","):
        part = chunk.strip()
        if not part:
            continue

        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start < 1 or end < 1 or start > end:
                raise ValueError(f"Invalid page range: '{part}'")
            for p in range(start, end + 1):
                if p <= total_pages:
                    selected.add(p - 1)
        else:
            p = int(part)
            if p < 1:
                raise ValueError(f"Invalid page number: '{part}'")
            if p <= total_pages:
                selected.add(p - 1)

    if not selected:
        raise ValueError("No valid pages were selected.")

    return sorted(selected)


def _page_to_markdown(page: Any) -> str:
    """Extract markdown if supported by PyMuPDF, otherwise fallback to plain text."""
    try:
        return page.get_text("markdown").strip()
    except (AssertionError, ValueError, TypeError):
        # Some PyMuPDF versions reject unknown formats with AssertionError.
        pass

    text = page.get_text("text").strip()
    return text


def pdf_to_markdown(input_pdf: Path, output_md: Path, pages: Iterable[int] | None = None) -> None:
    """Convert PDF pages to a markdown file."""
    fitz = importlib.import_module("fitz")
    with fitz.open(input_pdf) as doc:
        page_indices = list(
            pages) if pages is not None else list(range(len(doc)))
        chunks: list[str] = []

        for idx in page_indices:
            if idx < 0 or idx >= len(doc):
                continue
            page = doc[idx]
            md_text = _page_to_markdown(page)
            section = f"## Page {idx + 1}\n\n{md_text}" if md_text else f"## Page {idx + 1}\n"
            chunks.append(section)

    output_md.write_text(
        "\n\n---\n\n".join(chunks).rstrip() + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Path to input PDF file. Omit this when using --interactive.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to output markdown file. Defaults to input filename with .md extension.",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Optional page selection, e.g. '1-3,5,9'.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Open an interactive checklist to select one or more PDFs.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to search for PDFs in interactive mode (default: current directory).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for PDFs recursively in interactive mode.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        importlib.import_module("fitz")
    except ImportError:
        parser.error(
            "PyMuPDF is required. Install it with: pip install pymupdf")

    target_pdfs: list[Path]
    if args.interactive or args.input is None:
        if args.output is not None:
            parser.error(
                "--output is only allowed when converting a single input PDF.")
        try:
            target_pdfs = choose_pdfs_interactively(args.dir, args.recursive)
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    else:
        input_pdf = args.input
        if not input_pdf.exists() or input_pdf.suffix.lower() != ".pdf":
            parser.error("Input file must exist and be a .pdf file.")
        target_pdfs = [input_pdf]

    for input_pdf in target_pdfs:
        output_md = args.output if args.output else input_pdf.with_suffix(
            ".md")

        fitz = importlib.import_module("fitz")
        with fitz.open(input_pdf) as doc:
            total_pages = len(doc)

        selected_pages = None
        if args.pages:
            try:
                selected_pages = parse_page_spec(args.pages, total_pages)
            except ValueError as exc:
                parser.error(str(exc))

        pdf_to_markdown(input_pdf, output_md, selected_pages)
        print(f"Markdown saved to: {output_md}")


if __name__ == "__main__":
    main()
