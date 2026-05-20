"""
Document parser using docling.
Extracts: paragraphs, tables, code blocks.
Skips: images, footers, headers, captions, page numbers, and other noise.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DocItemLabel


NOISE_LABELS = {
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.CAPTION,
    DocItemLabel.FOOTNOTE,
}

TEXT_LABELS = {
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TEXT,
    DocItemLabel.TITLE,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.LIST_ITEM,
}


@dataclass
class ParsedDocument:
    source: str
    elements: list[dict[str, Any]] = field(default_factory=list)

    def paragraphs(self) -> list[str]:
        return [e["content"] for e in self.elements if e["type"] == "paragraph"]

    def tables(self) -> list[Any]:
        return [e["content"] for e in self.elements if e["type"] == "table"]

    def code_blocks(self) -> list[str]:
        return [e["content"] for e in self.elements if e["type"] == "code"]


def _is_noise(text: str) -> bool:
    """Filter out short meaningless fragments."""
    stripped = text.strip()
    if not stripped:
        return True
    # single tokens that look like page numbers or stray labels
    if len(stripped) <= 3 and (stripped.isdigit() or stripped in {"-", "—", "|", "•"}):
        return True
    return False


def parse_document(path: str | Path) -> ParsedDocument:
    path = Path(path)
    options = PdfPipelineOptions(do_ocr=False, generate_page_images=False)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )

    result = converter.convert(str(path))
    doc = result.document
    parsed = ParsedDocument(source=str(path))

    for item, _ in doc.iterate_items():
        label = getattr(item, "label", None)

        # Skip images and noise labels
        if label == DocItemLabel.PICTURE:
            continue
        if label in NOISE_LABELS:
            continue

        # Code blocks
        if label == DocItemLabel.CODE:
            text = item.text if hasattr(item, "text") else str(item)
            if not _is_noise(text):
                parsed.elements.append({"type": "code", "content": text.strip()})
            continue

        # Tables — export as a list of rows (list[list[str]])
        if label == DocItemLabel.TABLE:
            rows = _extract_table(item)
            if rows:
                parsed.elements.append({"type": "table", "content": rows})
            continue

        # Text / paragraphs
        if label in TEXT_LABELS or label is None:
            text = item.text if hasattr(item, "text") else ""
            if not _is_noise(text):
                parsed.elements.append({"type": "paragraph", "content": text.strip()})

    return parsed


def _extract_table(table_item) -> list[list[str]]:
    """Convert a docling TableItem into a plain list-of-rows."""
    try:
        df = table_item.export_to_dataframe()
        rows = [df.columns.tolist()] + df.values.tolist()
        return [[str(cell) for cell in row] for row in rows]
    except Exception:
        pass

    # Fallback: raw cell grid
    try:
        grid = table_item.data.grid
        rows = []
        for row in grid:
            rows.append([cell.text if cell else "" for cell in row])
        return rows
    except Exception:
        return []


def print_parsed(parsed: ParsedDocument) -> None:
    print(f"=== Document: {parsed.source} ===\n")
    for i, el in enumerate(parsed.elements, 1):
        if el["type"] == "paragraph":
            print(f"[PARA {i}] {el['content']}\n")
        elif el["type"] == "code":
            print(f"[CODE {i}]")
            print(el["content"])
            print()
        elif el["type"] == "table":
            print(f"[TABLE {i}]")
            for row in el["content"]:
                print("  | " + " | ".join(row) + " |")
            print()


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python doc_parser.py <document_path>")
        sys.exit(1)

    parsed = parse_document(sys.argv[1])
    print_parsed(parsed)
    print(f"Summary: {len(parsed.paragraphs())} paragraphs, "
          f"{len(parsed.tables())} tables, "
          f"{len(parsed.code_blocks())} code blocks")

    out_path = Path(sys.argv[1]).stem + ".json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"source": parsed.source, "elements": parsed.elements}, f, indent=2, ensure_ascii=False)
    print(f"Output written to {out_path}")
