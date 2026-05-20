from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Literal

from markdown_it import MarkdownIt


BlockType = Literal["paragraph", "bullet_list", "numbered_list", "code", "table"]


@dataclass
class ContentBlock:
    block_type: BlockType
    text: str
    items: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class DocSection:
    section_id: str
    title: str
    level: int
    content_blocks: list[ContentBlock] = field(default_factory=list)


class _SectionBuilder:
    def __init__(self) -> None:
        self.sections: list[DocSection] = []
        self._counter = 0
        self.current = self._new_section("Document", 1)

    def _new_section(self, title: str, level: int) -> DocSection:
        self._counter += 1
        section = DocSection(section_id=f"section_{self._counter:03d}", title=title.strip() or "Untitled", level=level)
        self.sections.append(section)
        return section

    def start_section(self, title: str, level: int) -> None:
        self.current = self._new_section(title, level)

    def add_block(self, block: ContentBlock | None) -> None:
        if block and (block.text.strip() or block.items or block.rows):
            self.current.content_blocks.append(block)

    def build(self) -> list[DocSection]:
        result = [s for s in self.sections if s.content_blocks or s.title != "Document"]
        # Merge consecutive bullet_list / numbered_list blocks of the same type
        # that result from block-level PDF extraction splitting each bullet
        for section in result:
            merged: list[ContentBlock] = []
            for block in section.content_blocks:
                if (
                    merged
                    and merged[-1].block_type == block.block_type
                    and block.block_type in ("bullet_list", "numbered_list")
                ):
                    merged[-1].items.extend(block.items)
                    merged[-1].text = "\n".join(merged[-1].items)
                else:
                    merged.append(block)
            section.content_blocks = merged
        return result


class _HTMLSectionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.builder = _SectionBuilder()
        self._tag_stack: list[str] = []
        self._current_heading: list[str] = []
        self._current_paragraph: list[str] = []
        self._current_code: list[str] = []
        self._current_li: list[str] = []
        self._list_kind: str | None = None
        self._list_items: list[str] = []
        self._table_row: list[str] = []
        self._table_rows: list[list[str]] = []
        self._table_cell: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        _ = attrs
        self._tag_stack.append(tag)
        if tag in {"ul", "ol"}:
            self._flush_paragraph()
            self._list_kind = "bullet_list" if tag == "ul" else "numbered_list"
            self._list_items = []
        elif tag == "pre":
            self._flush_paragraph()
            self._current_code = []
        elif tag == "table":
            self._flush_paragraph()
            self._table_rows = []
        elif tag == "tr":
            self._table_row = []
        elif tag in {"td", "th"}:
            self._table_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            title = " ".join(self._current_heading).strip()
            self._current_heading = []
            self.builder.start_section(title or "Untitled", int(tag[1]))
        elif tag == "p":
            self._flush_paragraph()
        elif tag == "li":
            text = " ".join(self._current_li).strip()
            self._current_li = []
            if text:
                self._list_items.append(_normalize_text(text))
        elif tag in {"ul", "ol"}:
            if self._list_items:
                self.builder.add_block(ContentBlock(block_type=self._list_kind or "bullet_list", text="\n".join(self._list_items), items=self._list_items.copy()))
            self._list_kind = None
            self._list_items = []
        elif tag == "pre":
            code = "\n".join(self._current_code).strip()
            if code:
                self.builder.add_block(ContentBlock(block_type="code", text=code))
            self._current_code = []
        elif tag in {"td", "th"}:
            cell = _normalize_text(" ".join(self._table_cell))
            self._table_row.append(cell)
            self._table_cell = []
        elif tag == "tr":
            if any(cell for cell in self._table_row):
                self._table_rows.append(self._table_row.copy())
            self._table_row = []
        elif tag == "table":
            if self._table_rows:
                table_text = "\n".join(" | ".join(row) for row in self._table_rows)
                self.builder.add_block(ContentBlock(block_type="table", text=table_text, rows=self._table_rows.copy()))
            self._table_rows = []
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if not data.strip():
            return
        current = self._tag_stack[-1] if self._tag_stack else ""
        if current in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._current_heading.append(data)
        elif current == "li":
            self._current_li.append(data)
        elif current in {"code", "pre"}:
            self._current_code.append(data)
        elif current in {"td", "th"}:
            self._table_cell.append(data)
        else:
            self._current_paragraph.append(data)

    def _flush_paragraph(self) -> None:
        text = _normalize_text(" ".join(self._current_paragraph))
        self._current_paragraph = []
        if text:
            self.builder.add_block(ContentBlock(block_type="paragraph", text=text))


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
BULLET_RE = re.compile(r"^[-*+]\s+(.*)$")
BULLET_UNICODE_RE = re.compile(r"^[•·▪▸►‣⁃◦]\s*(.+)$")
NUMBERED_RE = re.compile(r"^\d+[.)]\s+(.*)$")
SETEXT_RE = re.compile(r"^(=+|-+)\s*$")

# Matches PDF page-footer lines produced by PyMuPDF for this class of AMD/Xilinx docs.
# Pattern: "Chapter N: Title", version string, doc title, page number, "Send Feedback"
_PDF_FOOTER_RE = re.compile(
    r"^("
    r"Chapter\s+\d+[:\s]"           # "Chapter 1: Overview" / "Chapter 1 Overview"
    r"|Appendix\s+[A-Z][:\s]"       # "Appendix A: Primitives"
    r"|UG\d+\s*\(v"                 # "UG1273 (v2025.2)..."
    r"|Send\s+Feedback"             # "Send Feedback"
    r"|\s*\d+\s*$"                  # lone page number
    r")",
    re.IGNORECASE,
)

# Generic doc-title footer: long title lines that are identical across pages
# We detect these by collecting repeated lines across pages.

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _flush_markdown_paragraph(lines: list[str], builder: _SectionBuilder) -> None:
    text = _normalize_text(" ".join(lines))
    if text:
        builder.add_block(ContentBlock(block_type="paragraph", text=text))


def _parse_markdown(text: str) -> list[DocSection]:
    MarkdownIt().parse(text)
    lines = text.splitlines()
    builder = _SectionBuilder()
    paragraph_lines: list[str] = []
    code_lines: list[str] = []
    list_items: list[str] = []
    list_type: BlockType | None = None
    in_code = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()

        if stripped.startswith("```"):
            if paragraph_lines:
                _flush_markdown_paragraph(paragraph_lines, builder)
                paragraph_lines = []
            if list_items:
                builder.add_block(ContentBlock(block_type=list_type or "bullet_list", text="\n".join(list_items), items=list_items.copy()))
                list_items = []
                list_type = None
            if in_code:
                builder.add_block(ContentBlock(block_type="code", text="\n".join(code_lines).strip()))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            if paragraph_lines:
                _flush_markdown_paragraph(paragraph_lines, builder)
                paragraph_lines = []
            if list_items:
                builder.add_block(ContentBlock(block_type=list_type or "bullet_list", text="\n".join(list_items), items=list_items.copy()))
                list_items = []
                list_type = None
            builder.start_section(heading_match.group(2).strip(), len(heading_match.group(1)))
            i += 1
            continue

        if i + 1 < len(lines) and stripped and SETEXT_RE.match(lines[i + 1].strip()):
            if paragraph_lines:
                _flush_markdown_paragraph(paragraph_lines, builder)
                paragraph_lines = []
            level = 1 if lines[i + 1].strip().startswith("=") else 2
            builder.start_section(stripped.strip(), level)
            i += 2
            continue

        bullet_match = BULLET_RE.match(stripped)
        numbered_match = NUMBERED_RE.match(stripped)
        if bullet_match or numbered_match:
            if paragraph_lines:
                _flush_markdown_paragraph(paragraph_lines, builder)
                paragraph_lines = []
            next_type: BlockType = "bullet_list" if bullet_match else "numbered_list"
            if list_type and list_type != next_type and list_items:
                builder.add_block(ContentBlock(block_type=list_type, text="\n".join(list_items), items=list_items.copy()))
                list_items = []
            list_type = next_type
            list_items.append((bullet_match or numbered_match).group(1).strip())
            i += 1
            continue
        elif list_items:
            builder.add_block(ContentBlock(block_type=list_type or "bullet_list", text="\n".join(list_items), items=list_items.copy()))
            list_items = []
            list_type = None

        if "|" in stripped and i + 1 < len(lines) and re.match(r"^\s*\|?\s*[-:]+", lines[i + 1]):
            if paragraph_lines:
                _flush_markdown_paragraph(paragraph_lines, builder)
                paragraph_lines = []
            header = [cell.strip() for cell in stripped.strip("|").split("|")]
            separator_index = i + 1
            rows = [header]
            i = separator_index + 1
            while i < len(lines) and "|" in lines[i]:
                rows.append([cell.strip() for cell in lines[i].strip().strip("|").split("|")])
                i += 1
            builder.add_block(ContentBlock(block_type="table", text="\n".join(" | ".join(row) for row in rows), rows=rows))
            continue

        if not stripped.strip():
            if paragraph_lines:
                _flush_markdown_paragraph(paragraph_lines, builder)
                paragraph_lines = []
            i += 1
            continue

        paragraph_lines.append(stripped)
        i += 1

    if paragraph_lines:
        if len(paragraph_lines) == 1 and _looks_like_plain_heading(paragraph_lines[0]):
            if list_items:
                builder.add_block(ContentBlock(block_type=list_type or "bullet_list", text="\n".join(list_items), items=list_items.copy()))
                list_items = []
                list_type = None
            builder.start_section(paragraph_lines[0], 3)
            paragraph_lines = []
        else:
            _flush_markdown_paragraph(paragraph_lines, builder)
    if list_items:
        builder.add_block(ContentBlock(block_type=list_type or "bullet_list", text="\n".join(list_items), items=list_items.copy()))
    if code_lines:
        builder.add_block(ContentBlock(block_type="code", text="\n".join(code_lines).strip()))
    return builder.build()


# ---------------------------------------------------------------------------
# Heuristics for plain-text heading detection
# ---------------------------------------------------------------------------

# Known structural heading patterns found in AMD/Xilinx and similar tech docs
_STRUCTURAL_HEADING_RE = re.compile(
    r"^(Chapter\s+\d+[:\s-]|Appendix\s+[A-Z][:\s-]|Section\s+\d+[:\s-])",
    re.IGNORECASE,
)

def _looks_like_plain_heading(line: str) -> bool:
    """Return True only if *line* is very likely a section heading.

    Stricter than the original: avoids triggering on sentence fragments,
    parenthetical references like ``(UG1504).``, and mid-sentence lines
    ending with a colon that are continuation lines in PDF-extracted text.
    """
    clean = line.strip()
    if not clean:
        return False

    # Structural keywords are unambiguous headings
    if _STRUCTURAL_HEADING_RE.match(clean):
        return True

    # Must be short-ish (genuine headings rarely exceed ~8 words)
    words = clean.split()
    if len(words) > 8:
        return False

    # Reject lines that are clearly not headings:
    # - starts with punctuation or parenthesis (e.g. "(UG1504).")
    # - ends with a period (sentence ending, not a heading)
    # - starts with a lowercase letter (continuation fragment)
    if clean[0] in "(-.,;:)\"'":
        return False
    if clean.endswith("."):
        return False
    if clean[0].islower():
        return False

    # All-caps or title-case short line (3-8 words, no trailing period)
    letters = [c for c in clean if c.isalpha()]
    if not letters:
        return False

    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)

    # Accept if mostly uppercase (acronym-heavy headings like "AI ENGINE OVERVIEW")
    if upper_ratio > 0.65 and len(words) <= 8:
        return True

    # Accept "Title Case Heading Without Period" that is ≤6 words
    # Check: first letter of each word is uppercase (title case)
    is_title_case = all(w[0].isupper() for w in words if w[0].isalpha())
    if is_title_case and len(words) <= 6 and not clean.endswith((".", ",", ";")):
        return True

    return False


_CHAPTER_LABEL_RE = re.compile(r"^(Chapter\s+\d+|Appendix\s+[A-Z])$", re.IGNORECASE)


def _preprocess_pdf_lines(raw_lines: list[str], footer_lines: frozenset[str]) -> list[str]:
    """Clean and merge PDF-extracted lines before parsing.

    1. Drop footer/header lines.
    2. Merge isolated "Chapter N" label lines with the title line that follows,
       e.g. ["Chapter 1", "Overview"] → ["Chapter 1: Overview"].
    """
    # Drop footers first
    kept: list[str] = []
    for line in raw_lines:
        clean = line.strip()
        if clean in footer_lines:
            continue
        if _PDF_FOOTER_RE.match(clean):
            continue
        kept.append(line)

    # Merge "Chapter N" / "Appendix A" label with the following non-empty line
    merged: list[str] = []
    i = 0
    while i < len(kept):
        clean = kept[i].strip()
        if _CHAPTER_LABEL_RE.match(clean):
            # Look ahead for the next non-empty line
            j = i + 1
            while j < len(kept) and not kept[j].strip():
                j += 1
            if j < len(kept):
                next_clean = kept[j].strip()
                # Only merge if the next line looks like a title (short, no period)
                if next_clean and len(next_clean.split()) <= 8 and not next_clean.endswith("."):
                    merged.append(f"{clean}: {next_clean}")
                    i = j + 1
                    continue
        merged.append(kept[i])
        i += 1
    return merged


def _parse_plain_text(text: str, footer_lines: frozenset[str] | None = None) -> list[DocSection]:
    """Parse plain text (including PDF-extracted text) into DocSections.

    *footer_lines* is an optional set of exact line strings that should be
    silently dropped (page headers/footers identified during PDF extraction).
    """
    if footer_lines is None:
        footer_lines = frozenset()

    builder = _SectionBuilder()
    paragraph_lines: list[str] = []
    list_items: list[str] = []
    list_type: BlockType | None = None
    code_lines: list[str] = []
    in_code = False

    raw_lines = text.splitlines()
    lines = _preprocess_pdf_lines(raw_lines, footer_lines)

    for line in lines:
        stripped = line.rstrip()
        clean = stripped.strip()

        if clean.startswith("```"):
            if paragraph_lines:
                _flush_markdown_paragraph(paragraph_lines, builder)
                paragraph_lines = []
            if list_items:
                builder.add_block(ContentBlock(block_type=list_type or "bullet_list", text="\n".join(list_items), items=list_items.copy()))
                list_items = []
                list_type = None
            if in_code:
                builder.add_block(ContentBlock(block_type="code", text="\n".join(code_lines).strip()))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(stripped)
            continue

        # --- Unicode bullet characters from PDFs (•, ▪, etc.) ---
        unicode_bullet = BULLET_UNICODE_RE.match(clean)
        bullet_match = BULLET_RE.match(clean)
        numbered_match = NUMBERED_RE.match(clean)

        if unicode_bullet or bullet_match or numbered_match:
            if paragraph_lines:
                _flush_markdown_paragraph(paragraph_lines, builder)
                paragraph_lines = []
            next_type: BlockType = "numbered_list" if numbered_match else "bullet_list"
            if list_type and list_type != next_type and list_items:
                builder.add_block(ContentBlock(block_type=list_type, text="\n".join(list_items), items=list_items.copy()))
                list_items = []
            list_type = next_type
            if unicode_bullet:
                list_items.append(unicode_bullet.group(1).strip())
            elif bullet_match:
                list_items.append(bullet_match.group(1).strip())
            else:
                list_items.append(numbered_match.group(1).strip())
            continue

        # --- Continuation lines for the current bullet item ---
        # In PDF-extracted text, a bullet's wrapped lines appear as plain text
        # on subsequent lines with no bullet character.  We attach them to the
        # last item as long as the line is not blank, not a new bullet, and not
        # a heading.  "Note:" lines are treated as paragraph breaks, not continuation.
        if list_items and clean and not _looks_like_plain_heading(clean):
            # "Note:" lines break out of the list and become a separate paragraph
            if re.match(r"^Note\s*:", clean, re.IGNORECASE):
                builder.add_block(ContentBlock(block_type=list_type or "bullet_list", text="\n".join(list_items), items=list_items.copy()))
                list_items = []
                list_type = None
                paragraph_lines.append(clean)
                continue
            list_items[-1] = list_items[-1] + " " + clean
            continue

        if list_items and not clean:
            builder.add_block(ContentBlock(block_type=list_type or "bullet_list", text="\n".join(list_items), items=list_items.copy()))
            list_items = []
            list_type = None

        # --- Headings ---
        if _looks_like_plain_heading(clean):
            if paragraph_lines:
                _flush_markdown_paragraph(paragraph_lines, builder)
                paragraph_lines = []
            if list_items:
                builder.add_block(ContentBlock(block_type=list_type or "bullet_list", text="\n".join(list_items), items=list_items.copy()))
                list_items = []
                list_type = None
            builder.start_section(clean, 2)
            continue

        if not clean:
            if paragraph_lines:
                # Single-line paragraph that looks like a heading -> make it a section
                if len(paragraph_lines) == 1 and _looks_like_plain_heading(paragraph_lines[0]):
                    if list_items:
                        builder.add_block(ContentBlock(block_type=list_type or "bullet_list", text="\n".join(list_items), items=list_items.copy()))
                        list_items = []
                        list_type = None
                    builder.start_section(paragraph_lines[0], 3)
                    paragraph_lines = []
                else:
                    _flush_markdown_paragraph(paragraph_lines, builder)
                    paragraph_lines = []
            continue

        paragraph_lines.append(clean)

    if paragraph_lines:
        _flush_markdown_paragraph(paragraph_lines, builder)
    if list_items:
        builder.add_block(ContentBlock(block_type=list_type or "bullet_list", text="\n".join(list_items), items=list_items.copy()))
    if code_lines:
        builder.add_block(ContentBlock(block_type="code", text="\n".join(code_lines).strip()))
    return builder.build()


def _parse_html(text: str) -> list[DocSection]:
    parser = _HTMLSectionParser()
    parser.feed(text)
    parser.close()
    parser._flush_paragraph()
    return parser.builder.build()


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) <= 3 and (stripped.isdigit() or stripped in {"-", "—", "|", "•"}):
        return True
    return False


def _extract_table_docling(table_item) -> list[list[str]]:
    try:
        df = table_item.export_to_dataframe()
        rows = [df.columns.tolist()] + df.values.tolist()
        return [[str(cell) for cell in row] for row in rows]
    except Exception:
        pass
    try:
        grid = table_item.data.grid
        return [[cell.text if cell else "" for cell in row] for row in grid]
    except Exception:
        return []


def _parse_with_docling(path: Path) -> list[DocSection]:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import DocItemLabel

    options = PdfPipelineOptions(do_ocr=False, generate_page_images=False)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    doc = converter.convert(str(path)).document

    _NOISE = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.CAPTION, DocItemLabel.FOOTNOTE}

    builder = _SectionBuilder()
    list_items: list[str] = []
    list_type: BlockType = "bullet_list"

    def _flush_list() -> None:
        nonlocal list_items
        if list_items:
            builder.add_block(ContentBlock(block_type=list_type, text="\n".join(list_items), items=list_items.copy()))
            list_items = []

    for item, level in doc.iterate_items():
        label = getattr(item, "label", None)

        if label in _NOISE or label == DocItemLabel.PICTURE:
            continue

        text = (item.text if hasattr(item, "text") else "").strip()

        if label in {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}:
            _flush_list()
            if text:
                heading_level = int(getattr(item, "level", None) or level or 2)
                builder.start_section(text, heading_level)
            continue

        if label == DocItemLabel.CODE:
            _flush_list()
            if text:
                builder.add_block(ContentBlock(block_type="code", text=text))
            continue

        if label == DocItemLabel.TABLE:
            _flush_list()
            rows = _extract_table_docling(item)
            if rows:
                table_text = "\n".join(" | ".join(row) for row in rows)
                builder.add_block(ContentBlock(block_type="table", text=table_text, rows=rows))
            continue

        if label == DocItemLabel.LIST_ITEM:
            if text:
                list_items.append(text)
            continue

        if label in {DocItemLabel.PARAGRAPH, DocItemLabel.TEXT} or (label is None and text):
            _flush_list()
            if text and not _is_noise(text):
                builder.add_block(ContentBlock(block_type="paragraph", text=text))
            continue

    _flush_list()
    return builder.build()


def parse_document(doc_path: str) -> list[DocSection]:
    path = Path(doc_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_with_docling(path)
    text = path.read_text(encoding="utf-8")
    if suffix in {".md", ".markdown"}:
        return _parse_markdown(text)
    if suffix in {".html", ".htm"} or "<html" in text.lower():
        return _parse_html(text)
    return _parse_plain_text(text)
