from __future__ import annotations

from dataclasses import dataclass, field
import re

import spacy
import yake

try:
    from llama_index.core.node_parser import SentenceSplitter
except ImportError:
    SentenceSplitter = None

from .importance import score_importance
from .parser import DocSection


@dataclass
class CoverageUnit:
    unit_id: str
    section_id: str
    section_title: str
    text: str
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    importance: float = 1.0
    section_level: int = 1
    block_type: str = "paragraph"
    table_schema: list[str] = field(default_factory=list)
    member_units: list[dict] = field(default_factory=list)


_NLP = None
_LLAMA_SENTENCE_SPLITTER = SentenceSplitter(chunk_size=128, chunk_overlap=0) if SentenceSplitter is not None else None
_KW_EXTRACTOR = yake.KeywordExtractor(lan="en", n=3, top=8)
FLAG_RE = re.compile(r"--[a-zA-Z0-9][a-zA-Z0-9_-]*")
CONFIG_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
FUNC_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\(\)")
HTTP_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\s+/\S+")
STATUS_RE = re.compile(r"\b[1-5][0-9]{2}\b")


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError("spaCy model 'en_core_web_sm' is required. Run: python -m spacy download en_core_web_sm") from exc
    return _NLP


def _normalize_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = item.strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _extract_keywords(text: str) -> list[str]:
    return _normalize_items([keyword for keyword, _ in _KW_EXTRACTOR.extract_keywords(text)])


def _extract_symbols(text: str) -> list[str]:
    found = FLAG_RE.findall(text)
    found += CONFIG_RE.findall(text)
    found += FUNC_RE.findall(text)
    found += [match.group(0) for match in HTTP_RE.finditer(text)]
    found += STATUS_RE.findall(text)
    return _normalize_items(found)


def _extract_entities(text: str) -> list[str]:
    doc = _get_nlp()(text)
    return _normalize_items([ent.text.strip() for ent in doc.ents])


def _sentence_units(text: str) -> list[str]:
    if _LLAMA_SENTENCE_SPLITTER is not None:
        chunks = [chunk.strip() for chunk in _LLAMA_SENTENCE_SPLITTER.split_text(text) if chunk.strip()]
        if chunks:
            return chunks
    doc = _get_nlp()(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _normalize_sentence_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" ,;")
    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _atomic_units(text: str) -> list[str]:
    sentences = [_normalize_sentence_text(sentence) for sentence in _sentence_units(text)]
    sentences = [sentence for sentence in sentences if sentence]
    return sentences or [_normalize_sentence_text(text)]


def extract_coverage_units(sections: list[DocSection]) -> list[CoverageUnit]:
    units: list[CoverageUnit] = []
    unit_counter = 0
    for section in sections:
        for block in section.content_blocks:
            source_texts: list[str]
            in_procedure_list = block.block_type == "numbered_list"
            table_schema: list[str] = []
            if block.block_type in {"paragraph", "bullet_list", "numbered_list"}:
                source_texts = _atomic_units(block.text) or [block.text]
            elif block.block_type == "table":
                source_texts = [" | ".join(row) for row in block.rows] if block.rows else [block.text]
                table_schema = block.rows[0] if block.rows else []
            else:
                source_texts = [block.text]
            for source_text in source_texts:
                unit_counter += 1
                entities = _extract_entities(source_text)
                symbols = _extract_symbols(source_text)
                keywords = _extract_keywords(source_text)
                importance = score_importance(
                    text=source_text,
                    entities=entities,
                    symbols=symbols,
                    in_procedure_list=in_procedure_list,
                )
                units.append(
                    CoverageUnit(
                        unit_id=f"unit_{unit_counter:04d}",
                        section_id=section.section_id,
                        section_title=section.title,
                        text=source_text,
                        keywords=keywords,
                        entities=entities,
                        symbols=symbols,
                        importance=importance,
                        section_level=section.level,
                        block_type=block.block_type,
                        table_schema=table_schema,
                    )
                )
    return units
