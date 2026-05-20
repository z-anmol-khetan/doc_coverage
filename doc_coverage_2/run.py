from __future__ import annotations

import json
import sys
from pathlib import Path

import click

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doc_coverage_2.pipeline.clusterer import cluster_units
from doc_coverage_2.pipeline.contradiction import detect_contradictions
from doc_coverage_2.pipeline.extractor import extract_coverage_units
from doc_coverage_2.pipeline.matcher import load_qa_pairs, match_qa_pairs
from doc_coverage_2.pipeline.parser import parse_document
from doc_coverage_2.pipeline.report import build_results
from doc_coverage_2.pipeline.scorer import score_coverage


@click.command()
@click.option("--doc", "doc_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--qa", "qa_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--embedding-model", default=None, help="Preferred sentence-transformers model")
def main(doc_path: Path, qa_path: Path, output_path: Path, embedding_model: str | None) -> None:
    sections = parse_document(str(doc_path))
    units = [u for u in extract_coverage_units(sections) if u.importance >= 2.5]
    units = cluster_units(units, preferred_model=embedding_model)
    qa_pairs = load_qa_pairs(str(qa_path))
    match_result = match_qa_pairs(units, qa_pairs, preferred_model=embedding_model)
    scorecard = score_coverage(units, match_result)
    contradictions = detect_contradictions(units, match_result)
    results = build_results(sections, units, match_result, scorecard, contradictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    click.echo(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
