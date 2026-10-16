#!/usr/bin/env python3
"""Offline, non-overwriting relationship benchmark; no model or network calls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.learning.research import (  # noqa: E402
    MetadataProvider,
    ResearchError,
    TfidfProvider,
    build_profiles,
    catalog_hash,
    load_neighbor_provider,
    provenance,
    read_json,
    reviewed_judgments,
    run_config,
    score_judgments,
)


def write_new_json(path: Path, value: dict) -> None:
    if not path.parent.is_dir():
        raise ResearchError("output directory must already exist")
    content = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", type=Path, default=ROOT / "curriculum/seed.v1.json")
    parser.add_argument("--annotations", type=Path, default=ROOT / "curriculum/annotations.v1.json")
    parser.add_argument("--judgments", type=Path, default=ROOT / "research/relationship-judgments.v1.json")
    parser.add_argument("--neighbors", type=Path, help="optional precomputed neighbor JSON; no vectors are computed")
    parser.add_argument("--namespace", help="expected namespace required when neighbor artifact exists")
    parser.add_argument("--output", required=True, type=Path, help="new report path; existing files are never replaced")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    started = time.perf_counter()
    try:
        curriculum = read_json(args.curriculum)
        annotations = read_json(args.annotations)
        judgment_document = read_json(args.judgments)
        profiles = build_profiles(curriculum, annotations)
        accepted, admission = reviewed_judgments(judgment_document, purpose="evaluation")
        providers = [MetadataProvider(profiles), TfidfProvider(profiles)]
        neighbor_hash = catalog_hash(profiles)
        neighbor = load_neighbor_provider(args.neighbors, expected_namespace=args.namespace,
                                          expected_catalog_hash=neighbor_hash, profiles=profiles)
        if neighbor is not None:
            providers.append(neighbor)
        graph = curriculum.get("relationships", [])
        scores = score_judgments(accepted, profiles, providers, curriculum_relationships=graph)
        elapsed = time.perf_counter() - started
        config = run_config()
        config["methods"] = [provider.name for provider in providers]
        config["neighbor_namespace"] = neighbor.manifest["namespace"] if neighbor else None
        config["neighbor_status"] = "loaded" if neighbor else ("missing_or_not_supplied_metadata_fallback" if args.neighbors else "not_supplied")
        input_paths = [args.curriculum, args.annotations, args.judgments]
        if args.neighbors and args.neighbors.is_file():
            input_paths.append(args.neighbors)
        report = {
            "schema_version": 1,
            "report_type": "offline_relationship_benchmark",
            "status": "scored" if admission["rows_reviewed_eligible"] else "no_reviewed_labels",
            "diagnostic_only": True,
            "research_version": config["research_version"],
            "label_admission": admission,
            "catalog": {"problems": len(profiles), "input_sha256": neighbor_hash, "text_fields": ["annotation.title", "annotation.concept_rationale", "curriculum.problem.concepts"]},
            "review_boundary": "Accepted adjudicator labels are evaluation records; this report does not establish human gold quality or learner benefit.",
            "comparison_boundary": "Metadata and TF-IDF share judged candidates but use different input fields; treat their comparison as an input ablation, not an equal-information algorithm comparison.",
            "graph_leakage_boundary": "Judgment pairs present in the supplied curriculum relationship graph are excluded from held-out scoring regardless of relationship type.",
            "scoring": scores,
            "metrics_definition": {
                "recall_at_10": "graded labels with relevance > 0 retrieved in the top 10, divided by judged positive candidates",
                "ndcg_at_5": "graded gain 2^relevance-1 against the ideal ordering of judged candidates only",
                "precision_at_5": "prerequisite/transfer top-five precision at relevance >= 2",
                "direction_accuracy": "top-five forward-edge versus none classification; does not compare reverse orientations unless independently labeled",
            },
            "limitations": [
                "Unjudged candidates are excluded from each ranking and are not treated as negatives.",
                "Coverage is conditional on the reviewed candidate set; it is not full-catalog retrieval recall.",
                "Concept overlap with judgment evidence is reported as a diagnostic and is not causally corrected.",
                "Relationship ranking quality does not show improved learning or transfer.",
            ],
            "provenance": provenance(input_paths=input_paths, config=config, elapsed_seconds=elapsed,
                                     artifact_paths=[args.neighbors] if args.neighbors and args.neighbors.is_file() else []),
        }
        write_new_json(args.output, report)
        print(json.dumps({"output": str(args.output), "status": report["status"], "reviewed_labels": admission["rows_reviewed_eligible"], "metrics_queries": scores["query_count"]}, ensure_ascii=False))
        return 0
    except (ResearchError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"benchmark blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
