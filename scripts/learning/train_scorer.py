#!/usr/bin/env python3
"""Train the small research scorer only after all reviewed-data gates pass."""
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
    FEATURES,
    ResearchError,
    build_profiles,
    fit_feature_scorer,
    json_bytes,
    provenance,
    read_json,
    reviewed_judgments,
    sha256_bytes,
    sha256_file,
    training_gate,
    training_snapshot_sha256,
    validate_judgments,
)


def write_new_json(path: Path, value: dict) -> int:
    if not path.parent.is_dir():
        raise ResearchError("output directory must already exist")
    content = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return len(content)


def require_new_path(path: Path | None) -> None:
    if path is None:
        raise ResearchError("training gates passed but --output is required to create a scorer")
    if path.exists():
        raise ResearchError(f"refusing to overwrite existing output: {path}")
    if not path.parent.is_dir():
        raise ResearchError("output directory must already exist")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--approval-receipt", type=Path, help="separate approved training snapshot/config and owned-compute budget receipt")
    parser.add_argument("--training-curriculum", type=Path, default=ROOT / "curriculum/seed.v1.json")
    parser.add_argument("--training-annotations", type=Path, default=ROOT / "curriculum/annotations.v1.json")
    parser.add_argument("--evaluation-curriculum", type=Path, default=ROOT / "curriculum/seed.v1.json")
    parser.add_argument("--evaluation-annotations", type=Path, default=ROOT / "curriculum/annotations.v1.json")
    parser.add_argument("--output", type=Path, help="new scorer path; required only when all training gates pass")
    parser.add_argument("--report", type=Path, help="optional immutable JSON run or blocked-gate report")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    started = time.perf_counter()
    inputs = [args.training_data, args.split_manifest, args.training_curriculum,
              args.training_annotations, args.evaluation_curriculum, args.evaluation_annotations]
    try:
        dataset = read_json(args.training_data)
        split = read_json(args.split_manifest)
        train_catalog = read_json(args.training_curriculum)
        train_annotations = read_json(args.training_annotations)
        eval_catalog = read_json(args.evaluation_curriculum)
        eval_annotations = read_json(args.evaluation_annotations)
        receipt = read_json(args.approval_receipt) if args.approval_receipt else None
        if args.approval_receipt:
            inputs.append(args.approval_receipt)
        train_profiles = build_profiles(train_catalog, train_annotations)
        eval_profiles = build_profiles(eval_catalog, eval_annotations)
        snapshot_hash = training_snapshot_sha256(
            dataset_sha256=sha256_file(args.training_data),
            split_sha256=sha256_file(args.split_manifest),
            training_curriculum_sha256=sha256_file(args.training_curriculum),
            training_annotations_sha256=sha256_file(args.training_annotations),
            evaluation_curriculum_sha256=sha256_file(args.evaluation_curriculum),
            evaluation_annotations_sha256=sha256_file(args.evaluation_annotations),
        )
        gate = training_gate(dataset, split, train_profiles, eval_profiles, receipt,
                             sha256_file(args.training_data), snapshot_hash)
        if not gate["authorized_to_fit"]:
            report = {
                "schema_version": 1,
                "status": "blocked",
                "model_created": False,
                "gate": gate,
                "dataset_version": dataset.get("dataset_version"),
                "split_id": split.get("split_id"),
                "expected_training_snapshot_sha256": snapshot_hash,
                "boundary": "Blocked gates create no model. Training needs a separately approved snapshot/config receipt and owned-compute budget. Independent review provenance is not human empirical gold; the initial evaluation judgment file remains reserved by identity.",
                "provenance": provenance(input_paths=inputs, config={"scorer": "ridge-feature-v1", "loss": "ridge_mse", "alpha": 1.0, "seed": 17}, elapsed_seconds=time.perf_counter() - started, artifact_paths=inputs),
            }
            if args.report:
                write_new_json(args.report, report)
                report["report_path"] = str(args.report)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 2
        require_new_path(args.output)
        if args.report:
            require_new_path(args.report)
            if args.output.resolve() == args.report.resolve():
                raise ResearchError("model and run report must use different output paths")
        reviewed, admission = reviewed_judgments(dataset, purpose="training")
        reviewed = validate_judgments(reviewed, train_profiles)
        model = fit_feature_scorer(reviewed, train_profiles, alpha=1.0)
        model.update({
            "schema_version": 1,
            "model_type": "deterministic_ridge_feature_scorer",
            "namespace": "relationship-metadata-text-features-v1",
            "revision": "ridge-feature-v1",
            "license_reference": dataset.get("license_reference") or dataset.get("rights_reference"),
            "dimensions": len(FEATURES),
            "normalization": "raw bounded and cosine features; no vector normalization",
            "metric": "ridge regression relevance score",
            "preprocessing": "unicode-words-casefold-v1; curriculum concepts and authored annotation title/rationale only",
            "ordered_keys": sorted(train_profiles),
            "ordered_keys_sha256": sha256_bytes(json_bytes(sorted(train_profiles))),
            "input_sha256": {str(path): sha256_file(path) for path in inputs},
            "split_id": split["split_id"],
            "split_sha256": sha256_file(args.split_manifest),
            "training_snapshot_sha256": snapshot_hash,
            "training_rows": len(reviewed),
            "dataset_version": dataset.get("dataset_version"),
            "admission": admission,
        })
        model["run_provenance"] = provenance(
            input_paths=inputs,
            config={"scorer": "ridge-feature-v1", "loss": "ridge_mse", "alpha": 1.0, "seed": 17, "feature_names": list(FEATURES)},
            elapsed_seconds=time.perf_counter() - started,
            artifact_paths=inputs,
        )
        # Include the exact serialized artifact byte size without overwriting it.
        model["artifact_size_bytes"] = 0
        for _ in range(4):
            size = len((json.dumps(model, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
            if model["artifact_size_bytes"] == size:
                break
            model["artifact_size_bytes"] = size
        size = write_new_json(args.output, model)
        if size != model["artifact_size_bytes"]:
            raise ResearchError("internal artifact-size accounting mismatch")
        result = {"status": "trained_research_artifact", "model_created": True, "output": str(args.output), "training_rows": len(reviewed), "model_sha256": sha256_file(args.output), "artifact_size_bytes": size, "gate": gate}
        if args.report:
            write_new_json(args.report, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (ResearchError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "model_created": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
