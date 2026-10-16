"""Frozen-split training gates and deterministic feature scorer."""
from __future__ import annotations
import math
import re
from .research_contracts import FEATURES, REVIEWED_STATES, RESEARCH_VERSION, TOKENIZER_VERSION, ResearchError, json_bytes, sha256_bytes
from .research_profiles import Profile, TfidfIndex, feature_vector

def training_snapshot_sha256(*, dataset_sha256: str, split_sha256: str, training_curriculum_sha256: str,
                             training_annotations_sha256: str, evaluation_curriculum_sha256: str,
                             evaluation_annotations_sha256: str) -> str:
    training_config = {"scorer": "ridge-feature-v1", "loss": "ridge_mse", "alpha": 1.0, "seed": 17, "feature_names": list(FEATURES)}
    snapshot = {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
        "training_curriculum_sha256": training_curriculum_sha256,
        "training_annotations_sha256": training_annotations_sha256,
        "evaluation_curriculum_sha256": evaluation_curriculum_sha256,
        "evaluation_annotations_sha256": evaluation_annotations_sha256,
        "config_sha256": sha256_bytes(json_bytes(training_config)),
        "reserved_evaluation_sha256": "3fe417325e608a56d374f9d5187335de965585be5509eb028aa0f2ef2131c6e3",
    }
    return sha256_bytes(json_bytes(snapshot))


def training_gate(dataset: dict, split: dict, profiles: dict[str, Profile], evaluation_profiles: dict[str, Profile] | None = None,
                  approval_receipt: dict | None = None, dataset_sha256: str | None = None,
                  expected_snapshot_sha256: str | None = None) -> dict:
    rows = dataset.get("judgments", [])
    permitted = isinstance(dataset.get("permitted_uses"), list) and "training" in dataset["permitted_uses"]
    reviewed = [row for row in rows if isinstance(row, dict) and row.get("adjudication") in REVIEWED_STATES and
                isinstance(row.get("adjudicator") or row.get("reviewer"), str) and
                bool((row.get("adjudicator") or row.get("reviewer")).strip()) and
                isinstance(row.get("evidence"), str) and bool(row["evidence"].strip())]
    problems = {key for row in reviewed for key in (row.get("source"), row.get("target")) if isinstance(key, str)}
    families = {row.get("family") for row in reviewed if isinstance(row.get("family"), str)}
    platform_pairs = {row.get("platform_pair") for row in reviewed if isinstance(row.get("platform_pair"), str)}
    train_keys = set(split.get("training_problem_keys", [])) if isinstance(split.get("training_problem_keys"), list) else set()
    eval_keys = set(split.get("evaluation_problem_keys", [])) if isinstance(split.get("evaluation_problem_keys"), list) else set()
    train_families = set(split.get("training_families", [])) if isinstance(split.get("training_families"), list) else set()
    eval_families = set(split.get("evaluation_families", [])) if isinstance(split.get("evaluation_families"), list) else set()
    evaluation_profiles = evaluation_profiles or {}
    training_config = {"scorer": "ridge-feature-v1", "loss": "ridge_mse", "alpha": 1.0, "seed": 17, "feature_names": list(FEATURES)}
    receipt_bound = bool(
        isinstance(approval_receipt, dict) and approval_receipt.get("schema_version") == 1 and
        approval_receipt.get("purpose") == "training" and approval_receipt.get("decision") == "approved" and
        isinstance(approval_receipt.get("reviewer_kind"), str) and approval_receipt["reviewer_kind"].strip() and
        isinstance(approval_receipt.get("reviewer_id"), str) and approval_receipt["reviewer_id"].strip() and
        isinstance(approval_receipt.get("review_id"), str) and approval_receipt["review_id"].strip() and
        isinstance(dataset_sha256, str) and approval_receipt.get("dataset_sha256") == dataset_sha256 and
        isinstance(approval_receipt.get("training_snapshot_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", approval_receipt["training_snapshot_sha256"]) is not None and
        expected_snapshot_sha256 is not None and approval_receipt["training_snapshot_sha256"] == expected_snapshot_sha256 and
        approval_receipt.get("config_sha256") == sha256_bytes(json_bytes(training_config)) and
        isinstance(approval_receipt.get("compute_owner"), str) and bool(approval_receipt["compute_owner"].strip()) and
        isinstance(approval_receipt.get("compute_budget_seconds"), (int, float)) and type(approval_receipt["compute_budget_seconds"]) is not bool and math.isfinite(approval_receipt["compute_budget_seconds"]) and approval_receipt["compute_budget_seconds"] > 0 and
        type(approval_receipt.get("peak_rss_budget_bytes")) is int and approval_receipt["peak_rss_budget_bytes"] > 0
    )
    # The initial 180-row artifact is evaluation-only by identity, even if a
    # caller edits its mutable permission field or dataset version.
    initial_eval_digest = "3fe417325e608a56d374f9d5187335de965585be5509eb028aa0f2ef2131c6e3"
    is_reserved_eval = dataset_sha256 == initial_eval_digest or dataset.get("dataset_version") == "relationship-judgments.v1"
    overlap_problems = sorted(train_keys & eval_keys)
    overlap_families = sorted(train_families & eval_families)
    endpoint_mismatch = sorted(problems - train_keys)
    gates = {
        "training_use_permission": permitted,
        "training_rights_reference": bool(dataset.get("rights_reference") or dataset.get("license_reference")),
        "external_training_approval_receipt": receipt_bound,
        "initial_evaluation_dataset_not_rebranded": not is_reserved_eval,
        "reviewed_rows_at_least_1000": len(reviewed) >= 1000,
        "similarity_only_first_scorer": all(row.get("type")=="similarity" for row in reviewed),
        "distinct_problems_at_least_300": len(problems) >= 300,
        "platform_pair_groups_at_least_3": len(platform_pairs) >= 3,
        "families_at_least_3": len(families) >= 3,
        "reviewed_families_in_frozen_training_split": families.issubset(train_families),
        "frozen_split_manifest": split.get("schema_version") == 1 and bool(split.get("split_id")) and split.get("frozen") is True,
        "no_problem_overlap": not overlap_problems,
        "no_family_overlap": not overlap_families,
        "judgments_within_training_split": not endpoint_mismatch and (not reviewed or bool(train_keys)),
        "training_endpoints_have_profiles": problems.issubset(profiles),
        "training_profiles_exclude_evaluation_keys": set(profiles).issubset(train_keys),
        "evaluation_profiles_reserved": set(evaluation_profiles).issubset(eval_keys),
    }
    return {
        "authorized_to_fit": all(gates.values()),
        "gates": gates,
        "counts": {"reviewed_rows": len(reviewed), "distinct_problem_keys": len(problems), "families": len(families), "platform_pair_groups": len(platform_pairs)},
        "problem_overlap": overlap_problems,
        "family_overlap": overlap_families,
        "training_endpoint_outside_split": endpoint_mismatch,
        "missing_profile_keys": sorted(problems - set(profiles)),
        "evaluation_profile_keys_outside_split": sorted(set(evaluation_profiles) - eval_keys),
        "approval_receipt_bound_to_snapshot": receipt_bound,
        "expected_training_snapshot_sha256": expected_snapshot_sha256,
        "initial_evaluation_dataset_reserved": is_reserved_eval,
        "training_config": training_config,
    }

def solve_ridge(rows: list[dict[str, float]], targets: list[float], *, alpha: float = 1.0) -> list[float]:
    """Solve a tiny normal equation system without NumPy or nondeterministic BLAS."""
    dimension = len(FEATURES)
    matrix = [[0.0] * dimension for _ in range(dimension)]
    vector = [0.0] * dimension
    for feature_row, target in zip(rows, targets, strict=True):
        values = [float(feature_row.get(name, 0.0)) for name in FEATURES]
        for i in range(dimension):
            vector[i] += values[i] * target
            for j in range(dimension):
                matrix[i][j] += values[i] * values[j]
    for index in range(dimension):
        matrix[index][index] += alpha
    for pivot in range(dimension):
        best = max(range(pivot, dimension), key=lambda index: (abs(matrix[index][pivot]), -index))
        if abs(matrix[best][pivot]) < 1e-12:
            raise ResearchError("singular ridge system")
        matrix[pivot], matrix[best] = matrix[best], matrix[pivot]
        vector[pivot], vector[best] = vector[best], vector[pivot]
        divisor = matrix[pivot][pivot]
        matrix[pivot] = [value / divisor for value in matrix[pivot]]
        vector[pivot] /= divisor
        for row in range(dimension):
            if row == pivot:
                continue
            factor = matrix[row][pivot]
            matrix[row] = [value - factor * pivot_value for value, pivot_value in zip(matrix[row], matrix[pivot], strict=True)]
            vector[row] -= factor * vector[pivot]
    return vector

def fit_feature_scorer(rows: list[dict], profiles: dict[str, Profile], *, alpha: float = 1.0) -> dict:
    if any(row.get("type")!="similarity" for row in rows):
        raise ResearchError("first feature scorer supports similarity only")
    index = TfidfIndex(profiles)
    features, labels = [], []
    for row in rows:
        source, target = profiles[row["source"]], profiles[row["target"]]
        values = feature_vector(source, target, index.pair_features(row["source"], row["target"]))
        features.append(values)
        labels.append(float(row["relevance"]))
    coefficients = solve_ridge(features, labels, alpha=alpha)
    return {"feature_names": list(FEATURES), "coefficients": coefficients, "intercept": 0.0, "loss": "ridge_mse", "alpha": alpha, "seed": 17}

def scorer_score(model: dict, source: str, target: str, profiles: dict[str, Profile], index: TfidfIndex) -> float:
    values = feature_vector(profiles[source], profiles[target], index.pair_features(source, target))
    return math.fsum(coefficient * values[name] for coefficient, name in zip(model["coefficients"], model["feature_names"], strict=True)) + float(model.get("intercept", 0.0))

def promotion_gate(*, candidate_ndcg: float | None, baseline_ndcg: float | None, paired_interval: list[float] | None,
                   eligibility_regressions: int | None, subgroup_failures: list[str], fine_tuned: bool = False,
                   base_encoder_ndcg: float | None = None, paired_cluster_count: int | None = None,
                   subgroup_coverage_complete: bool = False) -> dict:
    gain = candidate_ndcg - baseline_ndcg if candidate_ndcg is not None and baseline_ndcg is not None else None
    checks = {
        "gain_at_least_0_03": gain is not None and gain >= 0.03,
        "uncertainty_supports_gain": paired_interval is not None and len(paired_interval) == 2 and paired_interval[0] > 0 and paired_cluster_count is not None and paired_cluster_count >= 2,
        "no_eligibility_regressions": eligibility_regressions == 0,
        "no_unacceptable_subgroups": subgroup_coverage_complete and not subgroup_failures,
        "beats_base_encoder": not fine_tuned or (candidate_ndcg is not None and base_encoder_ndcg is not None and candidate_ndcg > base_encoder_ndcg),
    }
    return {"promote": all(checks.values()), "checks": checks, "absolute_ndcg_gain": gain}

def run_config() -> dict:
    return {
        "research_version": RESEARCH_VERSION,
        "tokenizer": TOKENIZER_VERSION,
        "metadata_weights": {"concept_jaccard": 0.60, "same_platform": 0.25, "same_native_difficulty": 0.15},
        "relevance_positive": 1,
        "directional_precision_relevance_threshold": 2,
        "recall_cutoff": 10,
        "ndcg_cutoff": 5,
        "ranking_min_judged_candidates": 2,
        "directional_cutoff": 5,
        "bootstrap_seed": 17,
        "bootstrap_samples": 1000,
        "feature_names": list(FEATURES),
        "training_loss": "ridge_mse",
        "training_alpha": 1.0,
        "training_seed": 17,
    }
