import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.learning.benchmark import write_new_json
from src.learning.relationships import NamespaceMismatchError, RelationshipProvider
from src.learning.research import (
    FEATURES,
    MetadataProvider,
    NamespaceMismatchError as ResearchNamespaceMismatchError,
    ResearchError,
    TfidfProvider,
    _query_metric,
    build_profiles,
    catalog_hash,
    paired_bootstrap,
    promotion_gate,
    reviewed_judgments,
    score_judgments,
    sha256_bytes,
    json_bytes,
    training_gate,
    training_snapshot_sha256,
)


ROOT = Path(__file__).resolve().parents[2]


def catalog_fixture():
    problems = [
        {"problem_key": "leetcode:source", "native_difficulty": "Easy", "concepts": ["window", "array"]},
        {"problem_key": "leetcode:close", "native_difficulty": "Easy", "concepts": ["window", "array"]},
        {"problem_key": "atcoder:far", "native_difficulty": "Beginner", "concepts": ["graph"]},
        {"problem_key": "codeforces:other", "native_difficulty": "800", "concepts": []},
        {"problem_key": "leetcode:unjudged", "native_difficulty": "Easy", "concepts": ["window"]},
    ]
    annotations = {"annotations": {
        "leetcode:source": {"title": "Sliding Window", "concept_rationale": "Tracks an array window."},
        "leetcode:close": {"title": "Window on Array", "concept_rationale": "Maintains a sliding array window."},
        "atcoder:far": {"title": "Graph Path", "concept_rationale": "Traverses graph nodes."},
        "codeforces:other": {"title": "Other", "concept_rationale": "Different task."},
        "leetcode:unjudged": {"title": "Window Array", "concept_rationale": "Uses a moving window."},
    }}
    return {"schema_version": 1, "problems": problems}, annotations


def judgment(target, relevance, *, relation="similarity", direction="source_to_target", family="family-a", pair=None):
    pair = pair or f"leetcode->{target.split(':', 1)[0]}"
    return {"id": f"j-{target}-{relation}", "source": "leetcode:source", "target": target, "type": relation,
            "direction": direction, "relevance": relevance, "family": family,
            "platform_pair": pair, "evidence": "window concept is shared", "adjudication": "accepted",
            "adjudicator": "reviewer-1"}


class ResearchTests(unittest.TestCase):
    def setUp(self):
        curriculum, annotations = catalog_fixture()
        self.curriculum = curriculum
        self.profiles = build_profiles(curriculum, annotations)

    def test_text_allowlist_and_repeatable_tfidf(self):
        annotations = {"annotations": {key: {"title": profile.title, "concept_rationale": profile.concept_rationale,
                                             "full_statement": "SECRET STATEMENT"}
                                       for key, profile in self.profiles.items()}}
        rebuilt = build_profiles(self.curriculum, annotations)
        self.assertNotIn("SECRET", " ".join(profile.text for profile in rebuilt.values()))
        first = TfidfProvider(self.profiles)
        second = TfidfProvider(self.profiles)
        scores = [first.score("leetcode:source", key) for key in self.profiles if key != "leetcode:source"]
        self.assertEqual(scores, [second.score("leetcode:source", key) for key in self.profiles if key != "leetcode:source"])
        self.assertGreater(first.score("leetcode:source", "leetcode:close"), first.score("leetcode:source", "atcoder:far"))

    def test_metrics_use_graded_judged_candidates_and_direction_threshold(self):
        labels = {
            "leetcode:close": judgment("leetcode:close", 3, relation="prerequisite"),
            "atcoder:far": judgment("atcoder:far", 0, relation="prerequisite", pair="leetcode->atcoder"),
            "codeforces:other": judgment("codeforces:other", 1, relation="prerequisite", pair="leetcode->codeforces"),
        }
        result = _query_metric(labels, ["leetcode:close", "codeforces:other", "atcoder:far"], "prerequisite")
        self.assertEqual(result["recall_at_10"], 1.0)
        self.assertEqual(result["ndcg_at_5"], 1.0)
        self.assertEqual(result["precision_at_5"], 1 / 3)
        self.assertEqual(result["direction_accuracy"], 1 / 3)
        self.assertEqual(result["direction_tested"], 3)

    def test_unjudged_candidates_are_not_inserted_or_called_negative(self):
        rows = [judgment("leetcode:close", 3), judgment("atcoder:far", 0), judgment("codeforces:other", 1)]
        result = score_judgments(rows, self.profiles, [MetadataProvider(self.profiles), TfidfProvider(self.profiles)], bootstrap_samples=50)
        targets = {row["problem_key"] for row in result["per_query"][0]["rankings"]["metadata"]}
        self.assertNotIn("leetcode:unjudged", targets)
        self.assertEqual(len(targets), 3)
        self.assertIn("unjudged candidates were excluded", result["candidate_coverage_note"])

    def test_seed_graph_overlap_is_removed_in_both_orientations_and_all_types(self):
        rows = [judgment("leetcode:close", 3), judgment("atcoder:far", 2)]
        graph = [{"source": "leetcode:close", "target": "leetcode:source", "type": "prerequisite"}]
        result = score_judgments(rows, self.profiles, [MetadataProvider(self.profiles)], curriculum_relationships=graph)
        self.assertEqual(result["seed_graph_overlap_rows_excluded"], 1)
        self.assertEqual(result["query_count"], 1)
        self.assertEqual(result["per_query"][0]["judged_candidates"], 1)

    def test_grouped_bootstrap_is_paired_and_seeded(self):
        values = {"q1": {"base": 0.4, "new": 0.6}, "q2": {"base": 0.6, "new": 0.8}, "q3": {"base": 0.2, "new": 0.5}}
        first = paired_bootstrap(values, baseline="base", comparison="new", metric="ndcg_at_5", samples=200, seed=7)
        second = paired_bootstrap(values, baseline="base", comparison="new", metric="ndcg_at_5", samples=200, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["delta_mean"], 0.7 / 3)
        self.assertGreater(first["interval_95"][0], 0)

    def test_bootstrap_resamples_review_families_as_clusters(self):
        values = {"q1": {"base": 0.0, "new": 0.2}, "q2": {"base": 0.0, "new": 0.4},
                  "q3": {"base": 0.0, "new": 0.8}}
        clusters = {"q1": "family-a", "q2": "family-a", "q3": "family-b"}
        result = paired_bootstrap(values, baseline="base", comparison="new", metric="ndcg_at_5",
                                  samples=100, seed=11, cluster_by=clusters)
        self.assertEqual(result["n_query_groups"], 3)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result["cluster_unit"], "review_family")
        self.assertAlmostEqual(result["delta_mean"], 0.55)

    def test_draft_and_evaluation_only_rows_do_not_enter_training(self):
        doc = {"permitted_uses": ["evaluation"], "judgments": [judgment("leetcode:close", 3)]}
        accepted, diagnostics = reviewed_judgments(doc, purpose="training")
        self.assertEqual(accepted, [])
        self.assertFalse(diagnostics["permitted"])
        self.assertEqual(diagnostics["rows_reviewed_eligible"], 0)

    def test_train_gate_blocks_permissions_sizes_overlap_and_profile_leaks(self):
        rows = [judgment("leetcode:close", 3)]
        dataset = {"permitted_uses": ["evaluation"], "judgments": rows}
        split = {"schema_version": 1, "split_id": "split-1", "frozen": True,
                 "training_problem_keys": ["leetcode:source", "leetcode:close"],
                 "evaluation_problem_keys": ["leetcode:source"],
                 "training_families": ["family-a"], "evaluation_families": ["family-a"]}
        result = training_gate(dataset, split, self.profiles, {"leetcode:source": self.profiles["leetcode:source"]})
        self.assertFalse(result["authorized_to_fit"])
        self.assertFalse(result["gates"]["training_use_permission"])
        self.assertFalse(result["gates"]["no_problem_overlap"])
        self.assertFalse(result["gates"]["no_family_overlap"])
        self.assertTrue(result["gates"]["training_endpoints_have_profiles"])

    def test_fabricated_training_permission_and_adjudicator_fields_are_not_authorization(self):
        row = judgment("leetcode:close", 3)
        dataset = {"dataset_version": "new-name", "permitted_uses": ["training"], "rights_reference": "local receipt",
                   "judgments": [row]}
        split = {"schema_version": 1, "split_id": "split-1", "frozen": True,
                 "training_problem_keys": ["leetcode:source", "leetcode:close"],
                 "evaluation_problem_keys": ["atcoder:far"],
                 "training_families": ["family-a"], "evaluation_families": ["family-b"]}
        result = training_gate(dataset, split, self.profiles, {"atcoder:far": self.profiles["atcoder:far"]},
                               approval_receipt=None, dataset_sha256="a" * 64)
        self.assertFalse(result["authorized_to_fit"])
        self.assertFalse(result["gates"]["external_training_approval_receipt"])

    def test_wrong_training_snapshot_receipt_stays_blocked(self):
        row = judgment("leetcode:close", 3)
        dataset = {"dataset_version": "new-name", "permitted_uses": ["training"], "rights_reference": "local receipt",
                   "judgments": [row]}
        split = {"schema_version": 1, "split_id": "split-1", "frozen": True,
                 "training_problem_keys": ["leetcode:source", "leetcode:close"],
                 "evaluation_problem_keys": ["atcoder:far"],
                 "training_families": ["family-a"], "evaluation_families": ["family-b"]}
        expected = training_snapshot_sha256(dataset_sha256="a" * 64, split_sha256="b" * 64,
                                            training_curriculum_sha256="c" * 64, training_annotations_sha256="d" * 64,
                                            evaluation_curriculum_sha256="e" * 64, evaluation_annotations_sha256="f" * 64)
        receipt = {"schema_version": 1, "purpose": "training", "decision": "approved", "reviewer_kind": "independent_review",
                   "reviewer_id": "reviewer-1", "review_id": "review-1", "dataset_sha256": "a" * 64,
                   "training_snapshot_sha256": "0" * 64, "config_sha256": "x", "compute_owner": "local",
                   "compute_budget_seconds": 10, "peak_rss_budget_bytes": 1024}
        result = training_gate(dataset, split, self.profiles, {"atcoder:far": self.profiles["atcoder:far"]},
                               receipt, "a" * 64, expected)
        self.assertFalse(result["gates"]["external_training_approval_receipt"])
        self.assertEqual(result["expected_training_snapshot_sha256"], expected)

    def test_initial_evaluation_dataset_cannot_be_rebranded_as_training(self):
        dataset = json.loads((ROOT / "research/relationship-judgments.v1.json").read_text(encoding="utf-8"))
        dataset["permitted_uses"] = ["training"]
        dataset["dataset_version"] = "renamed"
        split = {"schema_version": 1, "split_id": "split-1", "frozen": True,
                 "training_problem_keys": list(self.profiles), "evaluation_problem_keys": [],
                 "training_families": ["family-a"], "evaluation_families": []}
        result = training_gate(dataset, split, self.profiles, {}, dataset_sha256="3fe417325e608a56d374f9d5187335de965585be5509eb028aa0f2ef2131c6e3")
        self.assertTrue(result["gates"]["initial_evaluation_dataset_not_rebranded"] is False)

    def test_missing_optional_neighbors_falls_back_without_network(self):
        provider = RelationshipProvider(self.curriculum, {"annotations": {key: {"title": profile.title, "concept_rationale": profile.concept_rationale} for key, profile in self.profiles.items()}},
                                        neighbor_path="/definitely/missing/neighbors.json", expected_namespace="optional-v1")
        self.assertEqual(provider.status, "metadata_fallback_missing_artifact")
        self.assertEqual(provider.candidates("leetcode:source")[0]["namespace"], provider.namespace)

    def test_neighbor_namespace_mismatch_is_a_hard_error(self):
        profiles_hash = catalog_hash(self.profiles)
        ordered = sorted(self.profiles)
        artifact = {"manifest": {"schema_version": 1, "namespace": "space-a", "revision": "r1", "license_reference": "local receipt",
                                 "dimensions": 3, "normalization": "l2", "metric": "cosine", "preprocessing": "v1",
                                 "input_sha256": profiles_hash, "ordered_keys": ordered, "ordered_keys_sha256": sha256_bytes(json_bytes(ordered))},
                    "neighbors": {"leetcode:source": [{"problem_key": "leetcode:close", "score": 0.8}]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neighbors.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(NamespaceMismatchError):
                RelationshipProvider(self.curriculum, {"annotations": {key: {"title": profile.title, "concept_rationale": profile.concept_rationale} for key, profile in self.profiles.items()}},
                                      neighbor_path=path, expected_namespace="space-b")

    def test_missing_query_in_optional_artifact_falls_back_to_metadata(self):
        profiles_hash = catalog_hash(self.profiles)
        ordered = sorted(self.profiles)
        artifact = {"manifest": {"schema_version": 1, "namespace": "space-a", "revision": "r1", "license_reference": "fixture",
                                 "dimensions": 3, "normalization": "l2", "metric": "cosine", "preprocessing": "v1",
                                 "input_sha256": profiles_hash, "ordered_keys": ordered, "ordered_keys_sha256": sha256_bytes(json_bytes(ordered))},
                    "neighbors": {"leetcode:close": [{"problem_key": "leetcode:source", "score": 0.8}]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neighbors.json"
            path.write_text(json.dumps(artifact))
            provider = RelationshipProvider(self.curriculum, {"annotations": {key: {"title": p.title,"concept_rationale": p.concept_rationale} for key,p in self.profiles.items()}}, neighbor_path=path, expected_namespace="space-a")
            candidates = provider.candidates("leetcode:source")
            self.assertTrue(candidates)
            self.assertEqual(candidates[0]["status"],"metadata_fallback_missing_query")
            self.assertNotEqual(candidates[0]["namespace"],"space-a")

    def test_synthetic_ridge_algebra_and_similarity_task_boundary(self):
        from src.learning.research import solve_ridge, fit_feature_scorer, scorer_score, TfidfIndex
        coefficients=solve_ridge([{name:1.0 if i==0 else 0.0 for i,name in enumerate(FEATURES)}],[2.0],alpha=1.0)
        self.assertAlmostEqual(coefficients[0],1.0)
        self.assertTrue(all(value==0 for value in coefficients[1:]))
        rows=[judgment("leetcode:close",3),judgment("atcoder:far",0)]
        model=fit_feature_scorer(rows,self.profiles)
        index=TfidfIndex(self.profiles)
        self.assertGreater(scorer_score(model,"leetcode:source","leetcode:close",self.profiles,index),scorer_score(model,"leetcode:source","atcoder:far",self.profiles,index))
        with self.assertRaisesRegex(ResearchError,"similarity only"):
            fit_feature_scorer([judgment("leetcode:close",3,relation="prerequisite")],self.profiles)

    def test_report_output_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_new_json(path, {"first": True})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"replacement": True})

    def test_promotion_gate_never_promotes_missing_uncertainty_or_subgroup_fails(self):
        result = promotion_gate(candidate_ndcg=0.8, baseline_ndcg=0.7, paired_interval=[0.02, 0.18],
                                eligibility_regressions=0, subgroup_failures=["codeforces->atcoder"])
        self.assertFalse(result["promote"])
        self.assertFalse(result["checks"]["no_unacceptable_subgroups"])
        self.assertFalse(result["checks"]["uncertainty_supports_gain"])
        missing = promotion_gate(candidate_ndcg=0.8, baseline_ndcg=0.7, paired_interval=None,
                                 eligibility_regressions=0, subgroup_failures=[])
        self.assertFalse(missing["promote"])

    def test_blocked_cli_prints_report_and_creates_no_model(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            split = base / "split.json"
            split.write_text(json.dumps({"schema_version": 1, "split_id": "not-reviewed", "frozen": True,
                                         "training_problem_keys": [], "evaluation_problem_keys": [],
                                         "training_families": [], "evaluation_families": []}), encoding="utf-8")
            report = base / "blocked.json"
            model = base / "never-created.json"
            process = subprocess.run([sys.executable, str(ROOT / "scripts/learning/train_scorer.py"),
                                      "--training-data", str(ROOT / "research/relationship-judgments.v1.json"),
                                      "--split-manifest", str(split), "--report", str(report), "--output", str(model)],
                                     capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 2)
            self.assertTrue(report.is_file())
            self.assertFalse(model.exists())
            body = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(body["status"], "blocked")
            self.assertFalse(body["model_created"])


if __name__ == "__main__":
    unittest.main()
