"""Offline, label-aware evaluation of ranked statement-retrieval runs.

Old labels are AI-reviewed semantic judgments. Missing judgments remain unknown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from taxonomy import load_taxonomy, normalize_tags

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = ROOT / "research/full_catalog/labels.v1.jsonl"
KS = (10, 20, 50)
SEED = 17
RESAMPLES = 500
REVIEW_QUERIES_PER_PLATFORM = 100


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_runs(path: Path) -> dict[str, Path]:
    path = Path(path)
    if path.is_file():
        return {path.parent.name: path}
    direct = path / "top50.jsonl"
    if direct.exists():
        return {path.name: direct}
    found = sorted(path.glob("*/top50.jsonl"))
    if not found:
        raise ValueError("no method top50.jsonl files found")
    return {item.parent.name: item for item in found}


def _key(row: dict) -> str:
    return row.get("problem_key") or row.get("key") or row.get("dataset_key")


def _index_corpus(rows: list[dict]) -> tuple[dict[str, dict], dict[str, str]]:
    by_key, aliases, conflicts = {}, {}, set()
    for row in rows:
        key = _key(row)
        if not isinstance(key, str) or not key or key in by_key:
            raise ValueError("missing_or_duplicate_corpus_key")
        by_key[key] = row
        native = row.get("native_key")
        if native:
            if native in aliases and aliases[native] != key:
                conflicts.add(native)
            aliases[native] = key
    for native in conflicts:
        aliases.pop(native, None)
    return by_key, aliases


def _native(key: str, by_key: dict[str, dict], aliases: dict[str, str]) -> str:
    row = by_key.get(key) or by_key.get(aliases.get(key, ""))
    return (row.get("native_key") or key) if row else key


def _parse_rankings(rows: list[dict]) -> dict[str, list[str]]:
    output = {}
    for row in rows:
        query = row.get("query")
        if not isinstance(query, str) or query in output:
            raise ValueError("duplicate_or_invalid_query")
        results = row.get("results")
        if not isinstance(results, list) or len(results) > 50:
            raise ValueError("results_must_be_top50_list")
        targets = []
        for result in results:
            target = result[0] if isinstance(result, list) and result else result.get("target") if isinstance(result, dict) else None
            if not isinstance(target, str):
                raise ValueError("invalid_target")
            targets.append(target)
        if len(targets) != len(set(targets)) or query in targets:
            raise ValueError("duplicate_or_self_target")
        output[query] = targets
    return output


def _label_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    labels = {}
    for row in rows:
        pair = (row["query"], row["target"])
        if pair in labels or row.get("label") not in (0, 1, 2, 3, None) or type(row.get("label")) is bool:
            raise ValueError("duplicate_or_invalid_label")
        labels[pair] = row
    return labels


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _review_query_sample(queries: list[str], by_key: dict[str, dict],
                         seed: int, per_platform: int) -> tuple[set[str], dict]:
    groups = defaultdict(list)
    for query in queries:
        platform = by_key.get(query, {}).get("platform", "unknown")
        groups[platform].append(query)
    sampled = set()
    counts = {}
    for platform, members in sorted(groups.items()):
        ranked = sorted(members, key=lambda key: (
            hashlib.sha256(f"review-queries-v1|{seed}|{platform}|{key}".encode()).hexdigest(), key))
        chosen = ranked[:per_platform]
        sampled.update(chosen)
        counts[platform] = {"eligible_queries": len(members), "sampled_queries": len(chosen)}
    return sampled, counts


def _bootstrap(per_query: dict[str, float | None], seed: int = SEED,
               resamples: int = RESAMPLES) -> dict:
    observed = [value for value in per_query.values() if value is not None]
    if not observed:
        return {"mean": None, "query_count": 0, "bootstrap_95_interval": None}
    rng = random.Random(seed)
    n = len(observed)
    samples = sorted(sum(observed[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(resamples))
    lo = samples[max(0, math.floor(.025 * (resamples - 1)))]
    hi = samples[min(resamples - 1, math.ceil(.975 * (resamples - 1)))]
    return {"mean": _mean(observed), "query_count": n,
            "bootstrap_95_interval": [lo, hi], "resamples": resamples,
            "resampling_unit": "query_with_replacement; family_independence_not_claimed"}


def _ndcg_common(order: list[str], grades: dict[str, int], common: set[str]) -> float | None:
    filtered = [target for target in order if target in common]
    if not filtered:
        return None
    ideal = sorted((grades[target] for target in common), reverse=True)
    def dcg(items):
        return sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(items))
    denominator = dcg(ideal)
    return dcg([grades[target] for target in filtered]) / denominator if denominator else None


def _primary_tags(row: dict | None, taxonomy: dict) -> set[str] | None:
    if not row or not row.get("source_tags"):
        return None
    normalized = normalize_tags(row["platform"], row["source_tags"], taxonomy)
    tags = {entry["concept_id"] for entry in normalized["mapped_concepts"]
            if entry["relation"] == "equivalent" and entry["confidence"] >= .85}
    return tags or None


def _difficulty(row: dict | None):
    if not row:
        return None
    value = row.get("native_difficulty", row.get("source_difficulty"))
    if value is None or isinstance(value, bool):
        return None
    if row["platform"] == "leetcode":
        return {"Easy": 0, "Medium": 1, "Hard": 2}.get(value)
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def _pair_diagnostics(q: dict | None, t: dict | None,
                      features: dict[str, tuple[set[str] | None, object]]) -> dict:
    qfeatures = features.get(_key(q)) if q else None
    tfeatures = features.get(_key(t)) if t else None
    qtags = qfeatures[0] if qfeatures else None
    ttags = tfeatures[0] if tfeatures else None
    tag_jaccard = len(qtags & ttags) / len(qtags | ttags) if qtags and ttags else None
    qd = qfeatures[1] if qfeatures else None
    td = tfeatures[1] if tfeatures else None
    same = bool(q and t and q["platform"] == t["platform"])
    distance = abs(qd - td) if same and qd is not None and td is not None else None
    return {"tag_jaccard": tag_jaccard, "native_difficulty_distance": distance,
            "difficulty_platform": q["platform"] if distance is not None else None,
            "cross_platform_difficulty_distance": None}


def evaluate(corpus: list[dict], rankings: dict[str, list[dict]], labels: list[dict],
             taxonomy: dict | None = None, *, seed: int = SEED,
             resamples: int = RESAMPLES, review_limit: int = 1000,
             review_queries_per_platform: int = REVIEW_QUERIES_PER_PLATFORM) -> dict:
    taxonomy = taxonomy or load_taxonomy()
    by_key, aliases = _index_corpus(corpus)
    features = {key: (_primary_tags(row, taxonomy), _difficulty(row))
                for key, row in by_key.items()}
    methods = {name: _parse_rankings(rows) for name, rows in rankings.items()}
    if not methods:
        raise ValueError("no_methods")
    label_by_pair = _label_index(labels)
    for method, run in methods.items():
        for query, targets in run.items():
            native_targets = [_native(target, by_key, aliases) for target in targets]
            if len(native_targets) != len(set(native_targets)):
                raise ValueError(f"duplicate_native_target:{method}:{query}")
    query_sets = [set(run) for run in methods.values()]
    common_queries = sorted(set.intersection(*query_sets))
    if not common_queries:
        raise ValueError("no_common_queries")
    review_queries, review_counts = _review_query_sample(
        common_queries, by_key, seed, review_queries_per_platform)
    reports = {}
    candidate_rows = {}
    for k in KS:
        common_judged = {}
        for query in common_queries:
            query_native = _native(query, by_key, aliases)
            each = []
            for run in methods.values():
                each.append({_native(t, by_key, aliases) for t in run[query][:k]
                             if label_by_pair.get((query_native, _native(t, by_key, aliases)), {}).get("label") is not None})
            common_judged[query] = set.intersection(*each)
        k_reports = {}
        for method, run in methods.items():
            per_query = defaultdict(dict)
            totals = Counter()
            target_hits = Counter()
            platform_diversity = []
            tag_values, distances = [], defaultdict(list)
            for query in common_queries:
                order = run[query][:k]
                native_order = [_native(t, by_key, aliases) for t in order]
                qnative = _native(query, by_key, aliases)
                grades = {t: label_by_pair[(qnative, t)]["label"] for t in native_order
                          if (qnative, t) in label_by_pair and label_by_pair[(qnative, t)]["label"] is not None}
                judged = len(grades)
                positive = sum(value >= 2 for value in grades.values())
                deceptive = sum(label_by_pair[(qnative, target)].get("deceptive") is True
                                for target in grades)
                slots = len(order)
                unknown = slots - judged
                totals.update(slots=slots, judged=judged, unknown=unknown, positive=positive,
                              deceptive_flagged=deceptive,
                              judged_queries=bool(judged), common_judged_slots=len(common_judged[query]))
                per_query["observed_precision_judged"][query] = positive / judged if judged else None
                per_query["precision_lower_bound"][query] = positive / slots if slots else None
                per_query["precision_upper_bound"][query] = (positive + unknown) / slots if slots else None
                per_query["ndcg_common_judged"][query] = _ndcg_common(native_order, grades, common_judged[query])
                target_hits.update(order)
                platforms = {by_key[t]["platform"] for t in order if t in by_key}
                platform_diversity.append(len(platforms))
                for rank, target in enumerate(order, start=1):
                    diag = _pair_diagnostics(by_key.get(query), by_key.get(target), features)
                    if diag["tag_jaccard"] is not None:
                        tag_values.append(diag["tag_jaccard"])
                    if diag["native_difficulty_distance"] is not None:
                        distances[diag["difficulty_platform"]].append(diag["native_difficulty_distance"])
                    if k == 50 and query in review_queries:
                        pair = (qnative, _native(target, by_key, aliases))
                        old = label_by_pair.get(pair)
                        if old is None or old["label"] is None:
                            item = candidate_rows.setdefault(pair, {"query": query, "target": target,
                                "native_query": qnative, "native_target": pair[1],
                                "old_label": None, "old_label_source": old.get("label_source") if old else None,
                                "methods": {}, "diagnostics": diag})
                            item["methods"][method] = rank
            denom = totals["slots"]
            values = {name: _bootstrap(metric, seed, resamples) for name, metric in per_query.items()}
            k_reports[method] = {
                "query_count": len(common_queries), "retrieved_slots": denom,
                "judged_slots": totals["judged"], "unjudged_slots": totals["unknown"],
                "ai_deceptive_flagged_judged_slots": totals["deceptive_flagged"],
                "hard_negative_flagged_rate_among_judged": totals["deceptive_flagged"] / totals["judged"] if totals["judged"] else None,
                "hard_negative_rate_basis": "historical AI deceptive flag among judged retrieved slots; not independent human error rate",
                "judged_coverage": totals["judged"] / denom if denom else None,
                "observed_precision_judged": totals["positive"] / totals["judged"] if totals["judged"] else None,
                "precision_bounds": [totals["positive"] / denom, (totals["positive"] + totals["unknown"]) / denom] if denom else None,
                "ndcg_common_judged_query_count": sum(per_query["ndcg_common_judged"][q] is not None for q in common_queries),
                "common_judged_slot_count": totals["common_judged_slots"],
                "query_bootstrap": values,
                "hubness": {"distinct_targets": len(target_hits), "max_indegree": max(target_hits.values(), default=0),
                            "top_target_indegrees": target_hits.most_common(10)},
                "diversity": {"mean_distinct_target_platforms_per_query": _mean(platform_diversity)},
                "tag_consistency": {"eligible_pairs": len(tag_values), "mean_primary_concept_jaccard": _mean(tag_values)},
                "native_difficulty_distance": {platform: {"eligible_pairs": len(values), "mean_absolute_native_distance": _mean(values),
                    "units": "ordinal_category_steps" if platform == "leetcode" else "native_source_units"}
                    for platform, values in sorted(distances.items())},
                "cross_platform_difficulty_distance": "unavailable; native scales are not equated",
            }
        reports[str(k)] = k_reports
    queue = _review_queue(candidate_rows.values(), by_key, review_limit)
    duplicate_queue = _duplicate_review_queue(methods, review_queries, by_key, aliases,
                                               label_by_pair)
    return {"schema": "statement-method-evaluation.v1", "label_source": "historical_ai_reviewed_semantic_labels_not_gold",
            "label_records": len(labels), "label_nulls": sum(row["label"] is None for row in labels),
            "common_query_count": len(common_queries), "query_set_rule": "intersection_across_methods",
            "method_query_counts": {name: len(run) for name, run in methods.items()},
            "positive_rule": "label_2_or_3", "metrics": reports,
            "ndcg_rule": "same_explicit_judged_target_intersection_per_query_and_K; unjudged_removed_before_discounting; all_zero_gain_undefined",
            "precision_rule": "observed denominator judged slots only; bounds assign every unjudged slot negative or positive",
            "review_queue_sampling": {"seed": seed, "max_queries_per_platform": review_queries_per_platform,
                                      "method": "smallest_seeded_SHA256_query_hash_per_platform",
                                      "platform_counts": review_counts,
                                      "sampling_applies_only_to_review_queues": True},
            "review_queue": queue, "ambiguous_duplicate_review_queue": duplicate_queue,
            "limitations": ["Unjudged and abstained pairs are never negatives.",
                            "Historical AI-reviewed labels are semantic judgments, not human gold or difficulty-ordering labels.",
                            "Query bootstrap does not assert independent duplicate families.",
                            "Cross-platform difficulty differences are unavailable; within-source distances are descriptive."]}


def _review_queue(candidates, by_key: dict, limit: int) -> dict:
    strata = defaultdict(list)
    for item in candidates:
        q, t = by_key.get(item["query"]), by_key.get(item["target"])
        diag = item["diagnostics"]
        platform = f"{q['platform'] if q else 'unknown'}>{t['platform'] if t else 'unknown'}"
        distance = diag["native_difficulty_distance"]
        difficulty = "unavailable" if distance is None else "same_native_value" if distance == 0 else "different_native_value"
        tag = diag["tag_jaccard"]
        topic = "unknown" if tag is None else "shared" if tag > 0 else "disjoint"
        for method in item["methods"]:
            strata[(method, platform, difficulty, topic)].append(item)
    for group in strata.values():
        group.sort(key=lambda item: (min(item["methods"].values()), item["query"], item["target"]))
    selected, seen = [], set()
    keys = sorted(strata)
    while len(selected) < limit and any(strata.values()):
        for key in keys:
            if not strata[key]:
                continue
            item = strata[key].pop(0)
            pair = (item["native_query"], item["native_target"])
            if pair in seen:
                continue
            seen.add(pair)
            selected.append({k: v for k, v in item.items() if k != "diagnostics"} | {
                "stratum": {"method": key[0], "platform_pair": key[1], "difficulty": key[2], "topic": key[3]},
                "review_status": "unjudged; human_review_requested; no_label_assigned"})
            if len(selected) >= limit:
                break
    return {"candidate_pairs": len(candidates), "selected_pairs": len(selected), "selection": "deterministic_round_robin_over_method_platform_difficulty_topic_strata", "items": selected}


def _duplicate_review_queue(methods: dict[str, dict[str, list[str]]],
                            sampled_queries: set[str], by_key: dict[str, dict],
                            aliases: dict[str, str], labels: dict, limit: int = 200) -> dict:
    """Unverified overlap candidates from top-10 agreement, without score cutoffs."""
    hits = defaultdict(dict)
    for method, run in methods.items():
        for query in sampled_queries:
            qnative = _native(query, by_key, aliases)
            for rank, target in enumerate(run[query][:10], 1):
                tnative = _native(target, by_key, aliases)
                hits[(qnative, tnative)][method] = rank
    selected = []
    for (qnative, tnative), ranks in sorted(
            hits.items(), key=lambda item: (-len(item[1]), min(item[1].values()), item[0])):
        if len(ranks) < 2:
            continue
        q = by_key.get(qnative) or by_key.get(aliases.get(qnative, ""))
        t = by_key.get(tnative) or by_key.get(aliases.get(tnative, ""))
        if not q or not t or q["platform"] == t["platform"]:
            continue
        if q.get("content_sha256") and q.get("content_sha256") == t.get("content_sha256"):
            continue
        old = labels.get((qnative, tnative))
        selected.append({"query": qnative, "target": tnative,
                         "platform_pair": f"{q['platform']}>{t['platform']}",
                         "method_ranks": dict(sorted(ranks.items())),
                         "historical_ai_label": old.get("label") if old else None,
                         "status": "possible_semantic_overlap; human_duplicate_review_pending; no_merge_or_label"})
        if len(selected) >= limit:
            break
    return {"selection": "top10_target_agreement_in_at_least_two_methods_on_sampled_queries; no_score_threshold_or_duplicate_claim",
            "selected_pairs": len(selected), "items": selected}


def run(corpus_path: Path, run_path: Path, output_path: Path,
        labels_path: Path = DEFAULT_LABELS, *, seed: int = SEED,
        resamples: int = RESAMPLES) -> dict:
    paths = resolve_runs(run_path)
    result = evaluate(jsonl(corpus_path), {name: jsonl(path) for name, path in paths.items()},
                      jsonl(labels_path), seed=seed, resamples=resamples)
    result["inputs"] = {"corpus": {"path": str(corpus_path), "sha256": sha256(corpus_path)},
                        "labels": {"path": str(labels_path), "sha256": sha256(labels_path)},
                        "taxonomy": {"path": str(Path(__file__).with_name("taxonomy.v1.json")),
                                     "sha256": sha256(Path(__file__).with_name("taxonomy.v1.json"))},
                        "runs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args()
    result = run(args.corpus, args.run, args.output, args.labels)
    print(json.dumps({"output": str(args.output), "common_queries": result["common_query_count"],
                      "methods": list(result["method_query_counts"]), "review_queue": result["review_queue"]["selected_pairs"]}))


if __name__ == "__main__":
    main()
