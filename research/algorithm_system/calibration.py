"""Within-source native-difficulty rank diagnostics; no cross-platform fit."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from difficulty import parse_models
from taxonomy import normalize_tags


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "private-learning/full-catalog/catalog-v1/catalog.v1.jsonl"
MODELS = ROOT / "private-learning/algorithm-system/atcoder-problem-models-2026-10-16-v1/problem-models.json"
LABEL_DIR = ROOT / "private-learning/full-catalog/labels-v1/merged"
TAXONOMY = Path(__file__).with_name("taxonomy.v1.json")
OUTPUT = Path(__file__).with_name("calibration.v1.json")
FIGURE = Path(__file__).with_name("difficulty-cdf.v1.svg")
SEED = 17
RESAMPLES = 500
LANDMARKS = (0.10, 0.25, 0.50, 0.75, 0.90)
PRIMARY_MIN_CONFIDENCE = 0.85
MIN_TOPIC_N = 20
LC_CATEGORY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}


def _value_order(value):
    if value in LC_CATEGORY_ORDER:
        return (0, LC_CATEGORY_ORDER[value])
    return (1, value)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile_nearest(values: list, p: float):
    """Nearest-rank percentile, preserving categorical levels and source values."""
    if not values:
        return None
    ordered = sorted(values, key=_value_order)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return ordered[index]


def bootstrap_landmarks(values: list, seed: int = SEED, resamples: int = RESAMPLES) -> dict:
    """Percentile intervals for within-source score quantiles, resampling rows."""
    observed = [value for value in values if value is not None]
    if not observed:
        return {}
    rng = random.Random(seed)
    draws = {p: [] for p in LANDMARKS}
    n = len(observed)
    for _ in range(resamples):
        sample = [observed[rng.randrange(n)] for _ in range(n)]
        for p in LANDMARKS:
            draws[p].append(percentile_nearest(sample, p))
    result = {}
    for p in LANDMARKS:
        point = percentile_nearest(observed, p)
        low = percentile_nearest(draws[p], 0.025)
        high = percentile_nearest(draws[p], 0.975)
        result[f"p{int(p * 100):02d}"] = {
            "estimate": point, "bootstrap_95_percentile_interval": [low, high],
            "resamples": resamples, "unit": "problem_record_with_replacement",
        }
    return result


def _cdf_lookup(values: list) -> dict:
    present = sorted((value for value in values if value is not None), key=_value_order)
    if not present:
        return {}
    intervals = {}
    n, i = len(present), 0
    while i < n:
        j = i + 1
        while j < n and present[j] == present[i]:
            j += 1
        lower, upper = i / n, j / n
        intervals[present[i]] = (lower, upper, (lower + upper) / 2)
        i = j
    return intervals


def cdf_summary(values: list) -> dict:
    return {str(key): {"lower": bounds[0], "upper": bounds[1], "mid": bounds[2]}
            for key, bounds in _cdf_lookup(values).items()}


def _primary_and_ancestors(row: dict, taxonomy: dict) -> tuple[set[str], set[str]]:
    normalized = normalize_tags(row["platform"], row.get("tags"), taxonomy)
    primary = {
        entry["concept_id"] for entry in normalized["mapped_concepts"]
        if entry["concept_id"] is not None
        and entry["relation"] == "equivalent"
        and entry["confidence"] >= PRIMARY_MIN_CONFIDENCE
    }
    # Keep hierarchy-derived categories separate from native primary concepts.
    ancestors = set(normalized["ancestors"])
    return primary, ancestors


def _midrank_lookup(values: list) -> dict:
    return _cdf_lookup(values)


def topic_rank_diagnostics(rows: list[dict], taxonomy: dict,
                           min_n: int = MIN_TOPIC_N) -> dict:
    """Measure rank movement within reviewed concept cohorts, without labels."""
    output = {}
    for platform in ("leetcode", "codeforces"):
        platform_rows = [r for r in rows if r["platform"] == platform and r["difficulty"] is not None]
        global_cdf = _midrank_lookup([r["difficulty"] for r in platform_rows])
        groups: dict[str, list] = {"primary_concept": defaultdict(list), "ancestor": defaultdict(list)}
        for row in platform_rows:
            primary, ancestors = _primary_and_ancestors(row["catalog_row"], taxonomy)
            for concept in primary:
                groups["primary_concept"][concept].append(row["difficulty"])
            for concept in ancestors:
                groups["ancestor"][concept].append(row["difficulty"])
        per_platform = {}
        for cohort_type, cohorts in groups.items():
            per_type = {}
            for concept, values in sorted(cohorts.items()):
                count = len(values)
                if count < min_n:
                    per_type[concept] = {"n": count, "status": "below_planning_minimum"}
                    continue
                topic_cdf = _midrank_lookup(values)
                shifts = []
                for value in values:
                    global_mid = global_cdf[value][2]
                    topic_mid = topic_cdf[value][2]
                    shifts.append(abs(topic_mid - global_mid))
                per_type[concept] = {
                    "n": count,
                    "absolute_global_vs_topic_midrank_shift": {
                        "median": percentile_nearest(sorted(shifts), 0.50),
                        "p90": percentile_nearest(sorted(shifts), 0.90),
                    },
                    "topic_native_value_cdf": cdf_summary(values),
                    "interpretation": "planning_diagnostic_not_calibration_ground_truth",
                }
            per_platform[cohort_type] = per_type
        output[platform] = per_platform
    return output


def _load_final_labels(merged_path: Path, adjudication_paths: list[Path]) -> list[dict]:
    adjudications = {}
    for path in adjudication_paths:
        for row in read_jsonl(path):
            adjudications[row["pair_id"]] = row
    pairs = []
    for row in read_jsonl(merged_path):
        final_label = row.get("label")
        if row.get("resolution") == "pending_adjudication":
            adjudicated = adjudications.get(row["pair_id"])
            final_label = adjudicated.get("label") if adjudicated else None
        pairs.append({"pair_id": row["pair_id"], "query": row["query"],
                      "target": row["target"], "label": final_label})
    return pairs


def _evaluation_summary(labels: list[dict], by_key: dict[str, dict]) -> tuple[dict, set[str]]:
    reserved = set()
    label_counts = Counter()
    strong = []
    cross_platform_strong = []
    observed_difficulty = []
    for pair in labels:
        q, t = pair["query"], pair["target"]
        reserved.update((q, t))
        label = pair["label"]
        label_counts[str(label)] += 1
        if label in (2, 3):
            strong.append(pair)
            if q in by_key and t in by_key and by_key[q]["platform"] != by_key[t]["platform"]:
                cross_platform_strong.append(pair)
                observed_difficulty.append(sum(
                    by_key[key]["difficulty"] is not None for key in (q, t)
                ))
    return ({
        "pair_records": len(labels), "final_label_counts": dict(sorted(label_counts.items())),
        "strong_labels_2_or_3": len(strong),
        "cross_platform_strong_pairs": len(cross_platform_strong),
        "cross_platform_strong_pairs_with_both_native_values": sum(n == 2 for n in observed_difficulty),
        "source": "existing AI-reviewed semantic relevance labels",
        "use": "evaluation_only; never difficulty truth or fit data",
        "difficulty_ordering_labels": 0,
    }, reserved)


def make_split_plan(rows: list[dict], evaluation_keys: set[str], seed: int = SEED) -> dict:
    """Stable, group-disjoint prospective split; evaluation endpoints are reserved."""
    by_key = {row["problem_key"]: row for row in rows}
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        # catalog representative_key carries only reviewed native identity aliases;
        # unverified duplicate-title candidates are deliberately not joined.
        groups[row.get("representative_key") or row["problem_key"]].append(row["problem_key"])
    members: dict[str, list[str]] = {"legacy_eval_only": [], "calibration": [],
                                    "model_selection": [], "final": []}
    group_counts = Counter()
    for group, keys in groups.items():
        if any(key in evaluation_keys for key in keys):
            partition = "legacy_eval_only"
        else:
            digest = hashlib.sha256(f"algorithm-difficulty-v1|{seed}|{group}".encode()).hexdigest()
            bucket = int(digest[:8], 16) % 100
            partition = "calibration" if bucket < 70 else ("model_selection" if bucket < 85 else "final")
        members[partition].extend(keys)
        group_counts[partition] += 1
    # Store digests and counts rather than publishing a second problem-key inventory.
    partitions = {}
    for name, keys in members.items():
        stable = "\n".join(sorted(keys)).encode()
        partitions[name] = {
            "records": len(keys), "identity_groups": group_counts[name],
            "problem_key_set_sha256": hashlib.sha256(stable).hexdigest(),
            "platform_counts": dict(sorted(Counter(by_key[key]["platform"] for key in keys).items())),
        }
    all_keys = set().union(*(set(keys) for keys in members.values()))
    overlap = sum(len(members[a]) + len(members[b]) - len(set(members[a]) | set(members[b]))
                  for i, a in enumerate(members) for b in list(members)[i + 1:])
    return {
        "schema": "difficulty-split-plan.v1", "seed": seed,
        "unit": "catalog_representative_key_group",
        "identity_group_rule": "reviewed representative_key aliases only; duplicate-title candidates remain separate",
        "assignment": "existing semantic-evaluation endpoints are eval-only; remaining groups use SHA-256 buckets 70/15/15",
        "partitions": partitions,
        "disjoint_record_overlap_count": overlap,
        "catalog_key_coverage": len(all_keys),
        "all_catalog_keys_partitioned_once": len(all_keys) == len(rows),
        "platform_source_disjointness": "not_claimed; partitions are within-platform problem groups, and source scales remain separate",
        "fit_status": "no supervised fit performed",
    }


def build_calibration(catalog_rows: list[dict], models: dict, taxonomy: dict,
                      labels: list[dict], *, seed: int = SEED,
                      resamples: int = RESAMPLES, min_topic_n: int = MIN_TOPIC_N) -> dict:
    rows = []
    for catalog_row in catalog_rows:
        platform = catalog_row["platform"]
        score = catalog_row.get("native_difficulty")
        if platform == "atcoder":
            task_id = catalog_row["problem_key"].partition(":")[2]
            score = models.get(task_id)
        rows.append({"problem_key": catalog_row["problem_key"], "platform": platform,
                     "difficulty": score, "catalog_row": catalog_row})
    by_key = {row["problem_key"]: row for row in rows}
    platform_output = {}
    for platform in ("leetcode", "codeforces", "atcoder"):
        values = [row["difficulty"] for row in rows if row["platform"] == platform]
        observed = [value for value in values if value is not None]
        platform_output[platform] = {
            "catalog_records": len(values), "observed_native_values": len(observed),
            "missing_native_values": len(values) - len(observed),
            "native_value_distinct_count": len(set(observed)),
            "native_value_cdf_midrank_intervals": cdf_summary(observed),
            "native_value_percentile_landmarks": bootstrap_landmarks(observed, seed, resamples),
            "scale_interpretation": {
                "leetcode": "ordinal native category only; Easy < Medium < Hard is the published category order",
                "codeforces": "native Codeforces rating points",
                "atcoder": "AtCoder Problems community estimate; not AtCoder-native and not Codeforces-equivalent",
            }[platform],
        }
    eval_summary, evaluation_keys = _evaluation_summary(labels, by_key)
    return {
        "schema": "difficulty-within-source-calibration.v1",
        "status": "descriptive_pre-fit_diagnostics_only",
        "resampling": {"seed": seed, "resamples": resamples, "unit": "problem_record_with_replacement",
                       "interval": "2.5th to 97.5th percentile of bootstrap quantile landmarks"},
        "topic_rule": {
            "taxonomy_version": taxonomy["version"], "primary_relation": "equivalent",
            "minimum_mapping_confidence": PRIMARY_MIN_CONFIDENCE,
            "ancestor_features_separate": True, "minimum_topic_observations": min_topic_n,
            "atcoder_topic_analysis": "unavailable: native tags are not present",
            "diagnostic": "absolute shift between global and topic-conditioned midrank; not truth error",
        },
        "platforms": platform_output,
        "topic_rank_diagnostics": topic_rank_diagnostics(rows, taxonomy, min_topic_n),
        "cross_platform_interpretation": "no quantile-to-equal-ability mapping; native source ranks are not assumed equivalent",
        "existing_labels": eval_summary,
        "split_plan": make_split_plan(catalog_rows, evaluation_keys, seed),
        "method_limits": [
            "Empirical ranks describe observed native distributions and do not establish comparable ability.",
            "Bootstrap intervals quantify finite-catalog sampling variation only; they do not model source or label uncertainty.",
            "Topic-conditioned rank shifts are descriptive and can reflect category composition, selection, or tag coverage.",
            "No difficulty ordering or target is present in the semantic relevance labels; no supervised calibration was fitted.",
            "Split partitions are prospective and problem-group disjoint; all current labeled-pair endpoints remain evaluation-only.",
        ],
    }


def write_cdf_figure(result: dict, output_path: Path = FIGURE) -> None:
    """Write a dependency-free, three-panel ECDF SVG with source-local x scales."""
    width, height = 1200, 470
    panel_lefts = (95, 475, 855)
    plot_top, plot_bottom, plot_width = 105, 370, 300
    panel_titles = {
        "leetcode": "LeetCode categories",
        "codeforces": "Codeforces ratings",
        "atcoder": "AtCoder Problems estimates",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        '<title>Within-source empirical difficulty distributions</title>',
        '<desc>Separate source-local empirical CDF panels. Vertical marks show tie intervals; no cross-platform score conversion is implied.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="600" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">Native difficulty distributions (separate scales)</text>',
    ]
    for platform, left in zip(("leetcode", "codeforces", "atcoder"), panel_lefts):
        record = result["platforms"][platform]
        intervals = record["native_value_cdf_midrank_intervals"]
        if platform == "leetcode":
            ordered = [(name, intervals[name]) for name in ("Easy", "Medium", "Hard") if name in intervals]
            x_values = list(range(len(ordered)))
            x_min, x_max = -0.3, 2.3
            tick_labels = [(i, name) for i, (name, _) in enumerate(ordered)]
        else:
            ordered = sorted(((float(name), rec) for name, rec in intervals.items()), key=lambda item: item[0])
            x_values = [item[0] for item in ordered]
            if not x_values:
                x_min, x_max = 0.0, 1.0
            else:
                x_min, x_max = x_values[0], x_values[-1]
                if x_min == x_max:
                    x_min, x_max = x_min - 0.5, x_max + 0.5
            tick_labels = [(0.0, str(int(x_min) if x_min.is_integer() else x_min)),
                           (1.0, str(int(x_max) if x_max.is_integer() else x_max))]
        sx = lambda value: left + (value - x_min) / (x_max - x_min) * plot_width
        sy = lambda value: plot_bottom - value * (plot_bottom - plot_top)
        parts.append(f'<text x="{left + plot_width / 2:.1f}" y="78" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold">{html.escape(panel_titles[platform])}</text>')
        # Horizontal grid and axes.
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            y = sy(fraction)
            parts.append(f'<path d="M {left} {y:.1f} H {left + plot_width}" stroke="#d9dee7" stroke-width="1"/>')
            parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#4b5563">{fraction:.2f}</text>')
        parts.append(f'<path d="M {left} {plot_top} V {plot_bottom} H {left + plot_width}" fill="none" stroke="#263445" stroke-width="1.2"/>')
        coords, tie_bars = [], []
        for index, (value, interval) in enumerate(ordered):
            x_value = index if platform == "leetcode" else value
            x, lower, upper, mid = sx(x_value), sy(interval["lower"]), sy(interval["upper"]), sy(interval["mid"])
            coords.append(f'{x:.2f},{mid:.2f}')
            tie_bars.append(f'M {x:.2f} {lower:.2f} V {upper:.2f}')
        if tie_bars:
            parts.append(f'<path d="{" ".join(tie_bars)}" stroke="#e08b37" stroke-width="0.7" opacity="0.62"/>')
            if coords:
                parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="#146c94" stroke-width="1.4"/>')
        for tick, label in tick_labels:
            x_value = x_min + tick * (x_max - x_min) if platform != "leetcode" else tick
            x = sx(x_value)
            parts.append(f'<path d="M {x:.1f} {plot_bottom} V {plot_bottom + 5}" stroke="#263445"/>')
            parts.append(f'<text x="{x:.1f}" y="{plot_bottom + 22}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#374151">{html.escape(label)}</text>')
        x_label = "Native category" if platform == "leetcode" else "Native difficulty units"
        parts.append(f'<text x="{left + plot_width / 2:.1f}" y="{plot_bottom + 48}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_label}</text>')
        parts.append(f'<text x="{left - 48}" y="{(plot_top + plot_bottom) / 2:.1f}" transform="rotate(-90 {left - 48} {(plot_top + plot_bottom) / 2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="12">Empirical CDF</text>')
        n = record["observed_native_values"]
        missing = record["missing_native_values"]
        parts.append(f'<text x="{left + plot_width / 2:.1f}" y="445" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4b5563">n={n:,}; missing={missing:,}</text>')
    parts.append('</svg>')
    Path(output_path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def run(catalog_path: Path = CATALOG, models_path: Path = MODELS,
        label_dir: Path = LABEL_DIR, taxonomy_path: Path = TAXONOMY,
        output_path: Path = OUTPUT, *, seed: int = SEED, resamples: int = RESAMPLES) -> dict:
    catalog_rows = read_jsonl(catalog_path)
    models = parse_models(models_path.read_bytes())
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    adjudication_dir = label_dir.parent / "adjudication"
    adjudication_paths = sorted(adjudication_dir.glob("adjudicate-*.jsonl"))
    labels = _load_final_labels(label_dir / "merged.jsonl", adjudication_paths)
    result = build_calibration(catalog_rows, models, taxonomy, labels, seed=seed, resamples=resamples)
    result["inputs"] = {
        "catalog_sha256": sha256_file(catalog_path),
        "atcoder_models_sha256": sha256_file(models_path),
        "taxonomy_sha256": sha256_file(taxonomy_path),
        "merged_labels_sha256": sha256_file(label_dir / "merged.jsonl"),
        "adjudication_sha256": {path.name: sha256_file(path) for path in adjudication_paths},
        "statement_text_read": False,
    }
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--labels", type=Path, default=LABEL_DIR)
    parser.add_argument("--taxonomy", type=Path, default=TAXONOMY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args()
    result = run(args.catalog, args.models, args.labels, args.taxonomy, args.output,
                 seed=args.seed, resamples=args.resamples)
    write_cdf_figure(result, args.figure)
    print(json.dumps({"output": str(args.output), "platforms": {
        p: {"catalog_records": r["catalog_records"], "observed_native_values": r["observed_native_values"],
            "missing_native_values": r["missing_native_values"]}
        for p, r in result["platforms"].items()},
        "labels": result["existing_labels"], "split": result["split_plan"]}, indent=2))


if __name__ == "__main__":
    main()
