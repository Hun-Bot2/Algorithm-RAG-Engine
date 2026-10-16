"""Judgment admission, leakage checks, ranking metrics, and uncertainty."""
from __future__ import annotations
from collections import Counter, defaultdict
import math
import random
from .research_contracts import RELATION_TYPES, REVIEWED_STATES, ResearchError, _clean_text, _tokens
from .research_profiles import Profile, TfidfIndex

def reviewed_judgments(document: dict, *, purpose: str = "evaluation") -> tuple[list[dict], dict]:
    permitted = document.get("permitted_uses", [])
    rows = document.get("judgments", [])
    if not isinstance(permitted, list) or not isinstance(rows, list):
        raise ResearchError("judgment document shape is invalid")
    state_counts = Counter(row.get("adjudication", "missing") for row in rows if isinstance(row, dict))
    accepted = []
    blocked_rows = 0
    if purpose in permitted:
        for row in rows:
            if not isinstance(row, dict):
                blocked_rows += 1
                continue
            state = row.get("adjudication")
            reviewer = row.get("adjudicator") or row.get("reviewer")
            evidence = row.get("evidence")
            if state not in REVIEWED_STATES or not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(evidence, str) or not evidence.strip():
                blocked_rows += 1
                continue
            accepted.append(row)
    diagnostics = {
        "purpose": purpose,
        "permitted": purpose in permitted,
        "rows_total": len(rows),
        "rows_reviewed_eligible": len(accepted),
        "rows_blocked_or_draft": blocked_rows + (len(rows) if purpose not in permitted else 0),
        "adjudication_counts": dict(sorted(state_counts.items())),
    }
    return accepted, diagnostics

def validate_judgments(rows: list[dict], profiles: dict[str, Profile]) -> list[dict]:
    result = []
    seen = set()
    seen_ids = set()
    for row in rows:
        required = {"id", "source", "target", "type", "relevance", "direction", "family", "platform_pair"}
        if not required.issubset(row):
            raise ResearchError("judgment row is missing required fields")
        source, target, relation = row["source"], row["target"], row["type"]
        if source not in profiles or target not in profiles or source == target:
            raise ResearchError("judgment endpoint missing from catalog or self-linked")
        if relation not in RELATION_TYPES:
            raise ResearchError("unsupported relationship type")
        if row["direction"] not in {"source_to_target", "target_to_source", "none"}:
            raise ResearchError("unsupported direction label")
        expected_pair = f"{source.split(':', 1)[0]}->{target.split(':', 1)[0]}"
        if row["platform_pair"] != expected_pair or not isinstance(row["family"], str) or not row["family"].strip():
            raise ResearchError("judgment family or platform pair is inconsistent")
        if not isinstance(row["id"], str) or not row["id"].strip():
            raise ResearchError("judgment id required")
        if row["id"] in seen_ids:
            raise ResearchError("duplicate judgment id")
        seen_ids.add(row["id"])
        relevance = row["relevance"]
        if type(relevance) is not int or relevance not in range(4):
            raise ResearchError("relevance must be an integer from 0 through 3")
        identity = (source, target, relation)
        if identity in seen:
            raise ResearchError("duplicate judgment pair and type")
        seen.add(identity)
        result.append(row)
    return result

def _dcg(values: list[int], cutoff: int) -> float:
    return math.fsum(((2 ** value) - 1) / math.log2(index + 2) for index, value in enumerate(values[:cutoff]))

def _query_metric(labels: dict[str, dict], ranked: list[str], relation: str, cutoff: int = 5,
                  direction_predictions: dict[str, str] | None = None) -> dict:
    relevance = {target: row["relevance"] for target, row in labels.items()}
    positives = sum(value > 0 for value in relevance.values())
    retrieved = ranked[:10]
    # A single judged candidate cannot test ranking; preserve pair evidence but
    # exclude it from retrieval metrics rather than awarding a perfect rank.
    rankable = len(labels) >= 2
    recall = sum(relevance.get(key, 0) > 0 for key in retrieved) / positives if positives and rankable else None
    ordered = [relevance.get(key, 0) for key in ranked]
    ideal = sorted(relevance.values(), reverse=True)
    dcg = _dcg(ordered, cutoff)
    idcg = _dcg(ideal, cutoff)
    ndcg = dcg / idcg if idcg and rankable else None
    row = {"recall_at_10": recall, "ndcg_at_5": ndcg}
    if relation in {"prerequisite", "transfer"}:
        top = ranked[:cutoff]
        row["precision_at_5"] = sum(relevance.get(key, 0) >= 2 for key in top) / len(top) if top else None
        direction_labeled = [key for key, label in labels.items() if label.get("direction") in {"source_to_target", "target_to_source", "none"}]
        predicted = set(top)
        row["direction_accuracy"] = sum(
            ((direction_predictions or {}).get(key, "source_to_target") if key in predicted else "none") ==
            (labels[key]["direction"] if labels[key]["relevance"] >= 2 else "none")
            for key in direction_labeled
        ) / len(direction_labeled) if direction_labeled else None
        row["direction_tested"] = len(direction_labeled)
    return row

def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return math.fsum(present) / len(present) if present else None

def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

def paired_bootstrap(by_query: dict[str, dict[str, float | None]], *, baseline: str, comparison: str, metric: str, samples: int = 1000, seed: int = 17, cluster_by: dict[str, str] | None = None, cluster_unit: str | None = None) -> dict:
    cluster_label = cluster_unit or ("review_family" if cluster_by is not None else "query")
    keys = sorted(key for key, values in by_query.items() if values.get(baseline) is not None and values.get(comparison) is not None)
    if len(keys) < 2:
        return {"n_query_groups": len(keys), "n_clusters": 0, "cluster_unit": cluster_label, "delta_mean": None, "interval_95": None, "status": "insufficient_query_groups"}
    differences_by_key = {key: float(by_query[key][comparison]) - float(by_query[key][baseline]) for key in keys}
    clusters = defaultdict(list)
    for key, difference in differences_by_key.items():
        clusters[cluster_by.get(key, key) if cluster_by is not None else key].append(difference)
    cluster_differences = {key: math.fsum(values) / len(values) for key, values in clusters.items()}
    cluster_keys = sorted(cluster_differences)
    if len(cluster_keys) < 2:
        return {"n_query_groups": len(keys), "n_clusters": len(cluster_keys), "cluster_unit": cluster_label, "delta_mean": None, "interval_95": None, "status": "insufficient_clusters"}
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    differences = list(cluster_differences.values())
    rng = random.Random(seed)
    boot = [math.fsum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences) for _ in range(samples)]
    return {
        "n_query_groups": len(keys),
        "n_clusters": len(cluster_keys),
        "cluster_unit": cluster_label,
        "delta_mean": math.fsum(differences) / len(differences),
        "interval_95": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
        "samples": samples,
        "seed": seed,
        "status": "computed",
    }

def score_judgments(rows: list[dict], profiles: dict[str, Profile], providers: list[object], *, curriculum_relationships: list[dict] | None = None, bootstrap_samples: int = 1000, bootstrap_seed: int = 17) -> dict:
    if not rows:
        return {"query_count": 0, "metrics": {}, "per_query": [], "paired_uncertainty": {}, "subgroups": {}, "candidate_coverage": None}
    rows = validate_judgments(rows, profiles)
    graph_pairs = {frozenset((row.get("source"), row.get("target"))) for row in curriculum_relationships or []}
    excluded_graph_rows = [row for row in rows if frozenset((row["source"], row["target"])) in graph_pairs]
    rows = [row for row in rows if frozenset((row["source"], row["target"])) not in graph_pairs]
    if not rows:
        return {"query_count": 0, "metrics": {}, "per_query": [], "paired_uncertainty": {}, "subgroups": {}, "candidate_coverage": None, "seed_graph_overlap_rows_excluded": len(excluded_graph_rows)}
    index = TfidfIndex(profiles)
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    row_by_id: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        grouped[(row["source"], row["type"])][row["target"]] = row
        row_by_id[(row["source"], row["target"], row["type"])] = row
    method_names = [provider.name for provider in providers]
    per_query = []
    query_values: dict[str, dict[str, dict]] = defaultdict(dict)
    subgroup_query_labels: dict[tuple[str, str, str, str], dict[str, dict]] = defaultdict(dict)
    source_by_query: dict[str, str] = {}
    for (source, relation), labels in sorted(grouped.items()):
        ranked_by_method = {}
        for provider in providers:
            ranked = sorted(labels, key=lambda target: (-provider.score(source, target, index), target))
            ranked_by_method[provider.name] = ranked
            query_key = f"{source}|{relation}"
            source_by_query[query_key] = source
            directions = {target: provider.direction(source, target) for target in labels} if callable(getattr(provider, "direction", None)) else {}
            query_values[provider.name][query_key] = _query_metric(labels, ranked, relation, direction_predictions=directions)
        query = {"source": source, "type": relation, "judged_candidates": len(labels), "positive_candidates": sum(row["relevance"] > 0 for row in labels.values()), "rankable": len(labels) >= 2, "rankings": {name: [{"problem_key": key, "rank": rank, "score": providers[method_names.index(name)].score(source, key, index), "relevance": labels[key]["relevance"], "direction": labels[key]["direction"]} for rank, key in enumerate(ranked, 1)] for name, ranked in ranked_by_method.items()}, "metrics": {name: query_values[name][f"{source}|{relation}"] for name in method_names}}
        per_query.append(query)
        for row in labels.values():
            subgroup_key = (row["platform_pair"], row["family"], source, relation)
            subgroup_query_labels[subgroup_key][row["target"]] = row
    metric_names = ["recall_at_10", "ndcg_at_5", "precision_at_5", "direction_accuracy"]
    metrics = {}
    for method in method_names:
        values = [item for query in query_values[method].values() for item in [query]]
        metrics[method] = {metric: _mean([value.get(metric) for value in values]) for metric in metric_names if any(metric in value for value in values)}
        metrics[method]["query_groups"] = len(values)
        metrics[method]["ranking_query_groups"] = sum(value.get("ndcg_at_5") is not None for value in values)
        metrics[method]["direction_tested"] = sum(int(value.get("direction_tested", 0)) for value in values)
    paired = {}
    if method_names:
        baseline = "metadata" if "metadata" in method_names else method_names[0]
        family_query_values: dict[str, dict[str, dict[str, float | None]]] = defaultdict(dict)
        family_by_key: dict[str, str] = {}
        for (platform_pair, family, source, relation), labels in sorted(subgroup_query_labels.items()):
            # Bootstrap clusters are review families; each source/family/type
            # subquery contributes equally within its family.
            key = f"{family}|{source}|{relation}|{platform_pair}"
            family_by_key[key] = family
            for provider in providers:
                ranked = sorted(labels, key=lambda target: (-provider.score(source, target, index), target))
                directions = {target: provider.direction(source, target) for target in labels} if callable(getattr(provider, "direction", None)) else {}
                family_query_values[provider.name][key] = _query_metric(labels, ranked, relation, direction_predictions=directions)
        for method in method_names:
            if method == baseline:
                continue
            paired[method] = {
                "by_source_problem": {metric: paired_bootstrap(
                    {key: {baseline: query_values[baseline].get(key, {}).get(metric), method: query_values[method].get(key, {}).get(metric)} for key in set(query_values[baseline]) | set(query_values[method])},
                    baseline=baseline, comparison=method, metric=metric, samples=bootstrap_samples, seed=bootstrap_seed,
                    cluster_by=source_by_query, cluster_unit="source_problem",
                ) for metric in ("recall_at_10", "ndcg_at_5")},
                "by_review_family": {metric: paired_bootstrap(
                    {key: {baseline: family_query_values[baseline].get(key, {}).get(metric), method: family_query_values[method].get(key, {}).get(metric)} for key in set(family_query_values[baseline]) | set(family_query_values[method])},
                    baseline=baseline, comparison=method, metric=metric, samples=bootstrap_samples, seed=bootstrap_seed,
                    cluster_by=family_by_key,
                ) for metric in ("recall_at_10", "ndcg_at_5")},
            }
    subgroup_values: dict[str, dict[str, list[float | None]]] = defaultdict(lambda: defaultdict(list))
    subgroup_counts: Counter[str] = Counter()
    subgroup_query_values: dict[str, dict[str, dict[str, dict[str, float | None]]]] = defaultdict(lambda: defaultdict(dict))
    for (platform_pair, family, source, relation), labels in sorted(subgroup_query_labels.items()):
        group_name = f"{platform_pair}|{family}"
        subgroup_counts[group_name] += len(labels)
        for provider in providers:
            ranked = sorted(labels, key=lambda target: (-provider.score(source, target, index), target))
            directions = {target: provider.direction(source, target) for target in labels} if callable(getattr(provider, "direction", None)) else {}
            values = _query_metric(labels, ranked, relation, direction_predictions=directions)
            for metric, value in values.items():
                if metric != "direction_tested":
                    subgroup_values[group_name][f"{provider.name}.{metric}"].append(value)
                    subgroup_query_values[group_name][provider.name].setdefault(f"{source}|{relation}", {})[metric] = value
    baseline = "metadata" if "metadata" in method_names else (method_names[0] if method_names else None)
    subgroups = {}
    for group, values_by_method in sorted(subgroup_values.items()):
        subgroups[group] = {"judgments": subgroup_counts[group],
                            "metrics": {key: _mean(values) for key, values in sorted(values_by_method.items())},
                            "paired_uncertainty": {}}
        if baseline:
            for method in method_names:
                if method == baseline:
                    continue
                subgroups[group]["paired_uncertainty"][method] = {metric: paired_bootstrap(
                    {key: {baseline: subgroup_query_values[group][baseline].get(key, {}).get(metric),
                           method: subgroup_query_values[group][method].get(key, {}).get(metric)}
                     for key in set(subgroup_query_values[group][baseline]) | set(subgroup_query_values[group][method])},
                    baseline=baseline, comparison=method, metric=metric, samples=bootstrap_samples,
                    seed=bootstrap_seed, cluster_by={key: key.split("|", 1)[0] for key in subgroup_query_values[group][baseline]},
                ) for metric in ("recall_at_10", "ndcg_at_5")}
    platforms = ["atcoder", "baekjoon", "codeforces", "leetcode"]
    pair_counts = Counter(row["platform_pair"] for row in rows)
    platform_pair_coverage = [{"platform_pair": f"{source}->{target}", "judgments": pair_counts[f"{source}->{target}"],
                               "status": "available" if pair_counts[f"{source}->{target}"] else "unavailable_no_reviewed_candidates"}
                              for source in platforms for target in platforms]
    mean_judged = _mean([float(len(labels)) / max(1, len(profiles) - 1) for labels in grouped.values()])
    directions = [row for row in rows if row.get("type") in {"prerequisite", "transfer"} and row.get("direction") in {"source_to_target", "target_to_source", "none"}]
    return {
        "query_count": len(grouped),
        "ranking_query_groups": sum(len(labels) >= 2 for labels in grouped.values()),
        "single_candidate_query_groups_excluded_from_ranking": sum(len(labels) < 2 for labels in grouped.values()),
        "metrics": metrics,
        "per_query": per_query,
        "paired_uncertainty": paired,
        "paired_uncertainty_cluster_unit": "source problem and review family; paired by the same evaluated query groups",
        "subgroups": subgroups,
        "platform_pair_coverage": platform_pair_coverage,
        "candidate_coverage": mean_judged,
        "candidate_coverage_note": "fraction of non-source catalog keys with a reviewed candidate label, macro-averaged by source and relationship type; unjudged candidates were excluded from ranking, not treated as negatives",
        "direction_coverage": {"reviewed_direction_rows": len(directions), "eligible_direction_rows": sum(row.get("type") in {"prerequisite", "transfer"} for row in rows), "accuracy_definition": "top-5 predicts source_to_target; a judged row is positive only when relevance >= 2 and its reviewed direction is source_to_target; all other labels are none", "accuracy": _mean([value.get("direction_accuracy") for key, value in query_values.get("metadata", {}).items() if key.rsplit("|", 1)[-1] in {"prerequisite", "transfer"}]) if directions else None, "reverse_direction_boundary": "This evaluates forward proposal versus none for the supplied ordered pair; it does not compare both orientations unless the dataset includes independently reviewed reverse candidates."},
        "directional_precision_threshold": 2,
        "concept_evidence_overlap": concept_evidence_overlap(rows, profiles),
        "seed_graph_overlap_rows_excluded": len(excluded_graph_rows),
        "seed_graph_overlap_pairs_excluded": len({(row["source"], row["target"]) for row in excluded_graph_rows}),
    }

def concept_evidence_overlap(rows: list[dict], profiles: dict[str, Profile]) -> dict:
    observed = 0
    for row in rows:
        concepts = set(_tokens(" ".join(profiles[row["source"]].concepts + profiles[row["target"]].concepts)))
        evidence = set(_tokens(_clean_text(row.get("evidence"))))
        if concepts & evidence:
            observed += 1
    return {"rows_with_token_overlap": observed, "rows_total": len(rows), "rate": observed / len(rows) if rows else None, "interpretation": "lexical overlap diagnostic only; it does not show causal leakage or correct for shared information"}
