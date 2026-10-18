"""Language-bias and cross-language metrics (numpy only, deterministic given seeds).

Every metric returns plain numbers plus a seeded query bootstrap where it is an average over queries.
Reduced language homophily and platform mixing are reported but are never selection criteria; the
random-vector control exists to show what mixing without meaning scores.
"""

from __future__ import annotations

import numpy as np

BOOT = 200
SEED = 17


def bootstrap(per_query: np.ndarray, resamples: int = BOOT, seed: int = SEED) -> dict:
    """Mean with a 95% percentile interval over query resamples."""
    values = np.asarray(per_query, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[idx].mean(axis=1)
    return {"mean": float(values.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": int(values.size)}


def bootstrap_ratio(num: np.ndarray, den: np.ndarray, resamples: int = BOOT, seed: int = SEED) -> dict:
    """Ratio of sums (sum num / sum den) with a query-resampled interval; None when undefined."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    if num.size == 0 or den.sum() == 0:
        return {"value": None, "lo": None, "hi": None, "n": int(num.size)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(resamples, num.size))
    d = den[idx].sum(axis=1)
    ok = d > 0
    ratios = num[idx].sum(axis=1)[ok] / d[ok]
    return {"value": float(num.sum() / den.sum()), "lo": float(np.percentile(ratios, 2.5)),
            "hi": float(np.percentile(ratios, 97.5)), "n": int(num.size)}


def bootstrap_lift(s, t, bs, bt, resamples: int = BOOT, seed: int = SEED) -> dict:
    """Interval for (sum s / sum t) / (sum bs / sum bt) with the same query resamples for both terms."""
    s, t, bs, bt = (np.asarray(x, dtype=np.float64) for x in (s, t, bs, bt))
    if s.size == 0 or t.sum() == 0 or bt.sum() == 0 or bs.sum() == 0:
        return {"value": None, "lo": None, "hi": None, "valid_resamples": 0}
    idx = np.random.default_rng(seed).integers(0, s.size, size=(resamples, s.size))
    obs_den, base_den, base_num = t[idx].sum(axis=1), bt[idx].sum(axis=1), bs[idx].sum(axis=1)
    ok = (obs_den > 0) & (base_den > 0) & (base_num > 0)
    lifts = (s[idx].sum(axis=1)[ok] / obs_den[ok]) / (base_num[ok] / base_den[ok])
    value = (s.sum() / t.sum()) / (bs.sum() / bt.sum())
    return {"value": float(value), "lo": float(np.percentile(lifts, 2.5)), "hi": float(np.percentile(lifts, 97.5)),
            "valid_resamples": int(ok.sum())}


def homophily(neighbors: np.ndarray, groups: np.ndarray, queries: np.ndarray, k: int = 10) -> dict:
    """Share of top-k neighbors in the query's language group, against the language-blind chance level."""
    groups = np.asarray(groups)
    n = len(groups)
    sizes = {g: int((groups == g).sum()) for g in np.unique(groups)}
    obs, chance = [], []
    for q in queries:
        nb = neighbors[q, :k]
        nb = nb[nb >= 0]
        if nb.size == 0:
            continue
        obs.append(float((groups[nb] == groups[q]).mean()))
        chance.append((sizes[groups[q]] - 1) / (n - 1))
    obs, chance = np.array(obs), np.array(chance)
    ratio = bootstrap_ratio(obs, chance)
    return {"observed": float(obs.mean()) if obs.size else None, "chance": float(chance.mean()) if chance.size else None,
            "excess_ratio": ratio, "queries": int(obs.size)}


def platform_composition(neighbors: np.ndarray, platforms: np.ndarray, queries: np.ndarray, k: int = 10) -> dict:
    counts = {}
    total = 0
    for q in queries:
        nb = neighbors[q, :k]
        nb = nb[nb >= 0]
        for p in platforms[nb]:
            counts[str(p)] = counts.get(str(p), 0) + 1
        total += nb.size
    return {name: c / total for name, c in sorted(counts.items())} if total else {}


def translation_retrieval(query_vecs: np.ndarray, cand_vecs: np.ndarray, true_cand: np.ndarray) -> dict:
    """Rank of each query's true candidate by cosine (strictly-greater count + 1, so ties count against)."""
    scores = query_vecs.astype(np.float64) @ cand_vecs.astype(np.float64).T
    true_scores = scores[np.arange(len(true_cand)), true_cand]
    ranks = 1 + (scores > true_scores[:, None] + 1e-12).sum(axis=1)
    c = cand_vecs.shape[0]
    chance_mrr = float((1.0 / np.arange(1, c + 1)).mean())
    return {"mrr": bootstrap(1.0 / ranks), "recall@1": bootstrap((ranks <= 1).astype(float)),
            "recall@10": bootstrap((ranks <= 10).astype(float)), "recall@50": bootstrap((ranks <= 50).astype(float)),
            "median_rank": float(np.median(ranks)), "candidates": int(c), "queries": int(len(ranks)),
            "chance_mrr": chance_mrr, "chance_recall@10": min(1.0, 10 / c), "per_query_rr": (1.0 / ranks).tolist()}


def top_neighbors(index_vecs: np.ndarray, query_vecs: np.ndarray, exclude: list[list[int]], k: int = 10) -> np.ndarray:
    """Exact top-k by cosine for each query, excluding that query's own nodes; ties by index."""
    scores = query_vecs.astype(np.float64) @ index_vecs.astype(np.float64).T
    out = np.full((len(query_vecs), k), -1, dtype=np.int64)
    for i, row in enumerate(scores):
        row = row.copy()
        row[exclude[i]] = -np.inf
        order = np.lexsort((np.arange(len(row)), -row))[:k]
        out[i] = order
    return out


def neighbor_consistency(index_vecs, qa_vecs, qb_vecs, exclude, k: int = 10) -> dict:
    """Top-k overlap between a title's two language versions queried in the same index."""
    a = top_neighbors(index_vecs, qa_vecs, exclude, k)
    b = top_neighbors(index_vecs, qb_vecs, exclude, k)
    overlap = np.array([len(set(x.tolist()) & set(y.tolist())) / k for x, y in zip(a, b)])
    return {"overlap": bootstrap(overlap), "chance": float(k / (index_vecs.shape[0] - 1)),
            "per_query_overlap": overlap.tolist()}


def paired_difference(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> dict:
    """Bootstrap interval for mean(a - b) over the same queries."""
    return bootstrap(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))


def shared_pairs(neighbors, concepts, queries, k: int = 10):
    """Per query: (pairs sharing a concept, pairs) among its tagged neighbors, plus tagged/total slots."""
    shared, total, tagged_slots, slots = [], [], 0, 0
    for q in queries:
        nb = [int(j) for j in neighbors[q, :k] if j >= 0]
        slots += len(nb)
        tagged = [j for j in nb if concepts[j]]
        tagged_slots += len(tagged)
        s = t = 0
        for i in range(len(tagged)):
            for j in range(i + 1, len(tagged)):
                t += 1
                s += bool(concepts[tagged[i]] & concepts[tagged[j]])
        shared.append(s)
        total.append(t)
    return np.array(shared, dtype=float), np.array(total, dtype=float), tagged_slots, slots


def matched_pair_baseline(neighbors, concepts, platforms, queries, k: int = 10, seed: int = SEED):
    """Same pair counts, but each tagged neighbor is replaced by a random tagged node of its platform."""
    pools = {int(p): np.array([i for i, c in enumerate(concepts) if c and platforms[i] == p]) for p in np.unique(platforms)}
    rng = np.random.default_rng(seed)
    shared, total = [], []
    for q in queries:
        nb = [int(j) for j in neighbors[q, :k] if j >= 0 and concepts[int(j)]]
        draws = [int(pools[int(platforms[j])][rng.integers(len(pools[int(platforms[j])]))]) for j in nb]
        s = t = 0
        for i in range(len(draws)):
            for j in range(i + 1, len(draws)):
                t += 1
                s += bool(concepts[draws[i]] & concepts[draws[j]])
        shared.append(s)
        total.append(t)
    return np.array(shared, dtype=float), np.array(total, dtype=float)


def concept_coherence(neighbors, concepts, platforms, queries, k: int = 10, seed: int = SEED) -> dict:
    """Do a query's tagged neighbors agree with each other more than platform-matched random nodes?"""
    s, t, tagged_slots, slots = shared_pairs(neighbors, concepts, queries, k)
    bs, bt = matched_pair_baseline(neighbors, concepts, platforms, queries, k, seed)
    obs = bootstrap_ratio(s, t, seed=seed)
    base = bootstrap_ratio(bs, bt, seed=seed)
    lift = None
    if obs["value"] is not None and base["value"]:
        lift = obs["value"] / base["value"]
    return {"observed_pair_share": obs, "baseline_pair_share": base, "lift": lift,
            "lift_interval": bootstrap_lift(s, t, bs, bt, seed=seed),
            "tagged_slot_coverage": tagged_slots / slots if slots else None,
            "queries_with_pairs": int((t > 0).sum()), "queries": int(len(queries))}


def remove_group_centroids(vectors: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's mean vector and re-normalize (post-hoc language debiasing)."""
    out = vectors.astype(np.float64).copy()
    for g in np.unique(groups):
        mask = groups == g
        out[mask] -= out[mask].mean(axis=0)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (out / norms).astype(np.float32)


def apply_selection_rule(table: dict, *, baseline: str, controls: tuple, sizes: dict,
                         min_gain: float = 0.10) -> dict:
    """Pre-registered rule (specs/unified-similarity-map/language-bias.md). Never uses homophily or mixing.

    table[name] = {english_retention, mrr, consistency, consistency_diff_vs_baseline_lo, coherence_lift,
                   coherence_coverage}; controls must include 'random'.
    """
    random_mrr = table[controls[0]]["mrr"] if controls else 0.0
    eligible, reasons = [], {}
    for name, m in table.items():
        if name == baseline or name in controls:
            continue
        why = []
        if m["english_retention"] is None or m["english_retention"] < 0.95:
            why.append("G1 english quality retention below 0.95")
        if not (m["mrr"] > 0.25 and m["mrr"] >= 10 * random_mrr):
            why.append("G2 translation MRR not above 0.25 and 10x the random control")
        assessable = m["coherence_coverage"] is not None and m["coherence_coverage"] >= 0.20 and m["coherence_lift"] is not None
        if assessable and m["coherence_lift"] < 1.0:
            why.append("G3 pure-Japanese concept coherence below chance")
        reasons[name] = why
        if not why:
            eligible.append(name)
    decision = {"baseline": baseline, "eligible": eligible, "ineligible_reasons": {k: v for k, v in reasons.items() if v}}
    if not eligible:
        decision.update(winner=None, replaced=False, note="no candidate passed the guards; baseline retained")
        return decision
    best = max(eligible, key=lambda n: table[n]["consistency"])
    tied = [n for n in eligible if table[n]["consistency"] >= table[best]["consistency"] - table[n].get("consistency_halfwidth", 0.0)]
    winner = sorted(tied, key=lambda n: (-table[n]["mrr"], sizes.get(n, 0)))[0]
    gain = table[winner]["consistency"] - table[baseline]["consistency"]
    lo = table[winner].get("consistency_diff_vs_baseline_lo")
    replaced = gain >= min_gain and lo is not None and lo > 0
    decision.update(winner=winner, replaced=bool(replaced), gain_vs_baseline=gain, tied=tied,
                    note="winner replaces the baseline" if replaced else "winner does not clear the baseline margin; baseline retained")
    return decision
