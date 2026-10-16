"""Phase 5 judged pool: frozen stratified query sample and pooled pairs (stdlib only).

Queries: 50 per platform (default), stratified by difficulty band (AtCoder: contest
family, since it has no difficulty), allocated proportionally with largest remainder
and a per-stratum minimum, then chosen by SHA-256("pool-v1:" + key). LeetCode and
Codeforces queries must exist in both A1 and A2, so every method is evaluable on them.

Pairs: union of each non-random method's top-`depth` plus each random run's top-
`random_depth`. Each pair keeps its per-method ranks and `min_rank`, so a judged
subset for any K <= depth is derivable. The committed pool holds keys only. The
private label sheet is blind: no method names, ranks or scores, and a hash-based
pair order.

    python3 research/full_catalog/pool.py \
        --catalog private-learning/full-catalog/catalog-v1/catalog.v1.jsonl \
        --pool research/full_catalog/pool.v1.json \
        --label-sheet private-learning/full-catalog/pool-v1/label_sheet.jsonl
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "private-learning/full-catalog"
JUDGMENTS = ROOT / "research/relationship-judgments.v1.json"

RUNS = {
    "A1/minilm": "retrieval-v1/minilm-A1-cross",
    "A1/tfidf_word": "baselines-v1/A1/tfidf_word",
    "A1/tfidf_char": "baselines-v1/A1/tfidf_char",
    "A1/random": "baselines-v1/A1/random",
    "A2/minilm": "retrieval-v1/minilm-A2-cross",
    "A2/tfidf_word": "baselines-v1/A2/tfidf_word",
    "A2/tfidf_char": "baselines-v1/A2/tfidf_char",
    "A2/tag_jaccard": "baselines-v1/A2/tag_jaccard",
    "A2/random": "baselines-v1/A2/random",
}
CF_BANDS = ((1200, "cf<1200"), (1600, "cf1200-1599"), (2000, "cf1600-1999"), (2400, "cf2000-2399"))
LABEL_SCALE = {0: "unrelated", 1: "partially_related", 2: "relevant", 3: "strongly_relevant"}


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def stratum(row):
    platform = row["platform"]
    if platform == "leetcode":
        return f"lc-{row['native_difficulty'].lower()}"
    if platform == "codeforces":
        rating = row["native_difficulty"]
        if rating is None:
            return "cf-unrated"
        for bound, name in CF_BANDS:
            if rating < bound:
                return name
        return "cf>=2400"
    match = re.match(r"[a-z]+", row["problem_key"].split(":", 1)[1].lower())
    family = match.group(0) if match else ""
    return f"at-{family}" if family in {"abc", "arc", "agc"} else "at-other"


def allocate(sizes, total, minimum=2):
    """Proportional allocation with a per-stratum minimum and largest remainder; capped by size."""
    names = sorted(sizes)
    alloc = {n: min(minimum, sizes[n]) for n in names}
    remaining = total - sum(alloc.values())
    if remaining < 0:
        raise ValueError("total_below_minimum_allocation")
    population = sum(sizes.values())
    quotas = {n: total * sizes[n] / population for n in names}
    for name in names:
        extra = min(max(int(quotas[name]) - alloc[name], 0), sizes[name] - alloc[name], remaining)
        alloc[name] += extra
        remaining -= extra
    order = sorted(names, key=lambda n: (-(quotas[n] - int(quotas[n])), n))
    while remaining > 0 and any(alloc[n] < sizes[n] for n in names):
        for name in order:
            if remaining and alloc[name] < sizes[name]:
                alloc[name] += 1
                remaining -= 1
    return alloc


def sample_queries(catalog_rows, universes, per_platform):
    """Return [(key, platform, stratum)] in key order."""
    by_key = {r["problem_key"]: r for r in catalog_rows}
    a1, a2 = universes["A1"], universes["A2"]
    chosen = []
    for platform in ("atcoder", "codeforces", "leetcode"):
        population = [k for k in sorted(a1) if by_key[k]["platform"] == platform
                      and (platform == "atcoder" or k in a2)]
        groups = {}
        for key in population:
            groups.setdefault(stratum(by_key[key]), []).append(key)
        alloc = allocate({s: len(v) for s, v in groups.items()}, min(per_platform, len(population)))
        for name, keys in groups.items():
            picked = sorted(keys, key=lambda k: _hash("pool-v1:" + k))[:alloc[name]]
            chosen += [(k, platform, name) for k in picked]
    return sorted(chosen)


def pool_pairs(queries, ranked_lists, depth, random_depth):
    """ranked_lists: {method: {query: [target keys in rank order]}}."""
    pairs = {}
    for query, _, _ in queries:
        for method in sorted(ranked_lists):
            targets = ranked_lists[method].get(query)
            if targets is None:
                continue
            limit = random_depth if method.endswith("/random") else depth
            for rank, target in enumerate(targets[:limit], start=1):
                entry = pairs.setdefault((query, target), {"query": query, "target": target, "ranks": {}})
                entry["ranks"][method] = rank
    out = []
    for (query, target), entry in sorted(pairs.items()):
        system = [r for m, r in entry["ranks"].items() if not m.endswith("/random")]
        entry["min_rank"] = min(system) if system else None
        entry["random_only"] = not system
        out.append(entry)
    return out


def label_sheet(pairs, catalog_rows, prior_pairs):
    """Blind rows for labeling: metadata only, no method/rank/score, hash-ordered within query."""
    by_key = {r["problem_key"]: r for r in catalog_rows}

    def card(key):
        r = by_key[key]
        return {"key": key, "title": r["title"], "url": r["url"], "tags": r["tags"],
                "difficulty": r["native_difficulty"]}

    rows = []
    for pair in sorted(pairs, key=lambda p: (p["query"], _hash(p["query"] + "|" + p["target"]))):
        rows.append({
            "pair_id": _hash(pair["query"] + "|" + pair["target"])[:16],
            "query": card(pair["query"]),
            "target": card(pair["target"]),
            "tier": 1 if pair["random_only"] or pair["min_rank"] <= 5 else 2,
            "has_prior_judgment": (pair["query"], pair["target"]) in prior_pairs,
            "label": None, "deceptive": None, "label_source": None, "labeler": None, "notes": None,
        })
    return rows


def _load_top10(path):
    lists = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            lists[item["query"]] = [k for k, _ in item["results"]]
    return lists


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True, help="new committed pool (keys only)")
    parser.add_argument("--label-sheet", type=Path, required=True, help="new private blind label sheet")
    parser.add_argument("--per-platform", type=int, default=50)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--random-depth", type=int, default=2)
    args = parser.parse_args()

    catalog_rows = [json.loads(l) for l in args.catalog.read_text(encoding="utf-8").splitlines() if l.strip()]
    files = {m: PRIVATE / d / "top10.jsonl" for m, d in RUNS.items()}
    ranked = {m: _load_top10(p) for m, p in files.items()}
    universes = {"A1": set(ranked["A1/minilm"]), "A2": set(ranked["A2/minilm"])}
    queries = sample_queries(catalog_rows, universes, args.per_platform)
    pairs = pool_pairs(queries, ranked, args.depth, args.random_depth)
    judgments = json.loads(JUDGMENTS.read_text(encoding="utf-8"))["judgments"]
    prior = {(j["source"], j["target"]) for j in judgments}

    pool = {
        "schema": "judged-pool.v1",
        "spec": "specs/adaptive-learning/full-catalog-similarity.md#phase-5--evaluation",
        "params": {"per_platform": args.per_platform, "depth": args.depth, "random_depth": args.random_depth,
                   "selection_hash": "sha256('pool-v1:' + key)", "stratum_minimum": 2},
        "label_scale": LABEL_SCALE,
        "label_rules": ("Unjudged pairs are never negatives. 'deceptive' marks title/narrative similarity "
                        "with a different algorithm. label_source is 'ai_reviewed' or 'human'. Labelers see "
                        "metadata and public URLs only, never the quarantined statements."),
        "inputs": {m: _hash_file(p) for m, p in sorted(files.items())},
        "strata": dict(sorted(Counter(s for _, _, s in queries).items())),
        "queries": [{"key": k, "platform": p, "stratum": s} for k, p, s in queries],
        "pairs": pairs,
        "summary": {
            "queries": len(queries),
            "pairs": len(pairs),
            "pairs_tier1": sum(1 for p in pairs if p["random_only"] or p["min_rank"] <= 5),
            "random_only_pairs": sum(1 for p in pairs if p["random_only"]),
            "pairs_with_prior_judgment": sum(1 for p in pairs if (p["query"], p["target"]) in prior),
        },
    }
    sheet = label_sheet(pairs, catalog_rows, prior)
    for path, text in ((args.pool, json.dumps(pool, indent=1, sort_keys=True) + "\n"),
                       (args.label_sheet, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                                  for r in sheet))):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x", encoding="utf-8") as handle:
            handle.write(text)


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
