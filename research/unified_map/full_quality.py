"""Exact neighbors and projection-quality metrics for the full-catalog map.

Every neighbor list is an exact blockwise search (no N x N matrix) that reuses
research.full_catalog.retrieve, so masking and key tie-breaking match the earlier runs.

    .venv/bin/python -m research.unified_map.full_quality neighbors \
        --run private-learning/unified-map/full-v1
    .venv/bin/python -m research.unified_map.full_quality metrics \
        --run private-learning/unified-map/full-v1 \
        --receipt research/unified_map/full-receipts/quality.v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from research.full_catalog.retrieve import Rows, dense_cosine, exclusion_mask, rank_row
from research.unified_map.build import _concepts

ROOT = Path(__file__).resolve().parents[2]
NODES = ROOT / "private-learning/unified-map/map-v2/nodes.jsonl"
STORED_K = 30
K = 10
QUERY_SAMPLE = 4000
SAMPLE_SEED = 17
BLOCK = 512


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_nodes() -> list[dict]:
    nodes = [json.loads(line) for line in NODES.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(nodes, key=lambda n: n["problem_key"])


def make_rows(nodes: list[dict]) -> Rows:
    return Rows([n["problem_key"] for n in nodes], [n["platform"] for n in nodes],
                [n["representative_key"] for n in nodes], [n["duplicate_candidate_group"] for n in nodes])


def ranked_blocks(rows: Rows, score_block, k: int, mode: str, *, block_size: int = BLOCK):
    """Yield (query, [(target, score), ...]) in key order, exact and blockwise.

    `cross` and `same` use retrieve.exclusion_mask unchanged (no duplicate-candidate
    exclusion). `all` excludes only self and exact duplicates (same representative); it lives
    here so the preserved research/full_catalog/retrieve.py stays byte-identical.
    """
    n = len(rows)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = np.asarray(score_block(start, stop))
        if block.shape != (stop - start, n) or not np.isfinite(block).all():
            raise ValueError("invalid_score_block")
        if mode == "all":
            mask = rows.representative[start:stop, None] == rows.representative[None, :]
        else:
            mask = exclusion_mask(rows, start, stop, mode, False)
        for offset in range(stop - start):
            picked = rank_row(block[offset], mask[offset], k)
            yield start + offset, [(int(i), float(block[offset, i])) for i in picked]


def neighbor_arrays(rows: Rows, score_block, k: int, mode: str, *, block_size: int = BLOCK):
    """Exact top-k as (indices int32, scores float32), padded with -1 / nan when short."""
    n = len(rows)
    idx = np.full((n, k), -1, dtype=np.int32)
    sc = np.full((n, k), np.nan, dtype=np.float32)
    for query, ranked in ranked_blocks(rows, score_block, k, mode, block_size=block_size):
        for slot, (target, score) in enumerate(ranked):
            idx[query, slot] = target
            sc[query, slot] = score
    return idx, sc


def euclidean_score_block(coords: np.ndarray):
    """Negative squared Euclidean distance, so larger is nearer; blockwise."""
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    norms = np.sum(coords * coords, axis=1)

    def block(start, stop):
        with np.errstate(all="ignore"):
            d2 = norms[start:stop, None] + norms[None, :] - 2.0 * (coords[start:stop] @ coords.T)
        return -np.maximum(d2, 0.0)

    return block


def preservation_counts(reference: np.ndarray, candidate: np.ndarray, k: int = K) -> np.ndarray:
    """Per-row |reference[:k] ∩ candidate[:k]|, ignoring -1 padding."""
    counts = np.zeros(len(reference), dtype=np.int16)
    for i in range(len(reference)):
        ref = {int(x) for x in reference[i, :k] if x >= 0}
        counts[i] = len(ref & {int(x) for x in candidate[i, :k] if x >= 0})
    return counts


def _ranks(score_row: np.ndarray, self_index: int) -> tuple[np.ndarray, np.ndarray]:
    """1-based ranks of every other point (best = 1; ties by index) and the ordered indices."""
    row = np.array(score_row, dtype=np.float64)
    row[self_index] = -np.inf
    order = np.lexsort((np.arange(len(row)), -row))
    ranks = np.empty(len(row), dtype=np.int64)
    ranks[order] = np.arange(1, len(row) + 1)
    return ranks, order


def trust_continuity(hi_block, lo_block, queries: np.ndarray, n: int, k: int = K, *, block: int = 256):
    """Trustworthiness and continuity at k, estimated on `queries` against all n points.

    Penalty form of Venna & Kaski: trust charges low-D neighbors that are not high-D
    neighbors by (high-D rank - k); continuity charges the symmetric case.
    """
    queries = np.asarray(queries, dtype=np.int64)
    trust_pen = np.zeros(len(queries))
    cont_pen = np.zeros(len(queries))
    for start in range(0, len(queries), block):
        chunk = queries[start:start + block]
        hi_rows = np.vstack([hi_block(q, q + 1) for q in chunk])
        lo_rows = np.vstack([lo_block(q, q + 1) for q in chunk])
        for offset, q in enumerate(chunk):
            hi_rank, hi_order = _ranks(hi_rows[offset], q)
            lo_rank, lo_order = _ranks(lo_rows[offset], q)
            hi_top, lo_top = set(hi_order[:k].tolist()), set(lo_order[:k].tolist())
            trust_pen[start + offset] = sum(hi_rank[j] - k for j in lo_top - hi_top)
            cont_pen[start + offset] = sum(lo_rank[j] - k for j in hi_top - lo_top)
    scale = 2.0 / (k * (2 * n - 3 * k - 1))
    return {"trustworthiness": float(1 - scale * trust_pen.mean()),
            "continuity": float(1 - scale * cont_pen.mean()), "queries": int(len(queries))}


def same_platform_fraction(neighbors: np.ndarray, platforms: np.ndarray, k: int = K) -> float:
    valid = neighbors[:, :k] >= 0
    same = platforms[neighbors[:, :k].clip(0)] == platforms[:, None]
    return float((same & valid).sum() / valid.sum())


def tag_agreement(neighbors: np.ndarray, concepts: list[set | None], k: int = K) -> dict:
    """Share of tagged neighbors sharing >=1 concept with a tagged query (descriptive only)."""
    shared = tagged = pairs = queries = 0
    for i, mine in enumerate(concepts):
        if not mine:
            continue
        queries += 1
        for j in neighbors[i, :k]:
            theirs = concepts[int(j)] if j >= 0 else None
            if not theirs:
                continue
            tagged += 1
            shared += bool(mine & theirs)
        pairs += int((neighbors[i, :k] >= 0).sum())
    return {"queries": queries, "tagged_neighbor_slots": tagged, "neighbor_slots": pairs,
            "share_with_concept": shared / tagged if tagged else None,
            "tagged_slot_coverage": tagged / pairs if pairs else None}


def random_tag_agreement(concepts: list[set | None], seed: int = SAMPLE_SEED, draws: int = 200000) -> float:
    """Chance rate that two random tagged nodes share a concept."""
    tagged = [i for i, c in enumerate(concepts) if c]
    rng = np.random.default_rng(seed)
    a = rng.choice(tagged, size=draws)
    b = rng.choice(tagged, size=draws)
    keep = a != b
    hits = sum(bool(concepts[x] & concepts[y]) for x, y in zip(a[keep], b[keep]))
    return hits / int(keep.sum())


def matched_random_tag_agreement(neighbors: np.ndarray, concepts: list[set | None], platforms: np.ndarray,
                                 k: int = K, seed: int = SAMPLE_SEED) -> dict:
    """Chance rate matched to each tagged neighbor slot's platform.

    Every tagged neighbor slot (i, j) is replaced by a random tagged node on j's platform,
    so the baseline has the same platform-pair composition as the neighbor list.
    """
    pools = {int(p): np.array([i for i, c in enumerate(concepts) if c and platforms[i] == p]) for p in np.unique(platforms)}
    rng = np.random.default_rng(seed)
    shared = total = 0
    for i, mine in enumerate(concepts):
        if not mine:
            continue
        for j in neighbors[i, :k]:
            if j < 0 or not concepts[int(j)]:
                continue
            pool = pools[int(platforms[int(j)])]
            draw = int(pool[rng.integers(len(pool))])
            if draw == i:
                continue
            total += 1
            shared += bool(mine & concepts[draw])
    return {"slots": total, "share_with_concept": shared / total if total else None}


def procrustes_disparity(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial import procrustes
    return float(procrustes(a.astype(np.float64), b.astype(np.float64))[2])


def concept_sets(nodes: list[dict]) -> list[set | None]:
    return [None if n["tags"] is None else _concepts(n["normalized_taxonomy"]) for n in nodes]


def cmd_neighbors(run: Path) -> None:
    nodes = load_nodes()
    rows = make_rows(nodes)
    vectors = np.load(run / "vectors.npy", allow_pickle=False)
    keys = json.loads((run / "keys.json").read_text(encoding="utf-8"))
    if keys != rows.keys:
        raise SystemExit("vector keys differ from nodes")
    out = run / "neighbors"
    out.mkdir(parents=True, exist_ok=False)
    receipt = {}
    for mode, name in (("all", "any"), ("cross", "cross")):
        started = time.perf_counter()
        idx, sc = neighbor_arrays(rows, dense_cosine(vectors), STORED_K, mode)
        np.save(out / f"{name}-idx.npy", idx, allow_pickle=False)
        np.save(out / f"{name}-score.npy", sc, allow_pickle=False)
        receipt[name] = {"seconds": round(time.perf_counter() - started, 2),
                         "short_rows": int((idx[:, -1] < 0).sum()),
                         "idx_sha256": sha256_file(out / f"{name}-idx.npy"),
                         "score_sha256": sha256_file(out / f"{name}-score.npy")}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _projection_files(run: Path) -> dict:
    found = {}
    for path in sorted((run / "projections").glob("*-seed*.npy")):
        method, seed = path.stem.split("-seed")
        found.setdefault(method, {})[int(seed)] = path
    return found


def cmd_metrics(run: Path, receipt_path: Path) -> None:
    nodes = load_nodes()
    rows = make_rows(nodes)
    n = len(nodes)
    vectors = np.load(run / "vectors.npy", allow_pickle=False)
    platforms = np.array([{"atcoder": 0, "codeforces": 1, "leetcode": 2}[x["platform"]] for x in nodes])
    meta = [json.loads(line) for line in (run / "vector_meta.jsonl").read_text(encoding="utf-8").splitlines()]
    tiers = np.array([m["reliability"] for m in meta])
    any_idx = np.load(run / "neighbors/any-idx.npy")
    cross_idx = np.load(run / "neighbors/cross-idx.npy")
    concepts = concept_sets(nodes)
    queries = np.sort(np.random.default_rng(SAMPLE_SEED).choice(n, size=QUERY_SAMPLE, replace=False))
    hi_block = dense_cosine(vectors)
    result = {"schema": "full-quality-receipt.v1", "nodes": n,
              "reference": "exact full-catalog title cosine; any-platform excludes self and exact duplicates only",
              "trust_query_sample": {"size": QUERY_SAMPLE, "seed": SAMPLE_SEED, "k": K},
              "high_dimensional": {
                  "same_platform_fraction_top10": same_platform_fraction(any_idx, platforms),
                  "tag_agreement_top10": tag_agreement(any_idx, concepts),
                  "tag_agreement_top10_platform_matched_random": matched_random_tag_agreement(any_idx, concepts, platforms),
                  "tag_agreement_cross_platform_top10": tag_agreement(cross_idx, concepts),
                  "tag_agreement_cross_platform_top10_platform_matched_random":
                      matched_random_tag_agreement(cross_idx, concepts, platforms),
                  "random_pair_tag_agreement": random_tag_agreement(concepts)},
              "platform_share": {p: float((platforms == i).mean()) for i, p in enumerate(("atcoder", "codeforces", "leetcode"))},
              "projections": {}, "stability": {}}
    per_node = {}
    neighbors_2d = {}
    for method, seeds in _projection_files(run).items():
        for seed, path in seeds.items():
            coords = np.load(path)
            block = euclidean_score_block(coords)
            lo_any, _ = neighbor_arrays(rows, block, K, "all")
            lo_cross, _ = neighbor_arrays(rows, block, K, "cross")
            neighbors_2d[(method, seed)] = lo_any
            pres_any = preservation_counts(any_idx, lo_any)
            pres_cross = preservation_counts(cross_idx, lo_cross)
            tc = trust_continuity(hi_block, block, queries, n)
            entry = {
                "preservation_top10_any": float(pres_any.mean() / K),
                "preservation_top10_cross_platform": float(pres_cross.mean() / K),
                **tc,
                "same_platform_fraction_top10": same_platform_fraction(lo_any, platforms),
                "tag_agreement_top10": tag_agreement(lo_any, concepts),
                "tag_agreement_top10_platform_matched_random": matched_random_tag_agreement(lo_any, concepts, platforms),
                "preservation_by_platform": {p: float(pres_any[platforms == i].mean() / K)
                                             for i, p in enumerate(("atcoder", "codeforces", "leetcode"))},
                "preservation_by_reliability": {t: float(pres_any[tiers == t].mean() / K)
                                                for t in ("reliable", "degraded", "unreliable") if (tiers == t).any()},
                "output_sha256": sha256_file(path)}
            result["projections"][f"{method}-seed{seed}"] = entry
            if seed == 17:
                per_node[method] = pres_any.astype(np.uint8)
    for method, seeds in _projection_files(run).items():
        ordered = sorted(seeds)
        pairs = [(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1:]]
        if not pairs:
            continue
        entries = []
        for a, b in pairs:
            disparity = procrustes_disparity(np.load(seeds[a]), np.load(seeds[b]))
            overlap = float(preservation_counts(neighbors_2d[(method, a)], neighbors_2d[(method, b)]).mean() / K)
            entries.append({"seeds": [a, b], "procrustes_disparity": disparity, "top10_2d_neighbor_overlap": overlap,
                            "identical_layout": bool(np.array_equal(np.load(seeds[a]), np.load(seeds[b])))})
        result["stability"][method] = entries
    np.savez(run / "preservation.npz", **per_node)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(receipt_path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=1, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("neighbors")
    a.add_argument("--run", type=Path, required=True)
    b = sub.add_parser("metrics")
    b.add_argument("--run", type=Path, required=True)
    b.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "neighbors":
        cmd_neighbors(args.run)
    else:
        cmd_metrics(args.run, args.receipt)


if __name__ == "__main__":
    main()
