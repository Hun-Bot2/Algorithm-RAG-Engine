"""Phase 4: exact blockwise top-K retrieval without materializing the N x N matrix.

Scores come from a caller-supplied block function, so dense cosine (normalized
vectors) and sparse TF-IDF share the same masking, ranking and output code.
Ranking is score descending, then key ascending; rows are required to be sorted by
key so that index order equals key order.

    .venv/bin/python research/full_catalog/retrieve.py \
        --catalog private-learning/full-catalog/catalog-v1/catalog.v1.jsonl \
        --vectors V.npy --keys V.keys.json --mode cross \
        --output private-learning/full-catalog/<new-run> --receipt <new>.json
"""

import argparse
import hashlib
import json
import platform as host_platform
import time
from pathlib import Path

import numpy as np

KS = (10, 20, 50)


def _codes(values):
    """Stable integer codes for hashable values; None maps to -1."""
    table = {v: i for i, v in enumerate(sorted({v for v in values if v is not None}))}
    return np.array([table[v] if v is not None else -1 for v in values], dtype=np.int64)


class Rows:
    """Retrieval universe: eligible rows sorted by key, with exclusion group codes."""

    def __init__(self, keys, platforms, representatives, candidate_groups):
        keys = list(keys)
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError("keys_must_be_unique_and_sorted")
        if not (len(keys) == len(platforms) == len(representatives) == len(candidate_groups)):
            raise ValueError("row_field_length_mismatch")
        self.keys = keys
        self.platform = _codes(platforms)
        self.representative = _codes(representatives)
        self.candidate = _codes(candidate_groups)

    @classmethod
    def from_catalog(cls, catalog_rows, keys):
        by_key = {r["problem_key"]: r for r in catalog_rows}
        rows = [by_key[k] for k in keys]
        return cls(keys, [r["platform"] for r in rows], [r["representative_key"] for r in rows],
                   [r["duplicate_candidate_group"] for r in rows])

    def __len__(self):
        return len(self.keys)


def exclusion_mask(rows, start, stop, mode, exclude_candidates):
    """Boolean (stop-start, N) mask; True means the target is excluded for that query."""
    q = slice(start, stop)
    same_platform = rows.platform[q, None] == rows.platform[None, :]
    if mode == "cross":
        mask = same_platform.copy()
    elif mode == "same":
        mask = ~same_platform
    else:
        raise ValueError(f"unknown_mode:{mode}")
    # Self and verified exact duplicates share a representative code.
    mask |= rows.representative[q, None] == rows.representative[None, :]
    if exclude_candidates:
        cand = rows.candidate[q, None]
        mask |= (cand == rows.candidate[None, :]) & (cand >= 0)
    return mask


def rank_row(scores, excluded, k):
    """Indices of the top-k allowed targets: score desc, then index (key) asc. Exact at ties."""
    allowed = np.flatnonzero(~excluded)
    if allowed.size == 0:
        return allowed
    values = scores[allowed]
    if allowed.size > k:
        threshold = np.partition(values, -k)[-k]
        keep = values >= threshold  # keeps every tie at the boundary before ordering
        allowed, values = allowed[keep], values[keep]
    order = np.lexsort((allowed, -values))
    return allowed[order[:k]]


def topk(rows, score_block, k, mode, *, block_size=1024, exclude_candidates=True):
    """Yield (query_index, [(target_index, score), ...]) for every row, in key order.

    score_block(start, stop) must return a float array of shape (stop-start, N).
    """
    if block_size < 1 or k < 1:
        raise ValueError("block_size_and_k_must_be_positive")
    n = len(rows)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = np.asarray(score_block(start, stop))
        if block.shape != (stop - start, n):
            raise ValueError(f"score_block_shape:{block.shape}")
        if not np.isfinite(block).all():
            raise ValueError("score_block_not_finite")
        mask = exclusion_mask(rows, start, stop, mode, exclude_candidates)
        for offset in range(stop - start):
            picked = rank_row(block[offset], mask[offset], k)
            yield start + offset, [(int(i), float(block[offset, i])) for i in picked]


def dense_cosine(vectors):
    """Block function for L2-normalized float32 vectors (cosine == dot product)."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("vectors_must_be_l2_normalized")

    def block(start, stop):
        # numpy 2.1 + Accelerate raises spurious FP-status warnings on float32 matmul;
        # results are checked for finiteness in topk() instead.
        with np.errstate(all="ignore"):
            return vectors[start:stop] @ vectors.T

    return block


def write_results(rows, results, output_dir, ks=KS):
    """Write one JSONL per K (prefixes of the same ranked list); return file hashes."""
    output_dir.mkdir(parents=True, exist_ok=False)
    handles = {k: open(output_dir / f"top{k}.jsonl", "x", encoding="utf-8") for k in ks}
    digests = {k: hashlib.sha256() for k in ks}
    counts = {"queries": 0, "short_lists": 0}
    try:
        for query, ranked in results:
            counts["queries"] += 1
            counts["short_lists"] += len(ranked) < max(ks)
            for k in ks:
                line = json.dumps({"query": rows.keys[query],
                                   "results": [[rows.keys[i], round(s, 6)] for i, s in ranked[:k]]},
                                  ensure_ascii=False, separators=(",", ":")) + "\n"
                handles[k].write(line)
                digests[k].update(line.encode())
    finally:
        for handle in handles.values():
            handle.close()
    return {f"top{k}.jsonl": digests[k].hexdigest() for k in ks}, counts


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True, help=".npy, L2-normalized, row order = --keys")
    parser.add_argument("--keys", type=Path, required=True, help="JSON list of sorted problem keys")
    parser.add_argument("--mode", choices=("cross", "same"), required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--keep-candidates", action="store_true",
                        help="do not exclude same duplicate-candidate-group targets")
    parser.add_argument("--output", type=Path, required=True, help="new private directory")
    parser.add_argument("--receipt", type=Path, required=True, help="new aggregate receipt path")
    args = parser.parse_args()

    keys = json.loads(args.keys.read_text(encoding="utf-8"))
    catalog_rows = [json.loads(line) for line in args.catalog.read_text(encoding="utf-8").splitlines()]
    eligible = {r["problem_key"] for r in catalog_rows if r["eligibility"] == "eligible"}
    if not set(keys) <= eligible:
        raise SystemExit("vector keys include rows that are not eligible in the catalog")
    rows = Rows.from_catalog(catalog_rows, keys)
    vectors = np.load(args.vectors)
    if vectors.shape[0] != len(rows):
        raise SystemExit("vector row count does not match keys")

    started = time.perf_counter()
    results = topk(rows, dense_cosine(vectors), max(KS), args.mode,
                   block_size=args.block_size, exclude_candidates=not args.keep_candidates)
    outputs, counts = write_results(rows, results, args.output)
    receipt = {
        "schema": "retrieval-receipt.v1",
        "mode": args.mode,
        "exclude_candidates": not args.keep_candidates,
        "block_size": args.block_size,
        "ks": list(KS),
        "inputs": {"catalog": _sha256_file(args.catalog), "vectors": _sha256_file(args.vectors),
                   "keys": _sha256_file(args.keys)},
        "outputs": outputs,
        "counts": {**counts, "rows": len(rows), "dim": int(vectors.shape[1])},
        "seconds": round(time.perf_counter() - started, 3),
        "environment": {"numpy": np.__version__, "python": host_platform.python_version(),
                        "machine": host_platform.machine()},
        "tie_break": "score desc, key asc; scores rounded to 6 decimals only in output files",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with open(args.receipt, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
