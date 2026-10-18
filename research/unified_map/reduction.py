"""Reproducible dimensionality reduction ablation for the unified-map pilot.

This research CLI reads the pinned A1 MiniLM title vectors, compares exact
cross-platform Top10 retrieval before and after compression, then compares 2D
neighborhoods on an explicitly stratified sample. It never re-embeds data.

Example (run from repository root):
    .venv/bin/python -m research.unified_map.reduction
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from research.full_catalog import inputs as full_catalog_inputs
from research.full_catalog.retrieve import Rows, dense_cosine, exclusion_mask, rank_row, topk


SEED = 17
PLATFORMS = ("atcoder", "codeforces", "leetcode")
DIMENSIONS = (256, 128, 64)
K = 10


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_sample(keys, platforms, cap=3000, seed=SEED):
    """Return sorted source indices sampled evenly by platform, up to cap."""
    if len(keys) != len(platforms) or cap < 1:
        raise ValueError("invalid_sample_inputs")
    groups = {p: np.array([i for i, value in enumerate(platforms) if value == p], dtype=np.int64)
              for p in PLATFORMS}
    groups = {p: v for p, v in groups.items() if v.size}
    if not groups:
        raise ValueError("no_supported_platforms")
    quota = cap // len(groups)
    remainder = cap % len(groups)
    rng = np.random.default_rng(seed)
    selected = []
    for slot, (platform_name, indices) in enumerate(groups.items()):
        count = min(indices.size, quota + (slot < remainder))
        selected.extend(rng.choice(indices, size=count, replace=False).tolist())
    # Fill unused quota when a small stratum was exhausted, preserving the cap.
    remaining = min(cap, len(keys)) - len(selected)
    if remaining:
        selected_set = set(selected)
        pool = np.array([i for i in range(len(keys)) if i not in selected_set], dtype=np.int64)
        selected.extend(rng.choice(pool, size=remaining, replace=False).tolist())
    return np.array(sorted(selected), dtype=np.int64)


def _normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError("vectors_must_be_finite_and_nonzero")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def compression_models(vectors, seed=SEED, dimensions=DIMENSIONS):
    """Fit deterministic, single-threaded PCA and TruncatedSVD projections."""
    from sklearn.decomposition import PCA, TruncatedSVD

    n, dim = vectors.shape
    if min(n, dim) <= max(dimensions):
        raise ValueError("input_rank_too_small_for_requested_dimensions")
    outputs = {}
    timings = {}
    for kind, model_type in (("pca", PCA), ("svd", TruncatedSVD)):
        for target_dim in dimensions:
            started = time.perf_counter()
            if kind == "pca":
                model = model_type(n_components=target_dim, svd_solver="randomized",
                                   random_state=seed, iterated_power=4)
            else:
                model = model_type(n_components=target_dim, algorithm="randomized",
                                   random_state=seed, n_iter=7)
            projected = model.fit_transform(vectors)
            outputs[f"{kind}{target_dim}"] = _normalize(projected)
            timings[f"{kind}{target_dim}"] = {
                "seconds": round(time.perf_counter() - started, 4),
                "components": int(model.components_.shape[0]),
                "component_bytes": int(model.components_.nbytes),
                "explained_variance_ratio_sum": round(float(model.explained_variance_ratio_.sum()), 8),
            }
            del model, projected
    return outputs, timings


def exact_neighbors(rows, vectors, *, block_size=512):
    """Return each row's exact cross-platform Top10 target indices."""
    neighbors = []
    for _, ranked in topk(rows, dense_cosine(vectors), K, "cross", block_size=block_size,
                          exclude_candidates=True):
        neighbors.append(np.array([i for i, _ in ranked], dtype=np.int32))
    return neighbors


def top10_preservation(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError("neighbor_query_count_mismatch")
    overlaps = []
    for expected, observed in zip(reference, candidate):
        denominator = len(expected)
        overlaps.append(len(set(expected.tolist()) & set(observed.tolist())) / denominator if denominator else 1.0)
    return {"queries": len(overlaps), "mean_top10_overlap": round(float(np.mean(overlaps)), 8),
            "queries_with_exact_top10": int(sum(v == 1.0 for v in overlaps)),
            "queries_with_candidates": int(sum(len(v) > 0 for v in reference))}


def sampled_reference_neighbors(vectors, rows, k=K):
    """Exact within-sample cross-platform neighbors for the 2D overlap baseline."""
    n = len(vectors)
    with np.errstate(all="ignore"):
        scores = vectors @ vectors.T
    result = []
    for i in range(n):
        excluded = exclusion_mask(rows, i, i + 1, "cross", exclude_candidates=True)[0]
        result.append(rank_row(scores[i], excluded, k).astype(np.int32))
    return result


def coordinates_2d(vectors, seed=SEED):
    """Produce PCA/SVD/UMAP/t-SNE 2D coordinates with fixed parameters."""
    from sklearn.decomposition import PCA, TruncatedSVD

    methods = {}
    models = [
        ("pca2", PCA(n_components=2, svd_solver="randomized", random_state=seed,
                      iterated_power=4), "fit_transform"),
        ("svd2", TruncatedSVD(n_components=2, algorithm="randomized", random_state=seed,
                              n_iter=7), "fit_transform"),
    ]
    for name, model, operation in models:
        started = time.perf_counter()
        coords = getattr(model, operation)(vectors)
        methods[name] = (np.asarray(coords, dtype=np.float32), {
            "seconds": round(time.perf_counter() - started, 4),
            "parameters": {"solver": "randomized", "seed": seed},
        })
    import umap
    started = time.perf_counter()
    umap_coords = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine",
                            random_state=seed, transform_seed=seed, n_jobs=1).fit_transform(vectors)
    methods["umap2"] = (np.asarray(umap_coords, dtype=np.float32), {
        "seconds": round(time.perf_counter() - started, 4),
        "parameters": {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1,
                       "metric": "cosine", "random_state": seed, "transform_seed": seed,
                       "n_jobs": 1},
    })
    from sklearn.manifold import TSNE
    started = time.perf_counter()
    tsne_coords = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                       max_iter=1000, method="barnes_hut", metric="cosine",
                       random_state=seed, n_jobs=1).fit_transform(vectors)
    methods["tsne2"] = (np.asarray(tsne_coords, dtype=np.float32), {
        "seconds": round(time.perf_counter() - started, 4),
        "parameters": {"n_components": 2, "perplexity": 30, "init": "pca",
                       "learning_rate": "auto", "max_iter": 1000,
                       "method": "barnes_hut", "metric": "cosine",
                       "random_state": seed, "n_jobs": 1},
    })
    return methods


def sampled_2d_neighbors(coords, rows, k=K):
    """Euclidean TopK in 2D, excluding same-platform items and self."""
    coords = np.asarray(coords, dtype=np.float32)
    n = len(coords)
    with np.errstate(all="ignore"):
        # Squared Euclidean distance preserves ascending Euclidean order.
        scores = -(np.sum(coords * coords, axis=1)[:, None]
                   + np.sum(coords * coords, axis=1)[None, :] - 2 * coords @ coords.T)
    result = []
    for i in range(n):
        excluded = exclusion_mask(rows, i, i + 1, "cross", exclude_candidates=True)[0]
        result.append(rank_row(scores[i], excluded, k).astype(np.int32))
    return result


def _verify_embedding_source(receipt_path, catalog_path, vectors_path, keys_path,
                             catalog_rows, keys, vectors):
    """Reject inputs that do not match the pinned A1 selection and embedding receipt."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "embedding-receipt.v1" or receipt.get("variant") != "A1":
        raise SystemExit("embedding_receipt_must_be_pinned_A1")
    expected = {
        "catalog": sha256_file(catalog_path),
        "keys.json": sha256_file(keys_path),
        "vectors.npy": sha256_file(vectors_path),
    }
    if receipt.get("inputs", {}).get("catalog") != expected["catalog"]:
        raise SystemExit("catalog_hash_does_not_match_embedding_receipt")
    for name in ("keys.json", "vectors.npy"):
        if receipt.get("outputs", {}).get(name) != expected[name]:
            raise SystemExit(f"{name}_hash_does_not_match_embedding_receipt")
    if receipt.get("rows") != len(keys) or receipt.get("model", {}).get("dimensions") != vectors.shape[1]:
        raise SystemExit("embedding_receipt_shape_mismatch")
    selected_rows, selected_texts = full_catalog_inputs.select(catalog_rows, "A1")
    selected_keys = [row["problem_key"] for row in selected_rows]
    if selected_keys != keys:
        raise SystemExit("A1_source_selection_keys_do_not_match_vectors")
    selection_digest = full_catalog_inputs.input_digest(selected_rows, selected_texts)
    if receipt.get("inputs", {}).get("variant_input") != selection_digest:
        raise SystemExit("A1_source_selection_digest_mismatch")
    return {"receipt": str(receipt_path), "receipt_sha256": sha256_file(receipt_path),
            "variant": "A1", "source_selection_digest": selection_digest,
            "verified_hashes": expected}


def _peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def _plot_coordinates(path, coords, sample_platforms, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"atcoder": "#0072B2", "codeforces": "#D55E00", "leetcode": "#009E73"}
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for platform_name in PLATFORMS:
        mask = np.array([p == platform_name for p in sample_platforms])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.55,
                       color=colors[platform_name], label=f"{platform_name} (n={int(mask.sum())})",
                       rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("coordinate 1")
    ax.set_ylabel("coordinate 2")
    ax.legend(markerscale=2, frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args):
    keys_path, vectors_path, catalog_path = args.keys, args.vectors, args.catalog
    keys = json.loads(keys_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path, mmap_mode="r")
    if vectors.dtype != np.float32 or vectors.ndim != 2 or len(keys) != len(vectors):
        raise SystemExit("expected float32 matrix with one row per key")
    catalog_rows = [json.loads(line) for line in catalog_path.read_text(encoding="utf-8").splitlines()]
    catalog_by_key = {row["problem_key"]: row for row in catalog_rows}
    eligible = {row["problem_key"] for row in catalog_rows if row["eligibility"] == "eligible"}
    if not set(keys) <= eligible:
        raise SystemExit("vector keys include ineligible or absent catalog rows")
    source_verification = _verify_embedding_source(args.embedding_receipt, catalog_path,
                                                   vectors_path, keys_path, catalog_rows,
                                                   keys, vectors)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = Rows.from_catalog(catalog_rows, keys)
    platforms = [catalog_by_key[key]["platform"] for key in keys]
    platform_coverage = {p: int(sum(value == p for value in platforms)) for p in PLATFORMS}
    sample_indices = stratified_sample(keys, platforms, args.sample_size, args.seed)
    sample_vectors = _normalize(vectors[sample_indices])
    sample_platforms = [platforms[i] for i in sample_indices]
    sample_keys = [keys[i] for i in sample_indices]
    sample_rows = Rows.from_catalog([catalog_by_key[key] for key in sample_keys], sample_keys)
    np.save(args.output / "sample-indices.npy", sample_indices, allow_pickle=False)
    (args.output / "sample-keys.json").write_text(
        json.dumps(sample_keys, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "schema": "unified-map-reduction.v1",
        "method": "A1 title embedding dimensionality reduction ablation",
        "seed": args.seed,
        "thread_cap": 1,
        "inputs": {str(p): sha256_file(p) for p in (catalog_path, vectors_path, keys_path)},
        "source_verification": source_verification,
        "coverage": {"input_rows": len(keys), "input_dimensions": int(vectors.shape[1]),
                     "input_dtype": str(vectors.dtype), "platform_rows": platform_coverage,
                     "eligible_catalog_rows": int(len(eligible)),
                     "retrieval": "all vector rows queried; cross-platform targets only; same candidate group excluded",
                     "sample": {"rows": len(sample_indices), "requested_cap": args.sample_size,
                                "seed": args.seed, "platform_rows": {
                                    p: int(sum(v == p for v in sample_platforms)) for p in PLATFORMS},
                                "neighborhood_baseline": "exact cosine within sampled cohort; cross-platform targets; same candidate group excluded",
                                "2d_distance": "Euclidean; cross-platform targets; same candidate group excluded"}},
        "compression": {},
        "sampled_2d": {},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "machine": platform.machine(), "platform": platform.platform()},
        "implementation_sha256": sha256_file(Path(__file__)),
        "package_versions": {name: importlib.metadata.version(name) for name in
                              ("numpy", "scikit-learn", "umap-learn", "matplotlib", "threadpoolctl")},
        "persisted_artifacts": {
            "original384.top10.npy": "query-row order; target values are zero-based indices into input keys.json",
            "<method>.top10.npy": "query-row order; target values are zero-based indices into input keys.json",
            "sample-indices.npy": "source-row indices into input keys.json, in coordinate row order",
            "sample-keys.json": "problem keys in coordinate row order",
            "sample-<method>.npy": "2D coordinates in sample-keys.json row order",
            "transform_parameters": "recorded in receipt; PCA/SVD/UMAP/t-SNE fitted objects are not serialized",
        },
    }

    with threadpool_limits(limits=1):
        reference_started = time.perf_counter()
        original_neighbors = exact_neighbors(rows, vectors)
        original_matrix = np.full((len(original_neighbors), K), -1, dtype=np.int32)
        for i, values in enumerate(original_neighbors):
            original_matrix[i, :len(values)] = values
        np.save(args.output / "original384.top10.npy", original_matrix, allow_pickle=False)
        summary["compression"]["original384"] = {
            "dimension": int(vectors.shape[1]), "top10_preservation": {
                "queries": len(original_neighbors), "mean_top10_overlap": 1.0,
                "queries_with_exact_top10": len(original_neighbors),
                "queries_with_candidates": int(sum(len(v) > 0 for v in original_neighbors))},
            "seconds": round(time.perf_counter() - reference_started, 4),
            "estimated_score_block_bytes": int(min(512, len(keys)) * len(keys) * 4),
            "vector_bytes": int(vectors.nbytes),
        }
        model_vectors, fit_info = compression_models(vectors, args.seed)
        summary["compression"]["fit"] = fit_info
        for name, reduced in model_vectors.items():
            before_rss = _peak_rss_bytes()
            started = time.perf_counter()
            observed = exact_neighbors(rows, reduced)
            file_name = f"{name}.top10.npy"
            path = args.output / file_name
            matrix = np.full((len(observed), K), -1, dtype=np.int32)
            for i, values in enumerate(observed):
                matrix[i, :len(values)] = values
            np.save(path, matrix, allow_pickle=False)
            metric = top10_preservation(original_neighbors, observed)
            summary["compression"][name] = {
                "dimension": int(reduced.shape[1]), "top10_preservation": metric,
                "seconds": round(time.perf_counter() - started, 4),
                "estimated_score_block_bytes": int(min(512, len(keys)) * len(keys) * 4),
                "vector_bytes": int(reduced.nbytes), "neighbors_file": file_name,
                "neighbors_file_bytes": int(path.stat().st_size),
                "process_peak_rss_bytes": _peak_rss_bytes(),
                "peak_rss_before_bytes": before_rss,
            }
            del observed, matrix

        sample_reference = sampled_reference_neighbors(sample_vectors, sample_rows)
        coordinate_runs = coordinates_2d(sample_vectors, args.seed)
        for name, (coords, info) in coordinate_runs.items():
            coord_file = f"sample-{name}.npy"
            np.save(args.output / coord_file, coords, allow_pickle=False)
            neighbors = sampled_2d_neighbors(coords, sample_rows)
            metric = top10_preservation(sample_reference, neighbors)
            figure_file = f"sample-{name}.png"
            _plot_coordinates(args.output / figure_file, coords, sample_platforms,
                               f"A1 title vectors: {name} (n={len(coords)})")
            summary["sampled_2d"][name] = {
                **info, "sample_rows": len(coords), "distance": "Euclidean",
                "top10_overlap_vs_sampled_384d": metric,
                "coordinates_file": coord_file,
                "coordinates_file_bytes": int((args.output / coord_file).stat().st_size),
                "figure_file": figure_file,
                "figure_file_bytes": int((args.output / figure_file).stat().st_size),
                "process_peak_rss_bytes": _peak_rss_bytes(),
            }
            del coords, neighbors

    summary["outputs"] = {}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "receipt.json":
            summary["outputs"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    receipt_path = args.output / "receipt.json"
    receipt_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path,
                        default=repo / "private-learning/full-catalog/catalog-v1/catalog.v1.jsonl")
    parser.add_argument("--vectors", type=Path,
                        default=repo / "private-learning/full-catalog/embeddings-v1/A1/vectors.npy")
    parser.add_argument("--keys", type=Path,
                        default=repo / "private-learning/full-catalog/embeddings-v1/A1/keys.json")
    parser.add_argument("--embedding-receipt", type=Path,
                        default=repo / "research/full_catalog/receipts/embeddings-A1.v1.json")
    parser.add_argument("--output", type=Path,
                        default=repo / "private-learning/unified-map/reduction-v1")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    result = run(args)
    print(json.dumps({"receipt": str(args.output / "receipt.json"),
                      "compression": {k: v.get("top10_preservation") for k, v in
                                      result["compression"].items() if isinstance(v, dict)},
                      "sampled_2d": {k: v["top10_overlap_vs_sampled_384d"] for k, v in
                                     result["sampled_2d"].items()}}, indent=2))


if __name__ == "__main__":
    main()
