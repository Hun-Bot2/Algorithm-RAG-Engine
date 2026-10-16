"""Reproducible, private diagnostics for the existing LeetCode vector index.

Run from the repository root with ``.venv/bin/python research/vector_geometry/experiment.py``.
The study reads, but never modifies, the production index. Output contains
aggregate measurements and plots, never full embedding coordinates.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import faiss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import umap
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def topk(values: np.ndarray, k: int = 10, *, largest: bool = False) -> np.ndarray:
    """Exact ordered top-k; stable index tie-break for reproducibility."""
    if largest:
        return np.lexsort((np.arange(len(values)), -values))[:k]
    return np.lexsort((np.arange(len(values)), values))[:k]


def normalized(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-length vector is incompatible with cosine geometry")
    return np.asarray(values / norms, dtype=np.float32)


def cosine_neighbors(values: np.ndarray, k: int = 10) -> np.ndarray:
    unit = normalized(values)
    similarity = unit @ unit.T
    if not np.isfinite(similarity).all():
        raise AssertionError("nonfinite cosine matrix")
    np.fill_diagonal(similarity, -np.inf)
    return np.array([topk(row, k, largest=True) for row in similarity], dtype=np.int32)


def euclidean_neighbors(values: np.ndarray, k: int = 10) -> np.ndarray:
    """Translation-invariant neighbors for visualization coordinates."""
    distance = pairwise_distances(values, metric="euclidean", n_jobs=1)
    if not np.isfinite(distance).all():
        raise AssertionError("nonfinite Euclidean matrix")
    np.fill_diagonal(distance, np.inf)
    return np.array([topk(row, k) for row in distance], dtype=np.int32)


def partial_topk(values: np.ndarray, k: int = 10) -> np.ndarray:
    candidates = np.argpartition(values, k - 1)[:k]
    return candidates[np.lexsort((candidates, values[candidates]))]


def overlap(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("neighbor arrays must have identical shape")
    k = left.shape[1]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(left, right)]))


def quantiles_ms(samples_ns: list[int]) -> dict:
    milliseconds = np.asarray(samples_ns, dtype=np.float64) / 1e6
    return {"median_ms": float(np.median(milliseconds)), "p95_ms": float(np.percentile(milliseconds, 95))}


def benchmark_search(vectors: np.ndarray, query_ids: np.ndarray) -> dict:
    """Compare exact raw squared-L2 search, excluding each query's own row."""
    start_build = time.perf_counter_ns()
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    build_ms = (time.perf_counter_ns() - start_build) / 1e6
    queries = vectors[query_ids]
    sample_numpy: list[int] = []
    sample_faiss: list[int] = []
    sample_dot: list[int] = []
    norms_sq = np.sum(vectors * vectors, axis=1)

    def numpy_search(query: np.ndarray, own: int):
        # Same squared-L2 expression as src/main/generate_index_map.py.
        distance = np.sum((vectors - query) ** 2, axis=1)
        distance[own] = np.inf
        ids = partial_topk(distance)
        return ids, distance[ids]

    def dot_search(query: np.ndarray, own: int):
        # Algebraically equivalent alternative; rounding differs slightly.
        distance = norms_sq + float(query @ query) - 2 * (vectors @ query)
        distance[own] = np.inf
        ids = partial_topk(distance)
        return ids, distance[ids]

    numpy_search(queries[0], int(query_ids[0]))
    dot_search(queries[0], int(query_ids[0]))
    index.search(queries[:1], 11)
    agreements = []
    max_abs_distance_error = 0.0
    dot_set_agreements = []
    # Interleave methods to reduce drift from thermal and background load.
    for _ in range(3):
        for own, query in zip(query_ids, queries):
            own = int(own)
            start = time.perf_counter_ns()
            numpy_ids, numpy_distance = numpy_search(query, own)
            sample_numpy.append(time.perf_counter_ns() - start)
            start = time.perf_counter_ns()
            distances, faiss_ids = index.search(query[None, :], 11)
            sample_faiss.append(time.perf_counter_ns() - start)
            valid = faiss_ids[0] != own
            faiss_ids = faiss_ids[0][valid][:10]
            faiss_distance = distances[0][valid][:10]
            start = time.perf_counter_ns()
            dot_ids, _ = dot_search(query, own)
            sample_dot.append(time.perf_counter_ns() - start)
            agreements.append(bool(np.array_equal(numpy_ids, faiss_ids)))
            dot_set_agreements.append(bool(set(numpy_ids) == set(dot_ids)))
            if not np.array_equal(numpy_ids, faiss_ids):
                raise AssertionError("FAISS FlatL2 and NumPy disagree on exact top-10 ranking")
            max_abs_distance_error = max(max_abs_distance_error, float(np.max(np.abs(numpy_distance - faiss_distance))))
    reference_error = 0.0
    for own in query_ids[:16]:
        own = int(own)
        # Independent direct float64 elementwise reduction; no BLAS/matmul.
        direct = np.sum((vectors.astype(np.float64) - vectors[own].astype(np.float64)) ** 2, axis=1)
        identity = norms_sq + float(np.sum(vectors[own] ** 2)) - 2 * (vectors @ vectors[own])
        if not np.isfinite(identity).all():
            raise AssertionError("nonfinite norm/dot distances")
        reference_error = max(reference_error, float(np.max(np.abs(direct - identity))))
        direct[own] = identity[own] = np.inf
        if set(topk(direct)) != set(topk(identity)):
            raise AssertionError("independent float64 top-10 check failed")
    if reference_error > 1e-5:
        raise AssertionError("norm/dot exceeds independent float64 tolerance")
    return {
        "independent_float64_query_count": min(16, len(query_ids)),
        "independent_float64_max_distance_error": reference_error,
        "metric": "raw squared L2, exact, self excluded",
        "query_count": int(len(query_ids)),
        "repeats": 3,
        "numpy_production_expression": quantiles_ms(sample_numpy),
        "faiss_flat_l2": quantiles_ms(sample_faiss),
        "numpy_norm_dot_identity": quantiles_ms(sample_dot),
        "numpy_faiss_identical_order_fraction": float(np.mean(agreements)),
        "numpy_dot_identical_top10_set_fraction": float(np.mean(dot_set_agreements)),
        "max_abs_top10_squared_distance_difference": max_abs_distance_error,
        "faiss_build_ms": build_ms,
        "faiss_index_bytes": int(index.ntotal * vectors.shape[1] * 4),
    }


def _scatter_grid(projections: dict[str, np.ndarray], labels: np.ndarray, out: Path) -> None:
    colors = {"Easy": "#238b45", "Medium": "#fd8d3c", "Hard": "#cb181d"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, (name, coords) in zip(axes.flat, projections.items()):
        for label in ("Easy", "Medium", "Hard"):
            mask = labels == label
            ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.48,
                       linewidths=0, color=colors[label], label=label, rasterized=True)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
    axes.flat[-1].axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, legend_labels, loc="center", title="Difficulty", frameon=False)
    fig.suptitle("LeetCode embedding projections by difficulty (diagnostic only)", fontsize=16)
    fig.savefig(out / "projections.png", dpi=160)
    plt.close(fig)


def _summary_figure(reductions: list[dict], projections: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for method, color in (("PCA", "#2166ac"), ("TruncatedSVD", "#b2182b")):
        rows = [r for r in reductions if r["method"] == method]
        axes[0].plot([r["dimensions"] for r in rows], [r["cosine_overlap_at_10"] for r in rows],
                     marker="o", label=method, color=color)
    axes[0].set(xlabel="Reduced dimensions", ylabel="Mean overlap@10 with original cosine neighbors",
                title="Local neighborhood preservation", ylim=(0, 1))
    axes[0].legend()
    names = list(projections)
    axes[1].barh(names, [projections[n]["euclidean_overlap_at_10"] for n in names], color="#677eab")
    axes[1].set(xlabel="Mean overlap@10 with original cosine neighbors", title="2D Euclidean neighbors vs original cosine", xlim=(0, 1))
    axes[1].invert_yaxis()
    fig.savefig(out / "overlap.png", dpi=160)
    plt.close(fig)


def plot_search(search: dict, out: Path) -> None:
    names = ["NumPy production expression", "NumPy norm/dot identity", "FAISS Flat L2"]
    keys = ["numpy_production_expression", "numpy_norm_dot_identity", "faiss_flat_l2"]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    y = np.arange(len(names))
    med = [search[key]["median_ms"] for key in keys]
    p95 = [search[key]["p95_ms"] for key in keys]
    ax.barh(y, med, color="#2166ac", label="Median")
    ax.scatter(p95, y, color="#b2182b", marker="D", label="p95")
    for row, value in enumerate(med):
        ax.text(max(value, p95[row]) + max(p95) * .02, row, f" {value:.3f} ms", va="center")
    ax.set(yticks=y, yticklabels=names, xlabel="Wall time per warm query (ms)",
           title="Exact raw L2: 128 catalog queries × 3 repeats, one thread")
    ax.set_xlim(0, max(p95) * 1.3)
    ax.invert_yaxis(); ax.legend()
    fig.savefig(out / "search_latency.png", dpi=160); plt.close(fig)


def _run(root: Path, out: Path) -> dict:
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    vector_path = root / "indexes/numpy_leetcode/vectors.npy"
    metadata_path = root / "indexes/numpy_leetcode/metadata.json"
    before = {"vectors.npy": sha256(vector_path), "metadata.json": sha256(metadata_path)}
    registry = json.loads((root / "assets/source_registry.v1.json").read_text())
    for path in (vector_path, metadata_path):
        entry = next(row for row in registry["assets"] if row["path"] == str(path.relative_to(root)))
        if before[path.name] != entry["sha256"]:
            raise AssertionError("historical asset differs from immutable registry: " + path.name)
    vectors = np.ascontiguousarray(np.load(vector_path), dtype=np.float32)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if vectors.ndim != 2 or len(metadata) != len(vectors):
        raise AssertionError("index and metadata have incompatible shape")
    n, d = vectors.shape
    if not np.isfinite(vectors).all():
        raise AssertionError("nonfinite index values")
    difficulties = np.array([row.get("difficulty") for row in metadata], dtype=object)
    counts = Counter(str(label) if label is not None else "missing" for label in difficulties)
    valid = np.isin(difficulties, ("Easy", "Medium", "Hard"))
    rng = np.random.default_rng(17)
    query_ids = rng.choice(n, size=min(128, n), replace=False)
    result = {
        "purpose": "offline index engineering and geometric diagnostics; no recommendation accuracy claim",
        "input_sha256_before": before,
        "shape": [n, d], "dtype": str(vectors.dtype),
        "raw_vector_bytes": int(vectors.nbytes),
        "vector_norm_range": [float(np.linalg.norm(vectors,axis=1).min()), float(np.linalg.norm(vectors,axis=1).max())],
        "metadata_missingness": {
            "difficulty_counts": dict(counts),
            "missing_id": sum(not row.get("id") for row in metadata),
            "missing_tags": sum(not row.get("tags") for row in metadata),
            "missing_title": sum(not row.get("title") for row in metadata),
        },
        "query_row_ids": query_ids.tolist(),
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__, "faiss": faiss.__version__, "umap": umap.__version__,
                     "matplotlib": matplotlib.__version__, "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        "parameters": {"seed": 17, "repeat_seed": 29, "threads": 1, "neighbors": 10,
                       "pca_svd_dimensions": [32, 64, 128, 256], "tsne_perplexity": 30,
                       "tsne_max_iter": 750, "tsne_init": "random", "umap_n_neighbors": 15, "umap_min_dist": 0.1},
    }
    experiment_started = time.perf_counter()
    with threadpool_limits(limits=1):
        faiss.omp_set_num_threads(1)
        result["search"] = benchmark_search(vectors, query_ids)
        original_neighbors = cosine_neighbors(vectors)
        reductions = []
        projections = {}
        projection_metrics = {}
        pca_cache = {}
        svd_cache = {}
        for method, cls, cache in (("PCA", PCA, pca_cache), ("TruncatedSVD", TruncatedSVD, svd_cache)):
            for components in (32, 64, 128, 256):
                start = time.perf_counter()
                model = cls(n_components=components, svd_solver="randomized", random_state=17) if method == "PCA" else cls(n_components=components, algorithm="randomized", random_state=17)
                reduced = np.asarray(model.fit_transform(vectors), dtype=np.float32)
                fit_seconds = time.perf_counter() - start
                if not np.isfinite(reduced).all() or not np.isfinite(model.components_).all():
                    raise AssertionError("nonfinite reduction")
                gram = np.einsum("ij,kj->ik", model.components_.astype(np.float64), model.components_.astype(np.float64), optimize=False)
                if not np.allclose(gram, np.eye(components), atol=1e-4):
                    raise AssertionError("reduction components are not orthonormal")
                neighbors = cosine_neighbors(reduced)
                reductions.append({"method": method, "dimensions": components,
                                   "fit_transform_seconds": fit_seconds,
                                   "vector_storage_bytes": int(reduced.nbytes),
                                   "storage_fraction_of_original": float(reduced.nbytes / vectors.nbytes),
                                   "explained_variance_ratio_sum": float(np.sum(model.explained_variance_ratio_)),
                                   "cosine_overlap_at_10": overlap(original_neighbors, neighbors)})
                if components == 256:
                    cache["two_d"] = reduced[:, :2]
        result["reductions"] = reductions
        # Distinct 2D fits avoid treating randomized 256-D prefixes as a 2D model.
        for method, cls in (("PCA 2D", PCA), ("SVD 2D", TruncatedSVD)):
            model = cls(n_components=2, svd_solver="randomized", random_state=17) if method.startswith("PCA") else cls(n_components=2, algorithm="randomized", random_state=17)
            start = time.perf_counter()
            coords = np.asarray(model.fit_transform(vectors), dtype=np.float32)
            projections[method] = coords
            projection_metrics[method] = {"fit_seconds": time.perf_counter() - start,
                                          "euclidean_overlap_at_10": overlap(original_neighbors, euclidean_neighbors(coords))}
        if len(set(difficulties[valid])) == 3:
            model = LinearDiscriminantAnalysis(n_components=2)
            start = time.perf_counter()
            coords = np.full((n, 2), np.nan, dtype=np.float32)
            coords[valid] = model.fit_transform(vectors[valid], difficulties[valid]).astype(np.float32)
            projections["Difficulty LDA 2D"] = coords
            valid_ids = np.flatnonzero(valid)
            lda_reference = original_neighbors[valid_ids]
            # Some original neighbors may lack labels, but that missingness remains visible.
            lda_neighbors_local = euclidean_neighbors(coords[valid])
            lda_neighbors_global = valid_ids[lda_neighbors_local]
            projection_metrics["Difficulty LDA 2D"] = {"fit_seconds": time.perf_counter() - start,
                    "euclidean_overlap_at_10": overlap(lda_reference, lda_neighbors_global),
                    "fitted_rows": int(valid.sum()), "in_sample_supervised_leakage": True}
        pca50 = PCA(n_components=50, svd_solver="randomized", random_state=17).fit_transform(vectors).astype(np.float32)
        pca50_neighbors = euclidean_neighbors(pca50)
        result["pca50_euclidean_overlap_at_10"] = overlap(original_neighbors, pca50_neighbors)
        for seed in (17, 29):
            start = time.perf_counter()
            tsne = TSNE(n_components=2, perplexity=30, max_iter=750, init="random", learning_rate="auto", random_state=seed)
            coords = tsne.fit_transform(pca50).astype(np.float32)
            name = "t-SNE 2D" if seed == 17 else "t-SNE 2D repeat"
            neighbors = euclidean_neighbors(coords)
            projection_metrics[name] = {"fit_seconds": time.perf_counter() - start,
                                         "euclidean_overlap_at_10": overlap(original_neighbors, neighbors),
                                         "euclidean_overlap_at_10_vs_pca50": overlap(pca50_neighbors, neighbors),
                                         "kl_divergence": float(tsne.kl_divergence_)}
            if seed == 17:
                projections[name] = coords
                tsne_neighbors = neighbors
            else:
                result["tsne_seed_neighbor_overlap_at_10"] = overlap(tsne_neighbors, neighbors)
        for seed in (17, 29):
            start = time.perf_counter()
            model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=seed, n_jobs=1)
            coords = model.fit_transform(pca50).astype(np.float32)
            name = "UMAP 2D" if seed == 17 else "UMAP 2D repeat"
            neighbors = euclidean_neighbors(coords)
            projection_metrics[name] = {"fit_seconds": time.perf_counter() - start,
                                         "euclidean_overlap_at_10": overlap(original_neighbors, neighbors),
                                         "euclidean_overlap_at_10_vs_pca50": overlap(pca50_neighbors, neighbors)}
            if seed == 17:
                projections[name] = coords
                umap_neighbors = neighbors
            else:
                result["umap_seed_neighbor_overlap_at_10"] = overlap(umap_neighbors, neighbors)
        result["projections"] = projection_metrics
        plot_search(result["search"], out)
        _scatter_grid(projections, difficulties, out)
        _summary_figure(reductions, {k: v for k, v in projection_metrics.items() if k in projections}, out)
    after = {"vectors.npy": sha256(vector_path), "metadata.json": sha256(metadata_path)}
    result["input_sha256_after"] = after
    if before != after:
        raise AssertionError("input asset changed during study")
    result["total_wall_seconds"] = time.perf_counter() - experiment_started
    result["projection_neighbor_metric"] = "Euclidean in 2D vs cosine in original; seeds compared using Euclidean"
    result["compression_neighbor_metric"] = "Cosine in reduced space vs cosine in original; centered PCA changes origin"
    result["search_timing_scope"] = "Warm resident arrays; no file loading, imports, network, query encoding, concurrency or OCI measurement; fixed method order; index build reported separately"
    result["figures"] = ["projections.png", "overlap.png", "search_latency.png"]
    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def run(root: Path, out: Path) -> dict:
    # Retain warnings as inspectable evidence rather than flooding notebook output.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = _run(root, out)
    counts = Counter((type(item.message).__name__, str(item.message)) for item in captured)
    result["runtime_warnings"] = [
        {"category": category, "message": message, "count": count}
        for (category, message), count in sorted(counts.items())
    ]
    result["numerical_validation"] = "All cosine/Euclidean neighbor matrices and reductions finite; reduction components orthonormal via float64 non-BLAS einsum; 16 sampled raw L2 queries cross-checked with direct float64 elementwise distances"
    (Path(out) / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Numerical checks passed; {sum(counts.values())} library warnings retained in results.json ({len(counts)} distinct messages).")
    return result


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[2]
    output = Path(os.environ.get("VECTOR_STUDY_OUTPUT", Path(__file__).parent / "output"))
    result = run(repository, output)
    print(json.dumps({"output": str(output), "search": result["search"],
                      "input_hashes_unchanged": result["input_sha256_before"] == result["input_sha256_after"]}, indent=2))
