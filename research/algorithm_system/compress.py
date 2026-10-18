"""Compression ablation for a manifest-bound 384D statement retrieval baseline.

The CLI refuses to run unless a completed minilm_chunk baseline, its corpus,
vector matrix, and Top50 retrieval artifact are all present and hash-bound in the
run manifest. Outputs are private, fresh-run artifacts; this module performs no
embedding and makes no relevance or model-promotion claim.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import platform as host_platform
import resource
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.full_catalog.retrieve import (  # noqa: E402
    KS, Rows, dense_cosine, exclusion_mask, rank_row, topk, write_results,
)

PCA_DIMS = (256, 128, 64)
SVD_DIMS = (256, 128, 64)
RANDOM_PROJECTION_SEED = 17
LATENCY_SAMPLE_SIZE = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_manifest(run_dir: Path) -> tuple[dict, Path, Path, Path]:
    manifest_path = run_dir / "manifest.json"
    corpus_path = run_dir / "corpus.jsonl"
    vectors_path = run_dir / "minilm_chunk.npy"
    if not all(path.is_file() for path in (manifest_path, corpus_path, vectors_path)):
        raise ValueError("missing_baseline_manifest_corpus_or_vectors")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = manifest.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("manifest_missing_baseline_gate")
    if baseline.get("status") != "complete":
        raise ValueError("full_statement_384d_baseline_not_complete")
    if baseline.get("method") != "minilm_chunk":
        raise ValueError("manifest_baseline_method_must_be_minilm_chunk")
    if baseline.get("dimensions") != 384:
        raise ValueError("manifest_baseline_dimension_must_be_384")
    top_rel = baseline.get("top50_path")
    if not isinstance(top_rel, str):
        raise ValueError("manifest_missing_baseline_top50_path")
    rel_path = PurePosixPath(top_rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError("unsafe_baseline_top50_path")
    top50_path = run_dir.joinpath(*rel_path.parts)
    if not top50_path.is_file():
        raise ValueError("manifest_top50_file_missing")
    expected = {
        "corpus_sha256": sha256_file(corpus_path),
        "vector_sha256": sha256_file(vectors_path),
        "top50_sha256": sha256_file(top50_path),
    }
    for field, actual in expected.items():
        if baseline.get(field) != actual:
            raise ValueError(f"baseline_manifest_hash_mismatch:{field}")
    return manifest, corpus_path, vectors_path, top50_path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_jsonl:{path.name}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"invalid_jsonl_record:{path.name}:{line_number}")
            rows.append(row)
    return rows


def _load_rows(corpus_path: Path, vectors_path: Path):
    corpus = _read_jsonl(corpus_path)
    keys = [row.get("problem_key") for row in corpus]
    platforms = [row.get("platform") for row in corpus]
    hashes = [row.get("content_sha256") for row in corpus]
    if not corpus or any(not isinstance(k, str) or not k for k in keys):
        raise ValueError("corpus_missing_problem_key")
    if any(not isinstance(p, str) or not p for p in platforms):
        raise ValueError("corpus_missing_platform")
    if any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes):
        raise ValueError("corpus_invalid_content_sha256")
    # Reuse the production research ranking contract: stable key order, no self
    # matches via per-row content representatives, and no unverified candidates.
    rows = Rows(keys, platforms, hashes, [None] * len(keys))
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape != (len(rows), 384):
        raise ValueError(f"baseline_vector_shape_must_be_n_by_384:{vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("baseline_vectors_not_finite")
    # dense_cosine verifies normalization; retaining its check makes original and
    # compressed rankings use exactly the same cosine retrieval implementation.
    dense_cosine(vectors)
    return corpus, rows, np.asarray(vectors, dtype=np.float32)


def _load_baseline_top50(path: Path, rows: Rows) -> dict[str, list[str]]:
    records = _read_jsonl(path)
    baseline = {}
    key_set = set(rows.keys)
    for record in records:
        query = record.get("query")
        pairs = record.get("results")
        if query not in key_set or query in baseline or not isinstance(pairs, list):
            raise ValueError("baseline_top50_invalid_or_duplicate_query")
        targets = []
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) < 1 or pair[0] not in key_set:
                raise ValueError(f"baseline_top50_invalid_target:{query}")
            targets.append(pair[0])
        if len(targets) > 50 or len(set(targets)) != len(targets):
            raise ValueError(f"baseline_top50_invalid_length_or_duplicates:{query}")
        baseline[query] = targets
    if set(baseline) != key_set:
        raise ValueError("baseline_top50_queries_do_not_match_corpus")
    return baseline


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("projection_produced_invalid_or_zero_vector")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def _numerical_rank(singular_values: np.ndarray, n: int, d: int) -> int:
    if not singular_values.size or singular_values[0] <= 0:
        return 0
    tol = max(n, d) * np.finfo(np.float64).eps
    return int(np.sum(singular_values > singular_values[0] * tol))


def svd_project(vectors: np.ndarray, dimensions: int):
    """Uncentered truncated SVD fit on all rows, then L2-normalized projection."""
    x = np.asarray(vectors, dtype=np.float64)
    n, d = x.shape
    if dimensions < 1 or dimensions > d:
        raise ValueError("invalid_projection_dimension")
    if n < dimensions:
        raise ValueError(f"insufficient_rows_for_svd:{n}:{dimensions}")
    started = time.perf_counter()
    with np.errstate(all="ignore"):
        _, singular_values, right_vectors = np.linalg.svd(x, full_matrices=False)
    rank = _numerical_rank(singular_values, n, d)
    if rank < dimensions:
        raise ValueError(f"insufficient_rank_for_svd:{rank}:{dimensions}")
    components = np.asarray(right_vectors[:dimensions].T, dtype=np.float32)
    with np.errstate(all="ignore"):
        projected = np.asarray(vectors, dtype=np.float32) @ components
    normalized = _normalize_vectors(projected)
    return normalized, {"rank": rank, "fit_seconds": time.perf_counter() - started,
                       "transform_bytes": int(components.nbytes), "components_bytes": int(components.nbytes)}


def _deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> tuple[int, str]:
    """Write a timestamp-stable, uncompressed NPZ suitable for transform hashing."""
    with Path(path).open("xb") as handle:
        with zipfile.ZipFile(handle, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())
    return path.stat().st_size, sha256_file(path)


def pca_project(vectors: np.ndarray, dimensions: int):
    """Centered PCA/SVD projection, returning unit vectors and fit diagnostics."""
    x = np.asarray(vectors, dtype=np.float64)
    n, d = x.shape
    if dimensions < 1 or dimensions > d:
        raise ValueError("invalid_projection_dimension")
    if n - 1 < dimensions:
        raise ValueError(f"insufficient_rows_for_pca:{n}:{dimensions}")
    started = time.perf_counter()
    mean = x.mean(axis=0)
    centered = x - mean
    with np.errstate(all="ignore"):
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    if not singular_values.size or singular_values[0] <= 0:
        raise ValueError("pca_input_has_zero_rank")
    rank = _numerical_rank(singular_values, n, d)
    if rank < dimensions:
        raise ValueError(f"insufficient_rank_for_pca:{rank}:{dimensions}")
    components = np.asarray(right_vectors[:dimensions].T, dtype=np.float32)
    center = np.asarray(mean, dtype=np.float32)
    with np.errstate(all="ignore"):
        projected = (np.asarray(vectors, dtype=np.float32) - center) @ components
    normalized = _normalize_vectors(projected)
    elapsed = time.perf_counter() - started
    return normalized, {"rank": rank, "fit_seconds": elapsed,
                        "transform_bytes": int(components.nbytes + center.nbytes),
                        "components_bytes": int(components.nbytes), "center_bytes": int(center.nbytes)}


def gaussian_project(vectors: np.ndarray, dimensions: int, seed=RANDOM_PROJECTION_SEED):
    """Seeded Gaussian random projection with an explicit materialized matrix."""
    x = np.asarray(vectors, dtype=np.float32)
    n, d = x.shape
    if dimensions < 1 or dimensions > d:
        raise ValueError("invalid_projection_dimension")
    if n - 1 < dimensions:
        raise ValueError(f"insufficient_rows_for_random_projection_comparison:{n}:{dimensions}")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((d, dimensions), dtype=np.float32) / np.sqrt(np.float32(dimensions))
    with np.errstate(all="ignore"):
        projected = x @ matrix
    normalized = _normalize_vectors(projected)
    elapsed = time.perf_counter() - started
    return normalized, {"fit_seconds": elapsed, "transform_bytes": int(matrix.nbytes),
                        "matrix_bytes": int(matrix.nbytes), "seed": int(seed),
                        "matrix_distribution": "independent standard normal / sqrt(dimensions)"}


def _latency_sample(vectors: np.ndarray, rows: Rows, seed=RANDOM_PROJECTION_SEED,
                    sample_size=LATENCY_SAMPLE_SIZE) -> dict:
    n = len(rows)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, min(n, sample_size), replace=False))

    def query(index):
        with np.errstate(all="ignore"):
            scores = np.asarray(vectors[index] @ vectors.T, dtype=np.float32)
        excluded = exclusion_mask(rows, int(index), int(index + 1), "cross", True)[0]
        return rank_row(scores, excluded, 50)

    for index in indices[:min(5, len(indices))]:
        query(int(index))
    elapsed = []
    for index in indices:
        start = time.perf_counter_ns()
        query(int(index))
        elapsed.append(time.perf_counter_ns() - start)
    values = np.asarray(elapsed, dtype=np.float64) / 1_000_000
    return {"sample_queries": len(indices), "seed": seed, "warmup_queries": min(5, len(indices)),
            "median_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, .95)),
            "mean_ms": float(values.mean()),
            "measurement": "single-query exact blockwise cosine+mask+stable-top50 after warmup"}


def _overlap_stream(rows: Rows, results, baseline: dict[str, list[str]], accum: dict[int, list[float]]):
    for query_index, ranked in results:
        query = rows.keys[query_index]
        actual = [rows.keys[target] for target, _ in ranked]
        reference = baseline[query]
        for k in KS:
            a, b = actual[:k], reference[:k]
            denominator = min(k, len(a), len(b))
            if denominator:
                accum[k].append(len(set(a) & set(b)) / denominator)
        yield query_index, ranked


def compare_method(name: str, vectors: np.ndarray, rows: Rows, baseline: dict[str, list[str]],
                   output_dir: Path, *, block_size=256) -> dict:
    if vectors.ndim != 2 or vectors.shape[0] != len(rows) or not np.isfinite(vectors).all():
        raise ValueError("method_vector_shape_or_finiteness_error")
    dense_block = dense_cosine(vectors)
    # Time exact retrieval separately from disk writes; persist the same deterministic
    # ranking in a second pass so storage latency does not contaminate compute time.
    started = time.perf_counter()
    for _ in topk(rows, dense_block, 50, "cross", block_size=block_size, exclude_candidates=True):
        pass
    retrieval_seconds = time.perf_counter() - started
    latency = _latency_sample(vectors, rows)
    overlaps = {k: [] for k in KS}
    method_dir = output_dir / name
    results = topk(rows, dense_block, 50, "cross", block_size=block_size, exclude_candidates=True)
    output_hashes, counts = write_results(rows, _overlap_stream(rows, results, baseline, overlaps), method_dir)
    return {"status": "complete", "rows": len(rows), "dimensions": int(vectors.shape[1]),
            "retrieval_seconds": retrieval_seconds, "query_latency": latency,
            "neighbor_overlap": {f"top{k}": {"mean_fraction": float(np.mean(overlaps[k])) if overlaps[k] else None,
                                               "queries": len(overlaps[k]),
                                               "comparable_query_count": len(overlaps[k])}
                                 for k in KS},
            "outputs": output_hashes, "counts": counts}


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def run(run_dir: Path, output_dir: Path, *, block_size=256) -> dict:
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    private_root = (ROOT / "private-learning" / "algorithm-system").resolve()
    if not output_dir.is_relative_to(private_root) or output_dir == private_root:
        raise ValueError("output_must_be_new_directory_under_private_algorithm_system")
    if output_dir.exists():
        raise FileExistsError(f"output_exists:{output_dir}")
    if output_dir.is_relative_to(run_dir) or run_dir.is_relative_to(output_dir):
        raise ValueError("input_run_and_output_must_be_separate")
    if block_size < 1:
        raise ValueError("block_size_must_be_positive")
    manifest, corpus_path, vectors_path, top50_path = _validate_manifest(run_dir)
    corpus, rows, vectors = _load_rows(corpus_path, vectors_path)
    baseline = _load_baseline_top50(top50_path, rows)
    n = len(rows)
    # Fit centered PCA and uncentered SVD over the same transductive population.
    started = time.perf_counter()
    x64 = np.asarray(vectors, dtype=np.float64)
    mean = x64.mean(axis=0)
    centered = x64 - mean
    with np.errstate(all="ignore"):
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    rank = _numerical_rank(singular_values, n, vectors.shape[1])
    pca_fit_seconds = time.perf_counter() - started
    svd_started = time.perf_counter()
    with np.errstate(all="ignore"):
        _, svd_singular_values, svd_right_vectors = np.linalg.svd(x64, full_matrices=False)
    svd_fit_seconds = time.perf_counter() - svd_started
    svd_rank = _numerical_rank(svd_singular_values, n, vectors.shape[1])
    pca_max_estimable = min(n - 1, rank)
    svd_max_estimable = min(n, svd_rank)

    output_dir.mkdir(parents=True, exist_ok=False)
    method_results = {}
    method_specs = {"original384": {"kind": "original"}}
    pca_fit = {"rank": rank, "fit_rows": n, "fit_seconds": pca_fit_seconds,
               "population": "all corpus queries and candidates (transductive)",
               "leakage_limit": "PCA fit includes every corpus row; not a held-out inductive transform."}
    for dimensions in PCA_DIMS:
        if dimensions > pca_max_estimable:
            method_results[f"pca{dimensions}"] = {"status": "not_estimable", "required_rank_and_rows": dimensions,
                                                   "observed_rank": rank, "observed_rows_minus_one": n - 1}
            continue
        components = np.asarray(right_vectors[:dimensions].T, dtype=np.float32)
        center = np.asarray(mean, dtype=np.float32)
        method_specs[f"pca{dimensions}"] = {"kind": "pca", "dimensions": dimensions,
                                            "components": components, "center": center,
                                            "rank": rank, "fit_population": pca_fit["population"],
                                            "fit_seconds_shared": pca_fit_seconds,
                                            "transform_bytes": int(components.nbytes + center.nbytes),
                                            "components_bytes": int(components.nbytes), "center_bytes": int(center.nbytes)}
    for dimensions in PCA_DIMS:
        if dimensions > pca_max_estimable:
            method_results[f"rp{dimensions}_seed{RANDOM_PROJECTION_SEED}"] = {
                "status": "not_estimable", "required_rank_and_rows": dimensions,
                "observed_rank": rank, "observed_rows_minus_one": n - 1}
            continue
        method_specs[f"rp{dimensions}_seed{RANDOM_PROJECTION_SEED}"] = {
            "kind": "random_projection", "dimensions": dimensions, "rank": rank}

    svd_fit = {"rank": svd_rank, "fit_rows": n, "fit_seconds": svd_fit_seconds,
               "population": "all corpus queries and candidates (transductive)",
               "centering": "none",
               "leakage_limit": "SVD fit includes every corpus row; not a held-out inductive transform."}
    for dimensions in SVD_DIMS:
        if dimensions > svd_max_estimable:
            method_results[f"svd{dimensions}"] = {"status": "not_estimable", "required_rank_and_rows": dimensions,
                                                   "observed_rank": svd_rank, "observed_rows": n}
            continue
        components = np.asarray(svd_right_vectors[:dimensions].T, dtype=np.float32)
        method_specs[f"svd{dimensions}"] = {"kind": "svd", "dimensions": dimensions,
                                            "components": components, "rank": svd_rank,
                                            "fit_population": svd_fit["population"],
                                            "fit_seconds_shared": svd_fit_seconds,
                                            "transform_bytes": int(components.nbytes),
                                            "components_bytes": int(components.nbytes)}

    for name, spec in method_specs.items():
        peak_before_method = _peak_rss_bytes()
        transform_started = time.perf_counter()
        if spec["kind"] == "original":
            method_vectors = vectors
            metadata = {"fit_population": "none; original baseline vectors", "transform_bytes": 0}
        elif spec["kind"] == "pca":
            with np.errstate(all="ignore"):
                method_vectors = _normalize_vectors((vectors - spec["center"]) @ spec["components"])
            metadata = {k: v for k, v in spec.items() if k not in {"kind", "components", "center", "dimensions"}}
            metadata["fit_population"] = metadata.pop("fit_population")
            metadata["dimensions"] = spec["dimensions"]
        elif spec["kind"] == "svd":
            with np.errstate(all="ignore"):
                method_vectors = _normalize_vectors(vectors @ spec["components"])
            metadata = {k: v for k, v in spec.items() if k not in {"kind", "components", "dimensions"}}
            metadata["dimensions"] = spec["dimensions"]
        else:
            dimensions = spec["dimensions"]
            rng = np.random.default_rng(RANDOM_PROJECTION_SEED)
            matrix = rng.standard_normal((384, dimensions), dtype=np.float32) / np.sqrt(np.float32(dimensions))
            with np.errstate(all="ignore"):
                method_vectors = _normalize_vectors(vectors @ matrix)
            metadata = {"rank": rank, "seed": RANDOM_PROJECTION_SEED,
                        "matrix_distribution": "independent standard normal / sqrt(dimensions)",
                        "transform_bytes": int(matrix.nbytes), "matrix_bytes": int(matrix.nbytes),
                        "fit_population": "none; fixed seeded random matrix", "dimensions": dimensions}
        transform_seconds = time.perf_counter() - transform_started
        transform_path = None
        if spec["kind"] == "pca":
            transform_arrays = {"mean": spec["center"], "components": spec["components"]}
        elif spec["kind"] == "svd":
            transform_arrays = {"components": spec["components"]}
        elif spec["kind"] == "random_projection":
            transform_arrays = {"matrix": matrix}
        else:
            transform_arrays = None
        if transform_arrays is not None:
            transform_path = output_dir / f"{name}.transform.npz"
            transform_file_bytes, transform_sha256 = _deterministic_npz(transform_path, transform_arrays)
            metadata["transform_path"] = transform_path.name
            metadata["transform_file_bytes"] = transform_file_bytes
            metadata["transform_sha256"] = transform_sha256
            metadata["transform_arrays"] = sorted(transform_arrays)
        array_path = output_dir / f"{name}.npy"
        np.save(array_path, method_vectors, allow_pickle=False)
        result = compare_method(name, method_vectors, rows, baseline, output_dir, block_size=block_size)
        result["transform_seconds"] = transform_seconds
        result.update(metadata)
        result["array_bytes"] = int(method_vectors.nbytes)
        result["array_file_bytes"] = array_path.stat().st_size
        result["process_peak_rss_bytes"] = _peak_rss_bytes()
        result["process_peak_rss_bytes_before_method"] = peak_before_method
        result["incremental_process_peak_rss_bytes"] = max(0, result["process_peak_rss_bytes"] - peak_before_method)
        result["output_tree_bytes"] = (sum(p.stat().st_size for p in (output_dir / name).rglob("*") if p.is_file())
                                        + array_path.stat().st_size
                                        + (transform_path.stat().st_size if transform_path else 0))
        result["disk_bytes_total"] = result["output_tree_bytes"]
        result["vector_sha256"] = sha256_file(array_path)
        method_results[name] = result

    receipt = {
        "schema": "statement-compression-receipt.v1",
        "source_run": str(run_dir),
        "source_manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "baseline": {"method": "minilm_chunk", "dimensions": 384,
                     "corpus_sha256": sha256_file(corpus_path),
                     "vector_sha256": sha256_file(vectors_path),
                     "top50_sha256": sha256_file(top50_path),
                     "top50_path": str(top50_path.relative_to(run_dir))},
        "rows": n, "platform_counts": dict(sorted(collections.Counter(r["platform"] for r in corpus).items())),
        "pca_fit": pca_fit, "svd_fit": svd_fit,
        "methods": method_results,
        "ks": list(KS), "retrieval": "exact cross-platform cosine; stable key tie order; same Rows and exclusions as research/full_catalog/retrieve.py",
        "comparison": "per-query TopK overlap only; neighborhood preservation is not relevance or recommendation quality",
        "query_latency_sample_seed": RANDOM_PROJECTION_SEED,
        "no_embedding_or_training": True, "no_umap": True, "no_model_promotion": True,
        "environment": {"numpy": np.__version__, "python": host_platform.python_version(),
                        "machine": host_platform.machine()},
    }
    receipt_path = output_dir / "receipt.json"
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="completed statement-run-v1 directory")
    parser.add_argument("--output", type=Path, required=True, help="new private output directory")
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args(argv)
    receipt = run(args.run, args.output, block_size=args.block_size)
    print(json.dumps({"rows": receipt["rows"], "methods": {k: v["status"] for k, v in receipt["methods"].items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))
