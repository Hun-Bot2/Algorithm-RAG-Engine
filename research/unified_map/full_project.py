"""Full-catalog 2D projections (PCA, UMAP, t-SNE) of the unified title vectors.

One method and seed per invocation, so seed repeats can run in parallel processes.
Parameters match research/unified_map/reduction.py (UMAP cosine/15/0.1, t-SNE
cosine/perplexity 30/1,000 iterations), but PCA is the exact full-SVD solution and all
24,297 nodes are fitted (not a sample). Single-threaded for reproducibility.

    OMP_NUM_THREADS=1 .venv/bin/python -m research.unified_map.full_project \
        --vectors private-learning/unified-map/full-v1/vectors.npy \
        --method umap --seed 17 --output private-learning/unified-map/full-v1/projections
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

METHODS = ("pca", "umap", "tsne")


def project(vectors: np.ndarray, method: str, seed: int, tsne_init: str = "pca") -> tuple[np.ndarray, dict]:
    """Return float32 (n, 2) coordinates and the exact parameters used.

    scikit-learn's t-SNE with `init="pca"` is deterministic whatever the seed, so a seed
    repeat only measures stability when `tsne_init="random"`.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("vectors_must_be_finite_2d")
    if method == "pca":
        from sklearn.decomposition import PCA
        params = {"n_components": 2, "svd_solver": "full"}
        model = PCA(**params)
        coords = model.fit_transform(vectors)
        params["explained_variance_ratio"] = [round(float(x), 6) for x in model.explained_variance_ratio_]
    elif method == "umap":
        import umap
        params = {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "metric": "cosine",
                  "random_state": seed, "transform_seed": seed, "n_jobs": 1}
        coords = umap.UMAP(**params).fit_transform(vectors)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        params = {"n_components": 2, "perplexity": 30, "init": tsne_init, "learning_rate": "auto",
                  "max_iter": 1000, "method": "barnes_hut", "metric": "cosine",
                  "random_state": seed, "n_jobs": 1}
        coords = TSNE(**params).fit_transform(vectors)
    else:
        raise ValueError(f"unknown_method:{method}")
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape != (len(vectors), 2) or not np.isfinite(coords).all():
        raise ValueError("projection_not_finite")
    return coords, params


def _npy_bytes(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tsne-init", choices=("pca", "random"), default="pca")
    parser.add_argument("--output", type=Path, required=True, help="directory; created if missing")
    args = parser.parse_args()

    vectors = np.load(args.vectors, allow_pickle=False)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        coords, params = project(vectors, args.method, args.seed, args.tsne_init)
    seconds = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    label = args.method if not (args.method == "tsne" and args.tsne_init == "random") else "tsne-random"
    stem = f"{label}-seed{args.seed}"
    data = _npy_bytes(coords)
    with open(args.output / f"{stem}.npy", "xb") as handle:
        handle.write(data)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = {
        "schema": "full-projection-receipt.v1", "method": label, "seed": args.seed,
        "rows": int(len(vectors)), "parameters": params, "seconds": round(seconds, 2),
        "peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
        "input_vectors_sha256": hashlib.sha256(args.vectors.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(data).hexdigest(),
        "environment": {n: importlib.metadata.version(n) for n in ("numpy", "scikit-learn", "umap-learn")}
                       | {"python": platform.python_version(), "machine": platform.machine()},
    }
    with open(args.output / f"{stem}.receipt.json", "x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
