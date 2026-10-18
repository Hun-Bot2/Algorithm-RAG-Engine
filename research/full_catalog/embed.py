"""Phase 3: encode A1/A2 inputs with the pinned local MiniLM (offline, CPU, 1 thread).

The model directory is hash-verified by src.learning.embedding_backend before loading.
HF/Transformers offline flags and Python socket guards are active during load and
inference. Vectors stay private; the receipt has hashes, token stats and resources.

    .venv/bin/python research/full_catalog/embed.py --variant A1 \
        --catalog private-learning/full-catalog/catalog-v1/catalog.v1.jsonl \
        --output private-learning/full-catalog/embeddings-v1/A1 \
        --receipt research/full_catalog/receipts/embeddings-A1.v1.json
"""

import argparse
import hashlib
import json
import os
import platform as host_platform
import resource
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

import inputs

ROOT = Path(__file__).resolve().parents[2]
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_DIR = ROOT / "private-learning/models" / f"minilm-{REVISION}"
MANIFEST = ROOT / "research/minilm_pilot/model.manifest.json"
EMBED_VARIANTS = ("A1", "A2")
BATCH_SIZE = 64
THREADS = 1


def token_stats(lengths, max_tokens):
    """Distribution of tokenized lengths (special tokens included) and truncation rate."""
    values = np.asarray(lengths, dtype=np.int64)
    if values.size == 0:
        raise ValueError("no_inputs")
    truncated = int((values > max_tokens).sum())
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
        "max_tokens": max_tokens,
        "truncated": truncated,
        "truncation_rate": truncated / values.size,
    }


def encode_all(encode_batch, texts, batch_size=BATCH_SIZE):
    """Encode in fixed-order batches; validate shape, finiteness and L2 norm."""
    parts = [np.asarray(encode_batch(texts[i:i + batch_size]), dtype=np.float32)
             for i in range(0, len(texts), batch_size)]
    vectors = np.ascontiguousarray(np.vstack(parts), dtype=np.float32)
    if vectors.shape[0] != len(texts):
        raise ValueError("vector_count_mismatch")
    if not np.isfinite(vectors).all():
        raise ValueError("non_finite_vectors")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
        raise ValueError("vectors_not_normalized")
    return vectors


def array_digest(vectors):
    return hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest()


def _peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(data)


def _run(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT))
    from importlib.metadata import version
    from src.learning.embedding_backend import LocalEmbeddingBackend
    import torch
    from threadpoolctl import threadpool_limits

    catalog_rows = inputs.load_catalog(args.catalog)
    rows, texts = inputs.select(catalog_rows, args.variant)
    keys = [r["problem_key"] for r in rows]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    torch.set_num_threads(THREADS)
    with threadpool_limits(limits=THREADS):
        started = time.perf_counter()
        backend = LocalEmbeddingBackend(MODEL_DIR, manifest)
        load_seconds = time.perf_counter() - started
        encoder = backend.encoder
        tokenizer = encoder.tokenizer
        lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]]
        started = time.perf_counter()
        vectors = encode_all(lambda batch: encoder.encode(
            batch, batch_size=BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False), texts)
        encode_seconds = time.perf_counter() - started

    args.output.mkdir(parents=True, exist_ok=False)
    buffer = _npy_bytes(vectors)
    keys_bytes = (json.dumps(keys, ensure_ascii=False) + "\n").encode()
    _write_new(args.output / "vectors.npy", buffer)
    _write_new(args.output / "keys.json", keys_bytes)
    return {
        "schema": "embedding-receipt.v1",
        "variant": args.variant,
        "rows": len(keys),
        "rows_by_platform": dict(sorted(Counter(r["platform"] for r in rows).items())),
        "model": {k: manifest[k] for k in ("model_id", "revision", "dimensions", "max_tokens", "namespace")},
        "model_files_sha256": manifest["files_sha256"],
        "settings": {"batch_size": BATCH_SIZE, "threads": THREADS, "device": "cpu",
                     "normalize_embeddings": True, "dtype": "float32", "order": "problem_key ascending"},
        "inputs": {"catalog": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
                   "variant_input": inputs.input_digest(rows, texts)},
        "outputs": {"vectors.npy": hashlib.sha256(buffer).hexdigest(),
                    "vectors_array": array_digest(vectors),
                    "keys.json": hashlib.sha256(keys_bytes).hexdigest()},
        "tokens": token_stats(lengths, manifest["max_tokens"]),
        "performance": {"model_load_seconds": round(load_seconds, 3),
                        "encode_seconds": round(encode_seconds, 3),
                        "texts_per_second": round(len(texts) / encode_seconds, 1),
                        "peak_rss_bytes": _peak_rss_bytes(),
                        "peak_rss_scope": "whole process (catalog load, model, encoding)"},
        "environment": {name: version(name) for name in
                        ("torch", "sentence-transformers", "transformers", "numpy")}
                       | {"python": host_platform.python_version(), "machine": host_platform.machine()},
        "training": False,
        "promotion": False,
    }


def _npy_bytes(vectors):
    import io
    buffer = io.BytesIO()
    np.save(buffer, vectors, allow_pickle=False)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=EMBED_VARIANTS, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new private directory")
    parser.add_argument("--receipt", type=Path, required=True, help="new aggregate receipt path")
    args = parser.parse_args()

    guard = RuntimeError("Python network disabled during offline embedding")
    with warnings.catch_warnings(record=True) as caught, \
            patch("socket.socket.connect", side_effect=guard), \
            patch("socket.create_connection", side_effect=guard):
        warnings.simplefilter("always")
        receipt = _run(args)
    receipt["offline_boundary"] = ("HF/Transformers offline flags and Python socket connect guards; "
                                   "not an OS-level egress firewall test")
    receipt["warnings"] = [{"category": c, "message": m[:300], "count": n} for (c, m), n in
                           sorted(Counter((type(w.message).__name__, str(w.message)) for w in caught).items())]
    _write_new(args.receipt, (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode())


if __name__ == "__main__":
    main()
