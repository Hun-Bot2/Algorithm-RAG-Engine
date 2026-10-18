"""Full-catalog title vectors: reuse the A1 vectors, encode only the missing nodes.

Same pinned MiniLM, settings (CPU, 1 thread, batch 64, L2-normalized float32), offline
guards and title text (`title.strip()`) as research/full_catalog/embed.py. Node order is
problem_key ascending. Every node gets an embedding-reliability tier from the share of
its tokens that the English tokenizer cannot represent ([UNK]).

    .venv/bin/python -m research.unified_map.full_embed \
        --output private-learning/unified-map/full-v1 \
        --receipt research/unified_map/full-receipts/embeddings.v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
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

ROOT = Path(__file__).resolve().parents[2]
NODES = ROOT / "private-learning/unified-map/map-v2/nodes.jsonl"
A1_DIR = ROOT / "private-learning/full-catalog/embeddings-v1/A1"
A1_RECEIPT = ROOT / "research/full_catalog/receipts/embeddings-A1.v1.json"
MODEL_DIR = ROOT / "private-learning/models/minilm-1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MANIFEST = ROOT / "research/minilm_pilot/model.manifest.json"
BATCH_SIZE = 64
THREADS = 1
SAMPLE_SEED = 17
REUSE_SAMPLE = 500
DEGRADED_UNK = 0.10
UNRELIABLE_UNK = 0.50
TIERS = ("reliable", "degraded", "unreliable")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def reliability_tier(unk_share: float, non_latin: bool = False) -> str:
    """Tier from a title's unknown-token share (0..1) and its non-Latin script flag.

    The English MiniLM tokenizer splits some non-Latin text (for example Cyrillic) into
    known pieces, so the unknown-share alone would overstate reliability. A non-Latin
    title is therefore never better than `degraded`.
    """
    if unk_share >= UNRELIABLE_UNK:
        return "unreliable"
    if unk_share >= DEGRADED_UNK or non_latin:
        return "degraded"
    return "reliable"


def unknown_share(token_ids: list[int], unk_id: int) -> float:
    """Fraction of content tokens equal to the tokenizer's unknown id; empty titles are fully unknown."""
    if not token_ids:
        return 1.0
    return sum(1 for t in token_ids if t == unk_id) / len(token_ids)


def assemble_vectors(keys, reused: dict, encoded: dict) -> tuple[np.ndarray, list[str]]:
    """Order reused/new vectors by key; every key must come from exactly one source."""
    overlap = set(reused) & set(encoded)
    if overlap:
        raise ValueError(f"vector_source_overlap:{len(overlap)}")
    rows, sources = [], []
    for key in keys:
        if key in reused:
            rows.append(reused[key])
            sources.append("a1_reused")
        elif key in encoded:
            rows.append(encoded[key])
            sources.append("new_title_encode")
        else:
            raise ValueError(f"missing_vector:{key}")
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float32), sources


def encode_in_batches(encode_batch, texts, batch_size=BATCH_SIZE) -> np.ndarray:
    parts = [np.asarray(encode_batch(texts[i:i + batch_size]), dtype=np.float32)
             for i in range(0, len(texts), batch_size)]
    vectors = np.ascontiguousarray(np.vstack(parts), dtype=np.float32)
    if vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("invalid_encoder_output")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
        raise ValueError("vectors_not_normalized")
    return vectors


def coverage_table(nodes, reused_keys) -> list[dict]:
    cells = Counter()
    for node in nodes:
        reused = node["problem_key"] in reused_keys
        cells[(node["platform"], node["eligibility"], bool(node["encoder_flags"]), reused)] += 1
    return [{"platform": p, "eligibility": e, "non_latin_title": nl, "had_a1_vector": r, "rows": n}
            for (p, e, nl, r), n in sorted(cells.items())]


def _npy_bytes(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(data)


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _run(args) -> dict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from importlib.metadata import version

    import torch
    from threadpoolctl import threadpool_limits

    from src.learning.embedding_backend import LocalEmbeddingBackend

    nodes = sorted((json.loads(line) for line in NODES.read_text(encoding="utf-8").splitlines() if line.strip()),
                   key=lambda n: n["problem_key"])
    keys = [n["problem_key"] for n in nodes]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate node keys")

    a1_receipt = json.loads(A1_RECEIPT.read_text(encoding="utf-8"))
    for name, expected in (("vectors.npy", a1_receipt["outputs"]["vectors.npy"]),
                           ("keys.json", a1_receipt["outputs"]["keys.json"])):
        if sha256_file(A1_DIR / name) != expected:
            raise SystemExit(f"A1 artifact hash mismatch: {name}")
    a1_keys = json.loads((A1_DIR / "keys.json").read_text(encoding="utf-8"))
    a1_vectors = np.load(A1_DIR / "vectors.npy", allow_pickle=False)
    reused = {k: a1_vectors[i] for i, k in enumerate(a1_keys)}
    missing = [n for n in nodes if n["problem_key"] not in reused]

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    torch.set_num_threads(THREADS)
    with threadpool_limits(limits=THREADS):
        backend = LocalEmbeddingBackend(MODEL_DIR, manifest)
        encoder = backend.encoder
        tokenizer = encoder.tokenizer

        def encode(batch):
            return encoder.encode(batch, batch_size=BATCH_SIZE, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)

        texts = {n["problem_key"]: n["title"].strip() for n in nodes}
        started = time.perf_counter()
        new_vectors = encode_in_batches(encode, [texts[n["problem_key"]] for n in missing])
        encode_seconds = time.perf_counter() - started
        encoded = {n["problem_key"]: new_vectors[i] for i, n in enumerate(missing)}

        rng = np.random.default_rng(SAMPLE_SEED)
        sample_idx = sorted(rng.choice(len(a1_keys), size=REUSE_SAMPLE, replace=False).tolist())
        sample_keys = [a1_keys[i] for i in sample_idx]
        redo = encode_in_batches(encode, [texts[k] for k in sample_keys])
        deviation = np.abs(redo - np.vstack([reused[k] for k in sample_keys]))
        cosines = np.sum(redo * np.vstack([reused[k] for k in sample_keys]), axis=1)

        ids = tokenizer([texts[k] for k in keys], add_special_tokens=False)["input_ids"]

    vectors, sources = assemble_vectors(keys, reused, encoded)
    unk = [unknown_share(i, tokenizer.unk_token_id) for i in ids]
    tiers = [reliability_tier(u, bool(n["encoder_flags"])) for u, n in zip(unk, nodes)]
    meta_lines = "".join(json.dumps({"problem_key": k, "vector_source": s, "unk_share": round(u, 4),
                                     "reliability": t, "token_count": len(i)}, sort_keys=True) + "\n"
                         for k, s, u, t, i in zip(keys, sources, unk, tiers, ids))

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    vector_bytes = _npy_bytes(vectors)
    keys_bytes = (json.dumps(keys, ensure_ascii=False) + "\n").encode()
    _write_new(out / "vectors.npy", vector_bytes)
    _write_new(out / "keys.json", keys_bytes)
    _write_new(out / "vector_meta.jsonl", meta_lines.encode())

    tier_by_group = Counter((n["platform"], t) for n, t in zip(nodes, tiers))
    return {
        "schema": "full-embedding-receipt.v1",
        "nodes": len(keys),
        "vectors": {"reused_a1": len(reused), "new_title_encode": len(encoded),
                    "dimension": int(vectors.shape[1])},
        "coverage_before": coverage_table(nodes, set(reused)),
        "reliability": {"thresholds": {"degraded_unk_share": DEGRADED_UNK, "unreliable_unk_share": UNRELIABLE_UNK,
                                       "non_latin_title_cap": "degraded"},
                        "counts": dict(sorted(Counter(tiers).items())),
                        "by_platform": {f"{p}/{t}": c for (p, t), c in sorted(tier_by_group.items())},
                        "mean_unk_share_of_non_latin_titles": round(float(np.mean(
                            [u for n, u in zip(nodes, unk) if n["encoder_flags"]])), 4)},
        "reuse_check": {"sample_size": REUSE_SAMPLE, "seed": SAMPLE_SEED,
                        "max_abs_component_deviation": float(deviation.max()),
                        "min_cosine_to_stored": float(cosines.min()),
                        "interpretation": "re-encoding sampled A1 titles in a different batch composition"},
        "inputs": {"nodes": sha256_file(NODES), "a1_vectors": sha256_file(A1_DIR / "vectors.npy"),
                   "a1_keys": sha256_file(A1_DIR / "keys.json"), "a1_receipt": sha256_file(A1_RECEIPT),
                   "model_manifest": sha256_file(MANIFEST)},
        "outputs": {"vectors.npy": sha256_bytes(vector_bytes), "keys.json": sha256_bytes(keys_bytes),
                    "vector_meta.jsonl": sha256_bytes(meta_lines.encode())},
        "model": {k: manifest[k] for k in ("model_id", "revision", "dimensions", "max_tokens")},
        "settings": {"batch_size": BATCH_SIZE, "threads": THREADS, "device": "cpu",
                     "text": "title.strip()", "order": "problem_key ascending"},
        "performance": {"encode_seconds_new_rows": round(encode_seconds, 3),
                        "peak_rss_bytes": _peak_rss_bytes()},
        "environment": {name: version(name) for name in ("torch", "sentence-transformers", "transformers", "numpy")}
                       | {"python": host_platform.python_version(), "machine": host_platform.machine()},
        "statement_fields_read": False,
        "training": False,
        "promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="new private directory")
    parser.add_argument("--receipt", type=Path, required=True, help="new aggregate receipt")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    guard = RuntimeError("Python network disabled during offline embedding")
    with warnings.catch_warnings(record=True) as caught, \
            patch("socket.socket.connect", side_effect=guard), \
            patch("socket.create_connection", side_effect=guard):
        warnings.simplefilter("always")
        receipt = _run(args)
    receipt["offline_boundary"] = "HF/Transformers offline flags and Python socket connect guards; not an OS-level egress test"
    receipt["warnings"] = [{"category": c, "message": m[:300], "count": n} for (c, m), n in
                           sorted(Counter((type(w.message).__name__, str(w.message)) for w in caught).items())]
    _write_new(args.receipt, (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode())


if __name__ == "__main__":
    main()
