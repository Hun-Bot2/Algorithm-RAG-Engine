"""Representations: how a title becomes a vector, for index titles and for ad-hoc query texts.

Every representation exposes `index_vectors` (one row per catalog node) and
`encode_query(texts, groups)` so the evaluation treats all candidates identically.
Models run locally, offline, one thread; nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MINILM_DIR = ROOT / "private-learning/models/minilm-1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_MANIFEST = ROOT / "research/minilm_pilot/model.manifest.json"
BATCH = 64


def normalize(vectors) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    if not np.isfinite(v).all() or np.any(norms == 0):
        raise ValueError("vectors_must_be_finite_and_nonzero")
    return np.ascontiguousarray(v / norms, dtype=np.float32)


def random_hash_vectors(texts: list[str], dim: int = 384, salt: str = "lang-bias-control-v1") -> np.ndarray:
    """Control encoder: a deterministic random unit vector per exact text (no semantics, no language)."""
    out = np.empty((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        seed = int.from_bytes(hashlib.sha256(f"{salt}:{text}".encode()).digest()[:8], "little")
        out[i] = np.random.default_rng(seed).normal(size=dim)
    return normalize(out)


class MiniLMEncoder:
    """Pinned English MiniLM through the repository's hash-verifying offline backend."""

    def __init__(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path.insert(0, str(ROOT))
        import torch
        from src.learning.embedding_backend import LocalEmbeddingBackend
        torch.set_num_threads(1)
        manifest = json.loads(MINILM_MANIFEST.read_text(encoding="utf-8"))
        self.encoder = LocalEmbeddingBackend(MINILM_DIR, manifest).encoder

    def encode(self, texts: list[str]) -> np.ndarray:
        parts = [self.encoder.encode(texts[i:i + BATCH], batch_size=BATCH, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False) for i in range(0, len(texts), BATCH)]
        return normalize(np.vstack(parts))


class SentenceTransformerEncoder:
    """An approved multilingual encoder, loaded offline from its hash-verified safetensors directory."""

    def __init__(self, name: str):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from sentence_transformers import SentenceTransformer
        from research.language_bias import models as MD
        torch.set_num_threads(1)
        manifest = json.loads(MD.manifest_path(name).read_text(encoding="utf-8"))
        MD.verify_directory(MD.model_dir(name), manifest)
        self.name, self.prefix = name, MD.PINNED[name]["prefix"]
        self.model = SentenceTransformer(str(MD.model_dir(name)), device="cpu", local_files_only=True,
                                         trust_remote_code=False, model_kwargs={"use_safetensors": True})

    def encode(self, texts: list[str]) -> np.ndarray:
        prefixed = [self.prefix + t for t in texts]
        parts = [self.model.encode(prefixed[i:i + BATCH], batch_size=BATCH, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False) for i in range(0, len(prefixed), BATCH)]
        return normalize(np.vstack(parts))


def cached_vectors(path: Path, texts: list[str], encode: Callable[[list[str]], np.ndarray]) -> tuple[np.ndarray, dict]:
    """Encode `texts` once; later runs reload the saved array after checking it matches the same texts."""
    import time
    digest = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["texts_sha256"] != digest:
            raise ValueError("cached vectors were built from different texts")
        return np.load(path, allow_pickle=False), meta
    started = time.perf_counter()
    vectors = encode(texts)
    seconds = time.perf_counter() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    meta = {"texts_sha256": digest, "rows": int(vectors.shape[0]), "dim": int(vectors.shape[1]),
            "encode_seconds": round(seconds, 2), "texts_per_second": round(len(texts) / seconds, 1),
            "vectors_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with open(meta_path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return vectors, meta


class CentroidRemoval:
    """Subtract a per-language mean vector (fitted on the index) and re-normalize.

    Groups with fewer than `min_group` members are left untouched: a one-node group's centroid is
    that node's own vector, and subtracting it would leave a zero vector.
    """

    def __init__(self, vectors: np.ndarray, groups: np.ndarray, min_group: int = 30):
        self.centroids = {g: vectors[groups == g].astype(np.float64).mean(axis=0)
                          for g in np.unique(groups) if (groups == g).sum() >= min_group}

    def transform(self, vectors: np.ndarray, groups) -> np.ndarray:
        out = vectors.astype(np.float64).copy()
        for i, g in enumerate(groups):
            if g in self.centroids:
                out[i] -= self.centroids[g]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)


@dataclass
class Representation:
    name: str
    description: str
    index_vectors: np.ndarray
    encode_query: Callable[[list[str], list[str]], np.ndarray]
    parameters_millions: float = 0.0
    notes: dict = field(default_factory=dict)
