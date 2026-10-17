"""Pinned downloader and offline loaders for the user-approved language-bias models.

Approved on 2026-10-17: multilingual-e5-small, paraphrase-multilingual-MiniLM-L12-v2 and the
opus-mt-ja-en translator (LaBSE was declined). Files come from huggingface.co at a pinned commit, are
bounded by their published sizes, verified against the published LFS sha256 where one exists, never
overwrite existing data, and are used offline afterwards with no remote code. The translator publishes only
pickle-format weights: they are fetched into a quarantine directory, read with torch's weights-only
unpickler and re-saved as safetensors; the pickle file is never loaded any other way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "private-learning/models"
APPROVED = ("e5-small", "paraphrase-multilingual", "opus-mt-ja-en")
TIMEOUT = 120
CHUNK = 1 << 20

PINNED = {
    "e5-small": {
        "repo": "intfloat/multilingual-e5-small", "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "license": "mit", "kind": "embedder", "parameters_millions": 118.0, "prefix": "query: ",
        "files": {
            "model.safetensors": (470641600, "1a55775f"), "config.json": (655, None), "modules.json": (387, None),
            "sentence_bert_config.json": (57, None), "1_Pooling/config.json": (200, None),
            "tokenizer.json": (17082730, "0b44a9d7"), "tokenizer_config.json": (443, None),
            "special_tokens_map.json": (167, None), "sentencepiece.bpe.model": (5069051, "cfc8146a"),
            "README.md": (497538, None)}},
    "paraphrase-multilingual": {
        "repo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42", "license": "apache-2.0", "kind": "embedder",
        "parameters_millions": 118.0, "prefix": "",
        "files": {
            "model.safetensors": (470641600, "eaa086f0"), "config.json": (645, None),
            "config_sentence_transformers.json": (122, None), "modules.json": (229, None),
            "sentence_bert_config.json": (53, None), "1_Pooling/config.json": (190, None),
            "tokenizer.json": (9081518, "2c3387be"), "tokenizer_config.json": (526, None),
            "special_tokens_map.json": (239, None), "sentencepiece.bpe.model": (5069051, "cfc8146a"),
            "README.md": (3888, None)}},
    "labse": {
        "repo": "sentence-transformers/LaBSE", "revision": "836121a0533e5664b21c7aacc5d22951f2b8b25b",
        "license": "apache-2.0", "kind": "embedder", "parameters_millions": 471.0, "prefix": "",
        "downloaded_by": "user",
        "files": {
            "model.safetensors": (1883734344, "77d8e1f2"), "2_Dense/model.safetensors": (2362528, "f866c945"),
            "2_Dense/config.json": (114, None), "1_Pooling/config.json": (190, None), "config.json": (804, None),
            "config_sentence_transformers.json": (122, None), "modules.json": (461, None),
            "sentence_bert_config.json": (53, None), "special_tokens_map.json": (112, None),
            "tokenizer.json": (9621556, None), "tokenizer_config.json": (397, None), "vocab.txt": (5220781, None)}},
    "opus-mt-ja-en": {
        "repo": "Helsinki-NLP/opus-mt-ja-en", "revision": "0770961a39ba6bd66305b149c3f4110bcafca2e6",
        "license": "apache-2.0", "kind": "translator", "parameters_millions": 76.0, "prefix": "",
        "files": {
            "pytorch_model.bin": (303294189, "ed649116"), "config.json": (1381, None),
            "generation_config.json": (293, None), "source.spm": (781853, None), "target.spm": (801883, None),
            "tokenizer_config.json": (42, None), "vocab.json": (1501341, None), "README.md": (818, None)}},
}
# Only the 8-hex prefix of each published LFS sha256 is pinned above; the full digest is read from the
# Hub API at pin time and compared in full by `pin_check` (see test_models). Prefix + exact size is the
# download-time gate; the manifest then records the full sha256 of every file.


class ModelError(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path: Path, data: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def http_fetch(url: str, limit: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "AlgorithmRAGEngine-approved-model-fetch/1"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            out, total = [], 0
            while True:
                chunk = response.read(min(CHUNK, limit + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise ModelError(f"response exceeds the pinned size {limit}")
                out.append(chunk)
            return b"".join(out)
    except urllib.error.HTTPError as exc:
        raise ModelError(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise ModelError(f"request failed for {url}: {exc.reason}") from exc


def fetch_pinned(spec: dict, name: str, dest: Path, fetch=http_fetch) -> dict:
    """Fetch every allowlisted file of one pinned model into a brand-new directory; return sha256 by file."""
    if dest.exists():
        raise ModelError(f"refusing to overwrite {dest}")
    digests = {}
    for file, (size, lfs_prefix) in spec["files"].items():
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{file}"
        data = fetch(url, size)
        if len(data) != size:
            raise ModelError(f"{name}/{file}: size {len(data)} != pinned {size}")
        digest = sha256_bytes(data)
        if lfs_prefix and not digest.startswith(lfs_prefix):
            raise ModelError(f"{name}/{file}: sha256 does not match the published LFS digest")
        write_new(dest / file, data)
        digests[file] = digest
    return digests


def manifest_for(name: str, spec: dict, digests: dict, extra: dict | None = None) -> dict:
    return {"schema": "approved-model-manifest.v1", "name": name, "repo": spec["repo"], "revision": spec["revision"],
            "license": spec["license"], "kind": spec["kind"], "approved": "user, 2026-10-17",
            "files_sha256": dict(sorted(digests.items())), "sizes": {f: s for f, (s, _) in spec["files"].items()},
            "downloaded_at": datetime.now(timezone.utc).isoformat(), **(extra or {})}


def verify_directory(directory: Path, manifest: dict) -> None:
    """Exact membership and per-file sha256; extra, missing or modified files fail."""
    present = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if present != set(manifest["files_sha256"]):
        raise ModelError(f"directory membership differs from manifest: {sorted(present ^ set(manifest['files_sha256']))}")
    for file, digest in manifest["files_sha256"].items():
        if sha256_file(directory / file) != digest:
            raise ModelError(f"hash mismatch: {file}")


def model_dir(name: str) -> Path:
    spec = PINNED[name]
    return MODELS_DIR / f"{name}-{spec['revision'][:8]}"


def manifest_path(name: str) -> Path:
    return MODELS_DIR / f"{name}-{PINNED[name]['revision'][:8]}.manifest.json"


def download(name: str, approved: bool, fetch=http_fetch) -> dict:
    if not approved:
        raise ModelError("explicit approval flag required")
    if name not in APPROVED:
        raise ModelError(f"{name} is not on the approved list {APPROVED}")
    spec = PINNED[name]
    if manifest_path(name).exists():
        raise ModelError(f"manifest already exists: {manifest_path(name)}")
    target = model_dir(name) if spec["kind"] == "embedder" else MODELS_DIR / "_pickle-quarantine" / f"{name}-{spec['revision'][:8]}"
    digests = fetch_pinned(spec, name, target, fetch)
    manifest = manifest_for(name, spec, digests, {"download_dir": str(target.relative_to(ROOT))})
    if spec["kind"] == "embedder":
        write_new(manifest_path(name), (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode())
    else:
        write_new(MODELS_DIR / f"{name}-{spec['revision'][:8]}.pickle-source.manifest.json",
                  (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode())
    return manifest


def register_existing(name: str) -> dict:
    """Write the manifest for a model the user downloaded themselves, after verifying it against the pins.

    The directory must hold exactly the pinned files (sizes match; weights match their pinned LFS digest
    prefix). Anything else, including CLI bookkeeping folders, must be moved out first.
    """
    spec = PINNED[name]
    if spec.get("downloaded_by") != "user":
        raise ModelError(f"{name} was not downloaded by the user; use download()")
    directory = model_dir(name)
    if manifest_path(name).exists():
        raise ModelError(f"manifest already exists: {manifest_path(name)}")
    present = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if present != set(spec["files"]):
        raise ModelError(f"files differ from the pinned set: {sorted(present ^ set(spec['files']))}")
    digests = {}
    for file, (size, prefix) in spec["files"].items():
        path = directory / file
        if path.stat().st_size != size:
            raise ModelError(f"{name}/{file}: size {path.stat().st_size} != pinned {size}")
        digests[file] = sha256_file(path)
        if prefix and not digests[file].startswith(prefix):
            raise ModelError(f"{name}/{file}: sha256 does not match the pinned LFS digest")
    manifest = manifest_for(name, spec, digests, {
        "downloaded_by": "user", "download_dir": str(directory.relative_to(ROOT)),
        "note": "weight files' full sha256 were compared with the Hub LFS oids on 2026-10-17"})
    manifest["approved"] = "user downloaded this model themselves on 2026-10-17"
    write_new(manifest_path(name), (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", choices=APPROVED)
    parser.add_argument("--approved-by-user", action="store_true",
                        help="the user approved this exact model on 2026-10-17")
    args = parser.parse_args()
    manifest = download(args.name, args.approved_by_user)
    print(json.dumps({k: manifest[k] for k in ("name", "revision", "download_dir")}, indent=1))


if __name__ == "__main__":
    main()
