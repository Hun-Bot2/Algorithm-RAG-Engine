#!/usr/bin/env python3
"""Download one pinned MiniLM snapshot using bounded, stdlib-only HTTP."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BASE = "https://huggingface.co/" + MODEL_ID
ALLOWED = (
    "model.safetensors",
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "README.md",
)
MAX_FILE = 120 * 1024 * 1024
MAX_TOTAL = 200 * 1024 * 1024
TIMEOUT = 60
LICENSE_REFERENCE = f"{BASE}/blob/{REVISION}/README.md"


class DownloadError(Exception):
    pass


def fetch(url, limit):
    request = urllib.request.Request(url, headers={"User-Agent": "Algorithm-RAG-Engine-local-model-fetch/1"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            chunks = []
            total = 0
            while True:
                chunk = response.read(min(1024 * 1024, limit + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise DownloadError(f"response exceeds {limit} byte limit")
                chunks.append(chunk)
            return b"".join(chunks)
    except urllib.error.HTTPError as exc:
        raise DownloadError(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise DownloadError(f"request failed for {url}: {exc.reason}") from exc


def write_new(path, payload, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, mode)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh model directory")
    parser.add_argument("--metadata", type=Path, required=True, help="path for raw Hugging Face revision metadata, outside model directory")
    parser.add_argument("--manifest", type=Path, required=True, help="path for LocalEmbeddingBackend manifest")
    parser.add_argument("--receipt", type=Path, required=True, help="path for structured download receipt")
    args = parser.parse_args()

    receipt = {"schema_version": 1, "model_id": MODEL_ID, "revision": REVISION,
               "status": "failed", "files": [], "error": None,
               "started_at": datetime.now(timezone.utc).isoformat()}
    output = args.output
    try:
        if output.exists():
            raise DownloadError(f"output directory already exists: {output}")
        resolved_output = output.resolve()
        auxiliary = (args.metadata.resolve(), args.manifest.resolve(), args.receipt.resolve())
        if len(set(auxiliary)) != len(auxiliary):
            raise DownloadError("metadata, manifest, and receipt paths must be distinct")
        if any(path == resolved_output or resolved_output in path.parents for path in auxiliary):
            raise DownloadError("metadata, manifest, and receipt must be outside the model directory")
        if any(path.exists() for path in (args.metadata, args.manifest, args.receipt)):
            raise DownloadError("metadata, manifest, and receipt paths must be fresh")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(mode=0o700)

        metadata_url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
        raw_metadata = fetch(metadata_url, 8 * 1024 * 1024)
        metadata = json.loads(raw_metadata)
        if metadata.get("id") != MODEL_ID or metadata.get("sha") != REVISION:
            raise DownloadError("Hugging Face metadata model id or revision did not match the pin")
        siblings = {item.get("rfilename"): item for item in metadata.get("siblings", []) if isinstance(item, dict)}
        absent = [name for name in ALLOWED if name not in siblings]
        if absent:
            raise DownloadError("pinned snapshot is missing allowlisted files: " + ", ".join(absent))
        model_entry = siblings["model.safetensors"]
        expected_lfs = model_entry.get("lfs")
        expected_hash = expected_lfs.get("sha256") if isinstance(expected_lfs, dict) else None
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise DownloadError("upstream model.safetensors LFS SHA-256 is unavailable")
        receipt["upstream_model_safetensors_sha256"] = expected_hash
        receipt["upstream_model_safetensors_size"] = expected_lfs.get("size")
        write_new(args.metadata, raw_metadata)
        receipt["metadata_path"] = str(args.metadata)
        receipt["metadata_sha256"] = hashlib.sha256(raw_metadata).hexdigest()

        total = 0
        hashes = {}
        for name in ALLOWED:
            entry_receipt = {"name": name, "status": "pending"}
            receipt["files"].append(entry_receipt)
            item = siblings[name]
            expected_size = item.get("size")
            lfs = item.get("lfs")
            if isinstance(lfs, dict) and isinstance(lfs.get("size"), int):
                expected_size = lfs["size"]
            if isinstance(expected_size, int) and (expected_size < 0 or expected_size > MAX_FILE):
                raise DownloadError(f"upstream size exceeds per-file limit for {name}")
            quoted = "/".join(urllib.parse.quote(part, safe="") for part in name.split("/"))
            url = f"{BASE}/resolve/{REVISION}/{quoted}"
            payload = fetch(url, MAX_FILE)
            if isinstance(expected_size, int) and len(payload) != expected_size:
                raise DownloadError(f"size mismatch for {name}: got {len(payload)}, expected {expected_size}")
            digest = hashlib.sha256(payload).hexdigest()
            if name == "model.safetensors" and digest != expected_hash:
                raise DownloadError("model.safetensors SHA-256 differs from upstream LFS SHA-256")
            if total + len(payload) > MAX_TOTAL:
                raise DownloadError("download exceeds total byte limit")
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            write_new(destination, payload)
            total += len(payload)
            hashes[name] = digest
            entry_receipt.update({"size": len(payload), "sha256": digest, "status": "downloaded"})

        manifest = {"schema_version": 1, "namespace": "minilm-english-summary-v1",
                    "model_id": MODEL_ID, "revision": REVISION,
                    "license_reference": LICENSE_REFERENCE, "dimensions": 384,
                    "max_tokens": 256, "files_sha256": hashes}
        write_new(args.manifest, json_bytes(manifest))
        receipt["status"] = "complete"
        receipt["total_bytes"] = total
        return 0
    except (DownloadError, OSError, ValueError, json.JSONDecodeError) as exc:
        receipt["error"] = str(exc)
        print(f"download_minilm: {exc}", file=sys.stderr)
        return 1
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            write_new(args.receipt, json_bytes(receipt))
        except FileExistsError:
            print(f"download_minilm: receipt path already exists: {args.receipt}", file=sys.stderr)
        except OSError as exc:
            print(f"download_minilm: could not write receipt: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
