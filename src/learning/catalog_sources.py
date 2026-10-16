"""Private, bounded metadata ingestion for Codeforces and AtCoder catalogs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SOURCES = {
    "codeforces": "https://codeforces.com/api/problemset.problems?lang=en",
    "atcoder": "https://kenkoooo.com/atcoder/resources/problems.json",
}
MAX_BYTES = 20 * 1024 * 1024
TIMEOUT_SECONDS = 30
MIN_START_INTERVAL = 2.1


@dataclass(frozen=True)
class FetchResult:
    body: bytes
    http_status: int | None
    error: str | None = None


def bounded_fetch(url: str) -> FetchResult:
    request = Request(url, headers={"User-Agent": "AlgorithmRAGEngine-catalog-metadata/1.0"})
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            status = response.status
            body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            return FetchResult(body[:MAX_BYTES], status, f"response_exceeds_{MAX_BYTES}_bytes")
        return FetchResult(body, status)
    except HTTPError as exc:
        body = exc.read(MAX_BYTES)
        return FetchResult(body, exc.code, f"http_error_{exc.code}")
    except (URLError, TimeoutError, OSError) as exc:
        return FetchResult(b"", None, f"transport_error:{type(exc).__name__}:{exc}")


def _json_bytes(body: bytes):
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_json_or_non_utf8") from exc


def parse_codeforces(body: bytes) -> tuple[list[dict], dict[str, int]]:
    payload = _json_bytes(body)
    if not isinstance(payload, dict) or payload.get("status") != "OK":
        raise ValueError("codeforces_api_status_not_OK")
    result = payload.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("problems"), list):
        raise ValueError("codeforces_missing_problem_list")
    rows, rejects, seen = [], {}, set()
    for item in result["problems"]:
        reason = None
        if not isinstance(item, dict):
            reason = "record_not_object"
        elif not isinstance(item.get("contestId"), int) or isinstance(item.get("contestId"), bool) or item["contestId"] < 1:
            reason = "missing_or_invalid_contest_id"
        elif not isinstance(item.get("index"), str) or not re.fullmatch(r"[A-Za-z0-9]+", item["index"]):
            reason = "missing_or_invalid_index"
        elif not isinstance(item.get("name"), str) or not item["name"].strip():
            reason = "missing_or_invalid_title"
        key = f"codeforces:{item['contestId']}{item['index']}" if reason is None else None
        if reason is None and key in seen:
            reason = "duplicate_identity"
        if reason:
            rejects[reason] = rejects.get(reason, 0) + 1
            continue
        seen.add(key)
        tags = item.get("tags")
        if tags is not None and (not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)):
            rejects["invalid_tags"] = rejects.get("invalid_tags", 0) + 1
            continue
        rating = item.get("rating")
        if rating is not None and (not isinstance(rating, (int, float)) or isinstance(rating, bool)
                                   or not math.isfinite(rating) or rating < 0):
            rejects["invalid_rating"] = rejects.get("invalid_rating", 0) + 1
            continue
        rows.append({"problem_key": key, "platform": "codeforces", "title": item["name"],
                     "url": f"https://codeforces.com/problemset/problem/{item['contestId']}/{item['index']}",
                     "native_difficulty": rating, "tags": tags,
                     "full_statement": None, "statement_status": "not_provided"})
    return rows, rejects


def parse_atcoder(body: bytes) -> tuple[list[dict], dict[str, int]]:
    payload = _json_bytes(body)
    if not isinstance(payload, list):
        raise ValueError("atcoder_catalog_not_list")
    rows, rejects, seen = [], {}, set()
    for item in payload:
        reason = None
        if not isinstance(item, dict):
            reason = "record_not_object"
        elif not isinstance(item.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", item["id"]):
            reason = "missing_or_invalid_task_id"
        elif not isinstance(item.get("name"), str) or not item["name"].strip():
            reason = "missing_or_invalid_title"
        elif item.get("contest_id") is not None and (not isinstance(item.get("contest_id"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", item["contest_id"])):
            reason = "invalid_contest_id"
        key = f"atcoder:{item['id']}" if reason is None else None
        if reason is None and key in seen:
            reason = "duplicate_identity"
        if reason:
            rejects[reason] = rejects.get(reason, 0) + 1
            continue
        seen.add(key)
        rows.append({"problem_key": key, "platform": "atcoder", "title": item["name"],
                     "url": f"https://atcoder.jp/contests/{item['contest_id']}/tasks/{item['id']}" if isinstance(item.get("contest_id"), str) and item["contest_id"] else None,
                     "native_difficulty": None, "tags": None,
                     "full_statement": None, "statement_status": "not_provided"})
    return rows, rejects


def _seed_keys() -> list[str]:
    root = Path(__file__).resolve().parents[2]
    seed = json.loads((root / "curriculum/seed.v1.json").read_text(encoding="utf-8"))
    problems = seed.get("problems", [])
    return sorted({p["problem_key"] for p in problems if isinstance(p, dict)
                   and isinstance(p.get("problem_key"), str)
                   and p["problem_key"].startswith(("codeforces:", "atcoder:"))})


def curriculum_coverage(keys, seen, aliases):
    """Expose native matches and reviewed aliases separately; never rewrite raw IDs."""
    mapping = {}
    for item in aliases:
        old, native = item.get("legacy_key"), item.get("native_key")
        if not isinstance(old, str) or not isinstance(native, str) or old == native or old.split(":")[0] != native.split(":")[0] or old in mapping:
            raise ValueError("invalid or duplicate identity alias")
        mapping[old] = native
    if any(value in mapping for value in mapping.values()):
        raise ValueError("identity alias chains are not permitted")
    result = {}
    for platform in ("codeforces", "atcoder"):
        requested = sorted(k for k in keys if k.startswith(platform + ":"))
        direct = sorted(k for k in requested if k in seen)
        missing = sorted(set(requested) - set(direct))
        linked = [{"legacy_key": k, "native_key": mapping[k]} for k in missing if mapping.get(k) in seen]
        result[platform] = {"seed_keys": requested, "matched": direct, "missing": missing,
                            "matched_via_reviewed_alias": linked,
                            "unresolved": [k for k in missing if mapping.get(k) not in seen]}
    return result


def _write_catalog(output: Path, source: str, fetch: FetchResult, retrieved_at: str) -> tuple[dict, list[dict]]:
    raw_path = output / f"{source}.raw"
    raw_path.write_bytes(fetch.body)
    receipt = {"source": source, "source_url": SOURCES[source], "retrieved_at": retrieved_at,
               "http_status": fetch.http_status, "byte_count": len(fetch.body),
               "sha256": hashlib.sha256(fetch.body).hexdigest(),
               "status": "failed" if fetch.error else "ok", "error": fetch.error}
    rows, rejects = [], {}
    if fetch.http_status is not None and not 200 <= fetch.http_status < 300 and not fetch.error:
        receipt.update(status="failed", error=f"http_error_{fetch.http_status}")
    if receipt["error"] is None:
        try:
            rows, rejects = (parse_codeforces if source == "codeforces" else parse_atcoder)(fetch.body)
            if not rows:
                raise ValueError("empty_catalog")
        except ValueError as exc:
            receipt.update(status="failed", error=f"schema_error:{exc}")
    receipt["rejected_records"] = rejects
    receipt["normalized_count"] = len(rows)
    return receipt, rows


def _verify_replay(directory: Path) -> dict[str, FetchResult]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {}
    for source in SOURCES:
        entry = manifest.get(source)
        if not isinstance(entry, dict):
            raise ValueError(f"replay manifest missing {source}")
        raw = (directory / f"{source}.raw").read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if len(raw) != entry.get("byte_count") or actual != entry.get("sha256"):
            raise ValueError(f"replay hash/size mismatch for {source}")
        result[source] = FetchResult(raw, entry.get("http_status"), entry.get("error"))
    return result


def collect(output: str | Path, *, fetch: bool = False,
            transport: Callable[[str], FetchResult] | None = None,
            replay: str | Path | None = None) -> dict:
    """Create a fresh immutable run directory; network access requires fetch=True."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if replay is not None and fetch:
        raise ValueError("choose fetch or replay")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    os.chmod(output, 0o700)
    timestamp = datetime.now(timezone.utc).isoformat()
    receipts, all_rows = {}, []
    replayed = _verify_replay(Path(replay)) if replay is not None else None
    replay_manifest = json.loads((Path(replay) / "manifest.json").read_text(encoding="utf-8")) if replay is not None else {}
    fetcher = transport or bounded_fetch
    last_start = None
    for source, url in SOURCES.items():
        if replayed is not None:
            result = replayed[source]
        elif fetch:
            if last_start is not None:
                wait = MIN_START_INTERVAL - (time.monotonic() - last_start)
                if wait > 0:
                    time.sleep(wait)
            last_start = time.monotonic()
            try:
                result = fetcher(url)
            except Exception as exc:  # preserve partial run as a failed source receipt
                result = FetchResult(b"", None, f"transport_error:{type(exc).__name__}:{exc}")
            if not isinstance(result, FetchResult):
                raise TypeError("transport must return FetchResult")
        else:
            result = FetchResult(b"", None, "fetch_not_enabled")
        original = replay_manifest.get(source, {}) if replay_manifest else {}
        retrieved_at = original.get("retrieved_at") if replay is not None else datetime.now(timezone.utc).isoformat()
        receipt, rows = _write_catalog(output, source, result, retrieved_at or timestamp)
        if replay is not None:
            receipt["replayed_from"] = str(Path(replay))
        receipts[source] = receipt
        all_rows.extend(rows)
    seen = {row["problem_key"] for row in all_rows}
    keys = _seed_keys()
    alias_path = Path(__file__).resolve().parents[2] / "assets/problem_identity_aliases.v1.json"
    alias_bytes = alias_path.read_bytes()
    aliases = json.loads(alias_bytes)
    if aliases.get("schema_version") != 1 or aliases.get("status") != "source_verified_identity_mapping":
        raise ValueError("unreviewed identity alias registry")
    coverage = curriculum_coverage(keys, seen, aliases["aliases"])
    normalized_bytes = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                                for row in sorted(all_rows, key=lambda row: row["problem_key"]))
    normalized_path = output / "problems.jsonl"
    normalized_path.write_bytes(normalized_bytes)
    os.chmod(normalized_path, 0o600)
    missingness = {field: sum(row.get(field) is None for row in all_rows)
                   for field in ("native_difficulty", "tags", "full_statement")}
    manifest = {"schema_version": 1, "processed_at": timestamp, **receipts, "coverage": coverage,
                "identity_aliases_sha256": hashlib.sha256(alias_bytes).hexdigest(),
                "normalized": {"byte_count": len(normalized_bytes), "sha256": hashlib.sha256(normalized_bytes).hexdigest(),
                               "record_count": len(all_rows), "missing_fields": missingness}}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for source in SOURCES:
        os.chmod(output / f"{source}.raw", 0o600)
    os.chmod(manifest_path, 0o600)
    return manifest
