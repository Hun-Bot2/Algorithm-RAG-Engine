"""AtCoder difficulty metadata collection and empirical rank baselines.

The AtCoder Problems dataset is community-provided and its ``difficulty``
values are estimates on that dataset's own scale. They are never converted to
Codeforces or LeetCode difficulty labels here.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import urllib.request
from pathlib import Path


SOURCE_URL = "https://kenkoooo.com/atcoder/resources/problem-models.json"
SOURCE_DOC = "https://github.com/kenkoooo/AtCoderProblems/blob/master/doc/api.md"
MAX_BYTES = 64 * 1024 * 1024
TIMEOUT_SECONDS = 30
ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "private-learning/full-catalog/catalog-v1/catalog.v1.jsonl"


def fetch_bytes(url=SOURCE_URL, timeout=TIMEOUT_SECONDS, max_bytes=MAX_BYTES):
    """Fetch one bounded response. No retries, credentials, or secret reads."""
    if url != SOURCE_URL:
        raise ValueError("only_documented_problem_models_endpoint_supported")
    request = urllib.request.Request(url, headers={"User-Agent": "Algorithm-RAG-Engine-research/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.status
        if status != 200:
            raise RuntimeError(f"unexpected_http_status:{status}")
        chunks = []
        total = 0
        while True:
            chunk = response.read(min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("response_exceeds_size_limit")
            chunks.append(chunk)
        return status, b"".join(chunks)


def parse_models(raw_bytes):
    """Parse dataset key and documented difficulty field without coercion."""
    value = json.loads(raw_bytes)
    if not isinstance(value, dict):
        raise ValueError("expected_top_level_object")
    models = {}
    for task_id, record in value.items():
        if not isinstance(task_id, str) or not isinstance(record, dict):
            raise ValueError("invalid_model_record")
        difficulty = record.get("difficulty")
        if difficulty is not None and (isinstance(difficulty, bool) or not isinstance(difficulty, (int, float))):
            raise ValueError(f"invalid_difficulty:{task_id}")
        if isinstance(difficulty, float) and not math.isfinite(difficulty):
            raise ValueError(f"non_finite_difficulty:{task_id}")
        models[task_id] = difficulty
    return models


def atcoder_task_id(problem_key):
    prefix, sep, task_id = problem_key.partition(":")
    return task_id if sep and prefix == "atcoder" else None


def prepare_native_rows(catalog_rows, models):
    """Join by unchanged native AtCoder task ID and preserve raw difficulty."""
    rows = []
    for row in catalog_rows:
        if row.get("platform") != "atcoder":
            continue
        task_id = atcoder_task_id(row["problem_key"])
        rows.append({
            "problem_key": row["problem_key"],
            "task_id": task_id,
            "contest_id": row.get("contest"),
            "native_difficulty": models.get(task_id) if task_id is not None else None,
            "difficulty_source": "atcoder_problems_estimate",
        })
    return sorted(rows, key=lambda row: row["problem_key"])


def midrank_empirical_cdf(values):
    """Return value -> (lower, upper, mid) empirical CDF interval, retaining ties."""
    present = sorted(value for value in values if value is not None)
    if not present:
        return {}
    n = len(present)
    result = {}
    i = 0
    while i < n:
        j = i + 1
        while j < n and present[j] == present[i]:
            j += 1
        lower, upper = i / n, j / n
        result[present[i]] = (lower, upper, (lower + upper) / 2)
        i = j
    return result


def quantile_baseline(rows):
    """Attach empirical rank intervals; these are provisional within-source ranks."""
    cdf = midrank_empirical_cdf([row.get("native_difficulty") for row in rows])
    output = []
    for row in rows:
        rank = cdf.get(row.get("native_difficulty"))
        output.append({**row, "empirical_cdf_interval": list(rank) if rank else None,
                       "interpretation": "within_atcoder_dataset_rank_only"})
    return output


def availability_receipt(models, native_rows, raw_sha256, retrieved_at, http_status,
                         codeforces_native_rating_nulls=None):
    catalog_ids = {row["task_id"] for row in native_rows if row["task_id"] is not None}
    matched = [row for row in native_rows if row["task_id"] in models]
    observed = [models[row["task_id"]] for row in matched]
    return {
        "schema_version": 1,
        "source": "AtCoder Problems community dataset; not an AtCoder-native rating",
        "source_url": SOURCE_URL,
        "source_documentation": SOURCE_DOC,
        "retrieved_at_utc": retrieved_at,
        "http_status": http_status,
        "raw_sha256": raw_sha256,
        "dataset_records": len(models),
        "catalog_atcoder_records": len(native_rows),
        "matched_catalog_records": len(matched),
        "matched_with_difficulty": sum(value is not None for value in observed),
        "matched_with_null_difficulty": sum(value is None for value in observed),
        "catalog_records_without_dataset_match": len(native_rows) - len(matched),
        "dataset_records_without_catalog_match": len(set(models) - catalog_ids),
        "negative_difficulty_values": sum(value is not None and value < 0 for value in observed),
        "lc_paid_status": "not_persisted; no LeetCode refresh performed",
        "codeforces_native_rating_nulls": codeforces_native_rating_nulls,
        "cross_platform_equivalence": "not_established",
        "raw_republication": "not included",
    }


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect(output_dir, receipt_path, catalog_path=CATALOG):
    """Perform the explicitly opted-in one-request metadata collection."""
    output_dir = Path(output_dir)
    receipt_path = Path(receipt_path)
    if output_dir.exists() or receipt_path.exists():
        raise FileExistsError("output_or_receipt_exists")
    catalog_rows = load_jsonl(catalog_path)
    status, raw = fetch_bytes()
    models = parse_models(raw)
    retrieved = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    digest = hashlib.sha256(raw).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "problem-models.json"
    raw_path.write_bytes(raw)
    (output_dir / "SHA256SUMS").write_text(f"{digest}  problem-models.json\n", encoding="ascii")
    receipt = availability_receipt(
        models, prepare_native_rows(catalog_rows, models), digest, retrieved, status,
        sum(row.get("native_difficulty") is None for row in catalog_rows
            if row.get("platform") == "codeforces"),
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-output", required=True, help="new, private directory for raw response and hash")
    parser.add_argument("--receipt", required=True, help="public summary receipt path")
    parser.add_argument("--catalog", default=str(CATALOG), help="existing catalog JSONL snapshot")
    args = parser.parse_args()
    receipt = collect(args.private_output, args.receipt, args.catalog)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
