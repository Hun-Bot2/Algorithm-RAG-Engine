"""Phase 1 full-catalog eligibility audit (standard library only, no network).

Reads the preserved LeetCode raw/processed JSONL and the reviewed-identity
Codeforces/AtCoder snapshot, and writes a deterministic JSON receipt. It never
modifies inputs, fills missing statements or merges records.

    python3 research/full_catalog/audit_catalog.py --output research/full_catalog/audit.json
"""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LC_RAW = ROOT / "data/raw/leetcode_raw_data.jsonl"
LC_PROCESSED = ROOT / "data/processed/leetcode_processed.jsonl"
EXTERNAL = ROOT / "private-learning/catalogs/2026-10-16-api-v1-reviewed-identities/problems.jsonl"
COLLECTOR = ROOT / "scripts/collection/leetcode_data_collection.py"

NON_LATIN = re.compile(r"[Ѐ-ӿ぀-ヿ一-鿿가-힯]")
NON_ALGORITHMIC_LC_TAGS = ("Database", "Shell", "Concurrency")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    rows, malformed = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1
    return rows, malformed


def normalized_text_hash(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def duplicate_groups(rows, key_field, value_fn):
    groups = collections.defaultdict(list)
    for row in rows:
        value = value_fn(row)
        if value:
            groups[value].append(row[key_field])
    return sorted(sorted(keys) for keys in groups.values() if len(keys) > 1)


def audit_leetcode():
    raw, raw_bad = read_jsonl(LC_RAW)
    processed, processed_bad = read_jsonl(LC_PROCESSED)
    ids = sorted(int(r["id"]) for r in raw)
    id_set = set(ids)
    missing_ids = [i for i in range(1, ids[-1] + 1) if i not in id_set]
    lengths = sorted(len(r["content"]) for r in raw)
    collector_source = COLLECTOR.read_text(encoding="utf-8")
    title_dups = duplicate_groups(processed, "slug", lambda r: r["title"].strip().lower())
    statement_dups = duplicate_groups(
        processed, "slug", lambda r: normalized_text_hash(r["content_clean"]) if r["content_clean"] else None
    )
    return {
        "inputs": {
            "raw": {"path": str(LC_RAW.relative_to(ROOT)), "sha256": sha256(LC_RAW)},
            "processed": {"path": str(LC_PROCESSED.relative_to(ROOT)), "sha256": sha256(LC_PROCESSED)},
        },
        "records": {"raw": len(raw), "processed": len(processed)},
        "malformed_lines": {"raw": raw_bad, "processed": processed_bad},
        "raw_processed_slug_sets_equal": {r["titleSlug"] for r in raw} == {r["slug"] for r in processed},
        "unique_ids": len(id_set),
        "unique_slugs": len({r["titleSlug"] for r in raw}),
        "id_range": [ids[0], ids[-1]],
        "ids_absent_below_max": len(missing_ids),
        "paid_only": {
            "collector_filters_isPaidOnly": "if prob['isPaidOnly']:" in collector_source,
            "persisted_field_present": any("isPaidOnly" in r for r in raw),
            "paid_status_of_absent_ids": "unknown: no persisted problem list or skip log",
            "records_with_premium_gate_text": sum(
                1 for r in raw if re.search(r"subscribe to unlock|premium (subscription|question)", r["content"], re.I)
            ),
        },
        "difficulty": dict(sorted(collections.Counter(r["difficulty"] for r in raw).items())),
        "statement": {
            "non_empty": sum(1 for r in raw if r["content"].strip()),
            "html_chars_min_median_p95_max": [
                lengths[0], lengths[len(lengths) // 2], lengths[int(0.95 * len(lengths))], lengths[-1]
            ],
            "contains_img": sum(1 for r in raw if "<img" in r["content"]),
            "language": "English (0 records with Cyrillic/CJK/Hangul characters)"
            if not any(NON_LATIN.search(r["content"]) for r in raw)
            else "mixed",
        },
        "tags": {
            "empty": sum(1 for r in raw if not r["tags"]),
            "distinct": len({t for r in raw for t in r["tags"]}),
            "non_algorithmic": {
                t: sum(1 for r in raw if t in r["tags"]) for t in NON_ALGORITHMIC_LC_TAGS
            },
        },
        "exact_duplicates": {"title_groups": title_dups, "normalized_statement_groups": statement_dups},
    }


def audit_external():
    rows, bad = read_jsonl(EXTERNAL)
    report = {
        "input": {"path": str(EXTERNAL.relative_to(ROOT)), "sha256": sha256(EXTERNAL)},
        "malformed_lines": bad,
        "unique_keys": len({r["problem_key"] for r in rows}),
        "platforms": {},
    }
    for platform in ("codeforces", "atcoder"):
        subset = [r for r in rows if r["platform"] == platform]
        title_groups = duplicate_groups(subset, "problem_key", lambda r: r["title"].strip().lower())
        report["platforms"][platform] = {
            "records": len(subset),
            "statement_status": dict(collections.Counter(r["statement_status"] for r in subset)),
            "full_statement_present": sum(1 for r in subset if r["full_statement"]),
            "title_empty": sum(1 for r in subset if not (r["title"] or "").strip()),
            "title_non_latin": sum(1 for r in subset if NON_LATIN.search(r["title"] or "")),
            "tags_null": sum(1 for r in subset if r["tags"] is None),
            "tags_empty_list": sum(1 for r in subset if r["tags"] == []),
            "native_difficulty_null": sum(1 for r in subset if r["native_difficulty"] is None),
            "url_missing": sum(1 for r in subset if not r["url"]),
            "title_duplicate_groups": len(title_groups),
            "rows_in_title_duplicate_groups": sum(len(g) for g in title_groups),
            "paid_status": "not applicable (public problemset metadata)",
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": "full-catalog-audit.v1", "leetcode": audit_leetcode(), "external": audit_external()}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
