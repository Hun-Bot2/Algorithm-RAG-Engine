"""Offline field-availability audit for existing cross-platform catalog files.

This reports counts, lengths, field/key presence, hashes, and structural HTML
marker counts only. It never writes statement text or sends network requests.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LC_RAW = ROOT / "data/raw/leetcode_raw_data.jsonl"
LC_PROCESSED = ROOT / "data/processed/leetcode_processed.jsonl"
EXTERNAL = ROOT / "private-learning/catalogs/2026-10-16-api-v1-reviewed-identities/problems.jsonl"
MERGED_LABELS = ROOT / "research/full_catalog/labels.v1.jsonl"
EVAL_DATA = ROOT / "Evaluation_100/data"
INDEX_METADATA = ROOT / "indexes/numpy_leetcode/metadata.json"

HTML_MARKERS = {
    "paragraph_open": re.compile(r"<p(?:\s|>)", re.I),
    "heading_open": re.compile(r"<h[1-6](?:\s|>)", re.I),
    "list_open": re.compile(r"<(?:ul|ol)(?:\s|>)", re.I),
    "pre_open": re.compile(r"<(?:pre|code)(?:\s|>)", re.I),
    "image_open": re.compile(r"<img(?:\s|>)", re.I),
    "math_marker": re.compile(r"(?:\$\$\$|<var(?:\s|>)|katex|mathjax)", re.I),
    "input_output_label": re.compile(r"(?:input|output|constraints|examples?)", re.I),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path):
    rows = []
    malformed = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
    return rows, malformed


def summary_fields(rows, names):
    return {
        field: {
            "present_nonnull": sum(field in r and r[field] is not None for r in rows),
            "nonempty": sum(bool(r.get(field)) for r in rows),
            "null_or_missing": sum(field not in r or r[field] is None for r in rows),
        }
        for field in names
    }


def leetcode():
    def scan(path, text_field):
        rows = 0
        malformed = 0
        keys = set()
        lengths = []
        counts = collections.Counter()
        tag_frequency = collections.Counter()
        field_counts = collections.Counter()
        for field in ("content", "content_clean", "constraints", "samples", "tags", "language", "content_sha256", "statement_status", "provenance", "url", "difficulty"):
            field_counts[f"present:{field}"] = 0
            field_counts[f"nonnull:{field}"] = 0
            field_counts[f"nonempty:{field}"] = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                rows += 1
                keys.update(row.keys())
                for field in ("content", "content_clean", "constraints", "samples", "tags", "language", "content_sha256", "statement_status", "provenance", "url", "difficulty"):
                    if field in row:
                        field_counts[f"present:{field}"] += 1
                    if field in row and row[field] is not None:
                        field_counts[f"nonnull:{field}"] += 1
                    if row.get(field):
                        field_counts[f"nonempty:{field}"] += 1
                if text_field:
                    value = row.get(text_field) or ""
                    lengths.append(len(value))
                    # Only count structural markers; the underlying text is discarded per row.
                    for name, pattern in HTML_MARKERS.items():
                        counts[name] += len(pattern.findall(value))
                    tag_frequency.update(row.get("tags") or [])
        fields = {}
        for field in ("content", "content_clean", "constraints", "samples", "tags", "language", "content_sha256", "statement_status", "provenance", "url", "difficulty"):
            fields[field] = {
                "present_nonnull": field_counts[f"nonnull:{field}"],
                "nonempty": field_counts[f"nonempty:{field}"],
                "null_or_missing": rows - field_counts[f"nonnull:{field}"],
            }
        return rows, malformed, sorted(keys), fields, sorted(lengths), counts, tag_frequency

    raw_n, bad_raw, raw_keys, raw_fields, lengths, marker_counts, tags = scan(LC_RAW, "content")
    processed_n, bad_processed, processed_keys, processed_fields, _, _, _ = scan(LC_PROCESSED, None)
    lengths = sorted(lengths)
    return {
        "source_files": {
            "raw": {"path": str(LC_RAW.relative_to(ROOT)), "sha256": digest(LC_RAW)},
            "processed": {"path": str(LC_PROCESSED.relative_to(ROOT)), "sha256": digest(LC_PROCESSED)},
        },
        "row_counts": {"raw": raw_n, "processed": processed_n, "malformed_raw": bad_raw, "malformed_processed": bad_processed},
        "record_keys": {"raw_union": raw_keys, "processed_union": processed_keys},
        "field_availability": {
            "raw": raw_fields,
            "processed": processed_fields,
        },
        "statement_length_chars": {
            "basis": "in-memory length count only; text is neither emitted nor used for retrieval",
            "min": lengths[0] if lengths else None,
            "median_upper": lengths[len(lengths) // 2] if lengths else None,
            "p95_nearest_rank": lengths[(95 * len(lengths) + 99) // 100 - 1] if lengths else None,
            "max": lengths[-1] if lengths else None,
            "empty": sum(n == 0 for n in lengths),
        },
        "html_marker_counts": {
            "interpretation": "heuristics over stored HTML only; not verified field extraction or statement parsing",
            "counts": dict(sorted(marker_counts.items())),
        },
        "tag_availability": {"rows_with_nonempty_tags": raw_fields["tags"]["nonempty"], "rows_empty_or_missing": raw_n - raw_fields["tags"]["nonempty"], "distinct_tags": len(tags), "frequency": dict(sorted(tags.items()))},
        "language_hash_status_provenance": {
            "language_values": "no explicit language field is present in the record schema; historical audit found English text by character heuristic",
            "per_record_statement_hash": "not stored as a field in the record schema",
            "statement_status": "not stored as an explicit field in the record schema",
            "retrieval_provenance": "timestamps and raw response hashes unavailable in record files; collection date is documented as January 2026",
        },
    }


def external():
    rows, bad = jsonl(EXTERNAL)
    result = {"source_file": {"path": str(EXTERNAL.relative_to(ROOT)), "sha256": digest(EXTERNAL)}, "rows": len(rows), "malformed": bad, "platforms": {}}
    for platform in ("codeforces", "atcoder"):
        subset = [r for r in rows if r.get("platform") == platform]
        result["platforms"][platform] = {
            "rows": len(subset),
            "record_keys": sorted(set().union(*(r.keys() for r in subset))),
            "field_availability": summary_fields(subset, ("full_statement", "statement_status", "constraints", "samples", "tags", "language", "content_sha256", "native_difficulty", "problem_key", "title", "url", "source", "retrieved_at", "source_sha256")),
            "statement_status_counts": dict(sorted(collections.Counter(str(r.get("statement_status")) for r in subset).items())),
            "tag_cardinality": {"nonempty_rows": sum(bool(r.get("tags")) for r in subset), "distinct_tags": len({t for r in subset for t in (r.get("tags") or [])})},
            "provenance_fields": {k: ("present_in_record_schema" if k in subset[0] else "absent_from_record_schema") for k in ("source", "retrieved_at", "source_sha256", "source_url")},
        }
    return result


def _language_heuristic(text):
    scripts = collections.Counter()
    for char in text:
        name = unicodedata.name(char, "")
        if "LATIN" in name:
            scripts["latin"] += 1
        elif "CYRILLIC" in name:
            scripts["cyrillic"] += 1
        elif any(token in name for token in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
            scripts["cjk_japanese_korean"] += 1
        elif char.isalpha():
            scripts["other_letter"] += 1
    active = [name for name, count in scripts.items() if count]
    if not active:
        return "no_detected_letters"
    return "mixed" if len(active) > 1 else active[0]


def legacy_evaluation():
    """Aggregate the seven legacy Evaluation_100 JSONLs without exporting text."""
    lc_projection = {"id": set(), "slug": set()}
    for line in LC_RAW.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line, object_pairs_hook=lambda pairs: {k: v for k, v in pairs if k in {"id", "titleSlug"}})
            if row.get("id") is not None:
                lc_projection["id"].add(str(row["id"]))
            if row.get("titleSlug"):
                lc_projection["slug"].add(str(row["titleSlug"]))
    legacy_lc = {"id": set(), "slug": set()}
    files = {}
    for path in sorted(EVAL_DATA.glob("*.jsonl")):
        if path.name not in {
            "baekjoon_normalized.jsonl", "baekjoon_preprocessed.jsonl", "baekjoon_raw_data.jsonl", "baekjoon_refined.jsonl",
            "leetcode_preprocessed.jsonl", "leetcode_raw_data.jsonl", "leetcode_refined.jsonl",
        }:
            continue
        rows = 0
        malformed = 0
        keys = set()
        fields = collections.defaultdict(lambda: {"present_nonnull": 0, "nonempty": 0, "null_or_missing": 0})
        tag_counts = collections.Counter()
        languages = collections.Counter()
        text_lengths = collections.defaultdict(list)
        id_values = set()
        slug_values = set()
        candidate_fields = ("content", "content_cleaned", "embedding_text", "constraints", "samples", "language", "content_sha256", "statement_status", "provenance", "tags", "id", "titleSlug")
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                rows += 1
                keys.update(row.keys())
                for field in candidate_fields:
                    value = row.get(field)
                    if field in row and value is not None:
                        fields[field]["present_nonnull"] += 1
                    if value:
                        fields[field]["nonempty"] += 1
                    if field not in row or value is None:
                        fields[field]["null_or_missing"] += 1
                tag_counts.update(row.get("tags") or [])
                if row.get("content"):
                    text_lengths["content"].append(len(row["content"]))
                    languages[_language_heuristic(row["content"])] += 1
                for field in ("content_cleaned", "embedding_text"):
                    if row.get(field):
                        text_lengths[field].append(len(row[field]))
                if path.name.startswith("leetcode_"):
                    if row.get("id") is not None:
                        id_values.add(str(row["id"]))
                    if row.get("titleSlug"):
                        slug_values.add(str(row["titleSlug"]))
        per_text_field = {}
        for field, values in sorted(text_lengths.items()):
            ordered = sorted(values)
            per_text_field[field] = {"nonempty_rows": len(values), "chars_min": ordered[0], "chars_median_upper": ordered[len(ordered) // 2], "chars_p95_nearest_rank": ordered[(95 * len(ordered) + 99) // 100 - 1], "chars_max": ordered[-1]}
        detail = {
            "sha256": digest(path),
            "size_bytes": path.stat().st_size,
            "rows": rows,
            "malformed": malformed,
            "record_keys": sorted(keys),
            "field_availability": {field: {**value, "null_or_missing": rows - value["present_nonnull"]} for field, value in sorted(fields.items())},
            "statement_text_fields": {
                "content": "stored source statement field in this legacy artifact; aggregates only",
                "content_cleaned": "derived cleaned text; not treated as source statement field",
                "embedding_text": "derived/refined retrieval input; not treated as an original full statement",
            },
            "text_lengths_chars": per_text_field,
            "language_heuristic_for_content_only": {"basis": "Unicode script counts only; not a language detector", "counts": dict(sorted(languages.items()))},
            "tag_availability": {"rows_with_nonempty_tags": fields["tags"]["nonempty"], "rows_null_or_missing_tags": fields["tags"]["null_or_missing"], "distinct_tags": len(tag_counts)},
            "explicit_constraints_field": {"present_nonnull": fields["constraints"]["present_nonnull"], "nonempty": fields["constraints"]["nonempty"], "null_or_missing": rows - fields["constraints"]["present_nonnull"]},
            "explicit_samples_field": {"present_nonnull": fields["samples"]["present_nonnull"], "nonempty": fields["samples"]["nonempty"], "null_or_missing": rows - fields["samples"]["present_nonnull"]},
        }
        if path.name == "leetcode_raw_data.jsonl":
            detail["production_leetcode_overlap"] = {
                "metadata_only_identity_comparison": True,
                "legacy_rows": rows,
                "id_overlap_rows": len(id_values & lc_projection["id"]),
                "slug_overlap_rows": len(slug_values & lc_projection["slug"]),
                "distinct_legacy_ids": len(id_values),
                "distinct_legacy_slugs": len(slug_values),
                "note": "ID and slug overlap are separate; the 100 legacy rows are not additional unique LeetCode coverage when keys overlap.",
            }
        files[path.name] = detail

    index = json.loads(INDEX_METADATA.read_text(encoding="utf-8"))
    indexed_by_slug = {str(r.get("slug")): r for r in index if r.get("slug")}
    production_by_slug = {}
    for line in LC_PROCESSED.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line, object_pairs_hook=lambda pairs: {k: v for k, v in pairs if k in {"slug", "tags"}})
            if row.get("slug"):
                production_by_slug[str(row["slug"])] = row.get("tags") or []
    compared = set(indexed_by_slug) & set(production_by_slug)
    return {
        "scope": "legacy evaluation artifacts, secondary to the three-platform catalog; do not add Baekjoon rows to cross-platform coverage",
        "files": files,
        "leetcode_index_metadata_tag_check": {
            "path": str(INDEX_METADATA.relative_to(ROOT)),
            "sha256": digest(INDEX_METADATA),
            "rows": len(index),
            "rows_with_nonempty_tags": sum(bool(r.get("tags")) for r in index),
            "distinct_tags": len({tag for r in index for tag in (r.get("tags") or [])}),
            "matched_production_slugs": len(compared),
            "tag_lists_equal_on_matched_slugs": sum(list(indexed_by_slug[s].get("tags") or []) == list(production_by_slug[s]) for s in compared),
            "tag_list_mismatch_slugs": sum(list(indexed_by_slug[s].get("tags") or []) != list(production_by_slug[s]) for s in compared),
        },
    }


def preservation_snapshot():
    """Hash protected historical inputs/runs without loading their text content."""
    roots = ("research/full_catalog", "research/minilm_pilot", "research/vector_geometry", "research/reports", "research/splits", "research/relationship-judgments.v1.json", "research/protocol.v1.json", "data", "indexes", "assets", "Evaluation_100/data", "private-learning/catalogs")
    files = set()
    for name in roots:
        path = ROOT / name
        if path.is_file():
            files.add(path)
        elif path.exists():
            files.update(p for p in path.rglob("*") if p.is_file())
    # Include the private artifacts backing existing full-catalog runs and labels.
    private = ROOT / "private-learning/full-catalog"
    files.update(p for p in private.rglob("*") if p.is_file())
    return {str(p.relative_to(ROOT)): {"sha256": digest(p), "size_bytes": p.stat().st_size} for p in sorted(files)}


def labels_summary():
    rows, malformed = jsonl(MERGED_LABELS)
    receipt_path = ROOT / "research/full_catalog/receipts/labels.v1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return {
        "path": str(MERGED_LABELS.relative_to(ROOT)),
        "sha256": digest(MERGED_LABELS),
        "rows": len(rows),
        "distinct_pair_ids": len({r.get("pair_id") for r in rows}),
        "judged_nonnull": sum(r.get("label") is not None for r in rows),
        "unjudged_null": sum(r.get("label") is None for r in rows),
        "malformed": malformed,
        "receipt_path": str(receipt_path.relative_to(ROOT)),
        "receipt_sha256": digest(receipt_path),
        "receipt_final_label_counts": receipt["final"]["labels"],
        "row_counts_match_receipt": len(rows) == receipt["final"]["pairs"] and {str(k) if k is not None else "None": v for k, v in collections.Counter(r.get("label") for r in rows).items()} == receipt["final"]["labels"],
    }


def main():
    before = preservation_snapshot()
    labels = labels_summary()
    body = {
        "schema": "statement-availability-audit.v1",
        "audit_date": "2026-10-16",
        "scope": "local offline availability and provenance audit only; no network, statement retrieval, embedding, or retrieval",
        "decision_boundary": {
            "leetcode": "3,053 stored statement-bearing records remain quarantined; this audit uses only aggregate availability, length, marker, and digest checks",
            "codeforces": "existing metadata snapshot only; no statement text collected",
            "atcoder": "existing metadata snapshot only; no statement text collected",
        },
        "leetcode": leetcode(),
        "existing_catalog_metadata": external(),
        "legacy_evaluation_secondary": legacy_evaluation(),
        "existing_labels": labels,
        "provenance_registry": {"source_registry": "assets/source_registry.v1.json", "cross_platform_registry": "assets/cross_platform_sources.v1.json", "record_level_statement_provenance": "absent for LeetCode; external metadata provenance is in the source registry and capture manifests"},
    }
    after = preservation_snapshot()
    if before != after:
        raise SystemExit("protected research/input artifact changed during audit")
    body["preservation_check"] = {"status": "unchanged_during_audit", "protected_file_count": len(before), "protected_file_manifest_sha256": hashlib.sha256(json.dumps(before, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    out = ROOT / "research/statements/availability-audit.v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    # The companion lists every protected file digest, plus separate frozen before/after roots.
    preserve = {
        "schema": "baseline-preservation.v1",
        "audit_date": "2026-10-16",
        "scope": "preserved full-catalog runs, labels, pool, receipts, scripts, earlier research runs, source data, indexes, and assets",
        "existing_full_catalog_labels": labels,
        "protected_files": before,
        "before_manifest_sha256": hashlib.sha256(json.dumps(before, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "after_manifest_sha256": hashlib.sha256(json.dumps(after, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "before_after_equal": before == after,
    }
    (out.parent / "BASELINE-PRESERVATION.v1.json").write_text(json.dumps(preserve, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
