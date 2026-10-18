"""Phase 2: build the non-destructive catalog.v1 with duplicate candidates and exclusions.

Standard library only, no network. Reads only metadata fields; LeetCode statement
fields are dropped while parsing (decision D1). Inputs must match the hashes pinned in
audit.json. No record is removed: exclusions and duplicate candidates are annotations.

    python3 research/full_catalog/catalog.py \
        --private-output private-learning/full-catalog/catalog-v1 \
        --receipt research/full_catalog/receipts/catalog.v1.json
"""

import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LC_PROCESSED = ROOT / "data/processed/leetcode_processed.jsonl"
EXTERNAL = ROOT / "private-learning/catalogs/2026-10-16-api-v1-reviewed-identities/problems.jsonl"
ALIASES = ROOT / "assets/problem_identity_aliases.v1.json"

FORBIDDEN_LC_FIELDS = frozenset({"content", "content_clean", "embedding_text"})
NON_LATIN = re.compile(r"[Ѐ-ӿ぀-ヿ一-鿿가-힯]")
CF_KEY = re.compile(r"codeforces:(\d+)([A-Z]+)(\d*)")
AT_URL = re.compile(r"https://atcoder\.jp/contests/([^/]+)/tasks/")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_title(title):
    return " ".join(unicodedata.normalize("NFKC", title).casefold().split())


def _drop_statement_fields(pairs):
    return {k: v for k, v in pairs if k not in FORBIDDEN_LC_FIELDS}


def load_leetcode(lines):
    """Parse LeetCode processed JSONL keeping only identity metadata."""
    rows = []
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line, object_pairs_hook=_drop_statement_fields)
        rows.append({
            "problem_key": f"leetcode:{item['slug']}",
            "platform": "leetcode",
            "title": item["title"],
            "tags": list(item["tags"]),
            "native_difficulty": item["difficulty"],
            "url": item["url"],
            "contest": None,
        })
    return rows


def load_external(lines):
    rows = []
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        if item["platform"] == "codeforces":
            match = CF_KEY.fullmatch(item["problem_key"])
            contest = int(match.group(1)) if match else None
        else:
            match = AT_URL.match(item["url"] or "")
            contest = match.group(1) if match else None
        rows.append({
            "problem_key": item["problem_key"],
            "platform": item["platform"],
            "title": item["title"],
            "tags": item["tags"],
            "native_difficulty": item["native_difficulty"],
            "url": item["url"],
            "contest": contest,
        })
    return rows


def exclusion_reasons(row, rules):
    reasons = []
    title = normalize_title(row["title"])
    contest = str(row["contest"]).casefold() if row["contest"] is not None else None
    for rule in rules:
        if rule["platform"] != row["platform"]:
            continue
        hit = False
        if "tags_any" in rule and row["tags"]:
            hit |= any(tag in row["tags"] for tag in rule["tags_any"])
        if "contest_prefixes" in rule and contest:
            hit |= any(contest.startswith(p.casefold()) for p in rule["contest_prefixes"])
        if "contest_ids" in rule and contest:
            hit |= contest in {c.casefold() for c in rule["contest_ids"]}
        if "titles" in rule:
            hit |= title in {normalize_title(t) for t in rule["titles"]}
        if hit:
            reasons.append(rule["code"])
    return sorted(reasons)


def _cf_parts(key):
    match = CF_KEY.fullmatch(key)
    return (int(match.group(1)), match.group(2), match.group(3)) if match else None


def classify_group(platform, keys):
    """Describe structural evidence for a same-title group; never a merge decision."""
    if platform != "codeforces":
        return "same_title", {}
    parts = [_cf_parts(k) for k in keys]
    if any(p is None for p in parts):
        return "same_title", {}
    contests = sorted({p[0] for p in parts})
    if len(contests) == 1 and all(p[2] for p in parts) and len({p[1] for p in parts}) == 1:
        return "easy_hard_versions", {"contest": contests[0]}
    adjacent = [[a, b] for a, b in zip(contests, contests[1:]) if b - a == 1]
    if adjacent:
        return "possible_division_mirror", {"adjacent_contest_pairs": adjacent}
    return "same_title", {}


def duplicate_candidates(rows):
    by_title = collections.defaultdict(list)
    for row in rows:
        by_title[(row["platform"], normalize_title(row["title"]))].append(row["problem_key"])
    groups = []
    for (platform, title), keys in by_title.items():
        if len(keys) < 2:
            continue
        keys = sorted(keys)
        kind, evidence = classify_group(platform, keys)
        group_id = f"dup-{platform}-{hashlib.sha256(title.encode()).hexdigest()[:12]}"
        groups.append({"group_id": group_id, "platform": platform, "kind": kind,
                       "status": "candidate_unverified", "keys": keys, "evidence": evidence})
    return sorted(groups, key=lambda g: g["group_id"])


def cross_platform_title_matches(rows):
    by_title = collections.defaultdict(list)
    for row in rows:
        by_title[normalize_title(row["title"])].append(row)
    matches = []
    for members in by_title.values():
        if len({m["platform"] for m in members}) > 1:
            matches.append(sorted(m["problem_key"] for m in members))
    return sorted(matches)


def exact_duplicate_representatives(keys, aliases):
    """Map key -> representative using only reviewed alias mappings present in the catalog."""
    present = set(keys)
    representative = {k: k for k in keys}
    for alias in aliases:
        legacy, native = alias["legacy_key"], alias["native_key"]
        if legacy in present and native in present:
            representative[legacy] = native
    return representative


def build_catalog(lc_rows, external_rows, rules, aliases):
    rows = sorted(lc_rows + external_rows, key=lambda r: r["problem_key"])
    keys = [r["problem_key"] for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("repeated_platform_key")
    groups = duplicate_candidates(rows)
    group_of = {k: g["group_id"] for g in groups for k in g["keys"]}
    representative = exact_duplicate_representatives(keys, aliases)
    catalog = []
    for row in rows:
        reasons = exclusion_reasons(row, rules)
        flags = ["non_latin_title"] if NON_LATIN.search(row["title"]) else []
        catalog.append({
            **row,
            "title_normalized": normalize_title(row["title"]),
            "eligibility": "eligible" if not reasons else "excluded",
            "exclusion_reasons": reasons,
            "encoder_flags": flags,
            "representative_key": representative[row["problem_key"]],
            "duplicate_candidate_group": group_of.get(row["problem_key"]),
        })
    return catalog, groups, cross_platform_title_matches(rows)


def validate_crosswalk(crosswalk, cf_vocab, lc_vocab):
    """Every tag referenced exists in data, and every observed tag is mapped or listed."""
    errors = []
    cf_listed = {e["cf"] for e in crosswalk["cf_to_lc"]} | set(crosswalk.get("cf_ignored", {}))
    lc_mapped = {m["tag"] for e in crosswalk["cf_to_lc"] for m in e["lc"]}
    lc_listed = lc_mapped | set(crosswalk.get("lc_unmapped", []))
    errors += [f"cf_tag_not_in_data:{t}" for t in sorted(cf_listed - cf_vocab)]
    errors += [f"cf_tag_unlisted:{t}" for t in sorted(cf_vocab - cf_listed)]
    errors += [f"lc_tag_not_in_data:{t}" for t in sorted(lc_listed - lc_vocab)]
    errors += [f"lc_tag_unlisted:{t}" for t in sorted(lc_vocab - lc_listed)]
    errors += [f"lc_tag_both_mapped_and_unmapped:{t}"
               for t in sorted(lc_mapped & set(crosswalk.get("lc_unmapped", [])))]
    return errors


def summarize(catalog, groups, cross_matches):
    platforms = sorted({r["platform"] for r in catalog})
    summary = {}
    for p in platforms:
        rows = [r for r in catalog if r["platform"] == p]
        reasons = collections.Counter(code for r in rows for code in r["exclusion_reasons"])
        eligible = [r for r in rows if r["eligibility"] == "eligible"]
        summary[p] = {
            "records": len(rows),
            "eligible": len(eligible),
            "excluded": len(rows) - len(eligible),
            "exclusion_reason_counts": dict(sorted(reasons.items())),
            "multi_reason_rows": sum(1 for r in rows if len(r["exclusion_reasons"]) > 1),
            "eligible_non_latin_title": sum(1 for r in eligible if r["encoder_flags"]),
            "eligible_minilm": sum(1 for r in eligible if not r["encoder_flags"]),
            "eligible_with_tags": sum(1 for r in eligible if r["tags"]),
            "duplicate_candidate_groups": dict(sorted(collections.Counter(
                g["kind"] for g in groups if g["platform"] == p).items())),
            "rows_in_duplicate_candidate_groups": sum(len(g["keys"]) for g in groups if g["platform"] == p),
        }
    return {
        "records": len(catalog),
        "eligible": sum(1 for r in catalog if r["eligibility"] == "eligible"),
        "exact_duplicate_rows": sum(1 for r in catalog if r["representative_key"] != r["problem_key"]),
        "cross_platform_exact_title_matches": len(cross_matches),
        "platforms": summary,
    }


def _write_new(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(text)


def _verify_inputs(audit_path):
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    expected = {
        LC_PROCESSED: audit["leetcode"]["inputs"]["processed"]["sha256"],
        EXTERNAL: audit["external"]["input"]["sha256"],
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise SystemExit(f"input hash mismatch: {path.relative_to(ROOT)}")
    return {str(p.relative_to(ROOT)): d for p, d in expected.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--private-output", type=Path, required=True,
                        help="new directory under private-learning/ for row-level outputs")
    parser.add_argument("--receipt", type=Path, required=True, help="committed aggregate receipt path")
    parser.add_argument("--audit", type=Path, default=HERE / "audit.json")
    args = parser.parse_args()

    input_hashes = _verify_inputs(args.audit)
    rules_path, crosswalk_path = HERE / "exclusions.v1.json", HERE / "tag_crosswalk.v1.json"
    rules = json.loads(rules_path.read_text(encoding="utf-8"))["rules"]
    aliases = json.loads(ALIASES.read_text(encoding="utf-8"))["aliases"]
    lc_rows = load_leetcode(LC_PROCESSED.read_text(encoding="utf-8").splitlines())
    external_rows = load_external(EXTERNAL.read_text(encoding="utf-8").splitlines())
    catalog, groups, cross_matches = build_catalog(lc_rows, external_rows, rules, aliases)

    crosswalk = json.loads(crosswalk_path.read_text(encoding="utf-8"))
    cf_vocab = {t for r in catalog if r["platform"] == "codeforces" for t in r["tags"] or []}
    lc_vocab = {t for r in catalog if r["platform"] == "leetcode" for t in r["tags"] or []}
    crosswalk_errors = validate_crosswalk(crosswalk, cf_vocab, lc_vocab)
    if crosswalk_errors:
        raise SystemExit("crosswalk invalid: " + ", ".join(crosswalk_errors))

    catalog_text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in catalog)
    groups_text = json.dumps({"schema": "duplicate-candidates.v1", "groups": groups,
                              "cross_platform_exact_title_matches": cross_matches},
                             ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    out = args.private_output
    _write_new(out / "catalog.v1.jsonl", catalog_text)
    _write_new(out / "duplicate_candidates.v1.json", groups_text)

    excluded = sorted([r["problem_key"], r["exclusion_reasons"]] for r in catalog if r["exclusion_reasons"])
    receipt = {
        "schema": "full-catalog-receipt.v1",
        "inputs": {**input_hashes,
                   str(rules_path.relative_to(ROOT)): sha256_file(rules_path),
                   str(crosswalk_path.relative_to(ROOT)): sha256_file(crosswalk_path),
                   str(ALIASES.relative_to(ROOT)): sha256_file(ALIASES)},
        "outputs": {"catalog.v1.jsonl": hashlib.sha256(catalog_text.encode()).hexdigest(),
                    "duplicate_candidates.v1.json": hashlib.sha256(groups_text.encode()).hexdigest()},
        "summary": summarize(catalog, groups, cross_matches),
        "excluded_keys": excluded,
        "duplicate_candidate_groups": [
            {"group_id": g["group_id"], "kind": g["kind"], "keys": g["keys"], "evidence": g["evidence"]}
            for g in groups],
        "statement_fields_read": False,
    }
    _write_new(args.receipt, json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
