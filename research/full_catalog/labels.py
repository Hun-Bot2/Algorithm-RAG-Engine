"""Judged-pool labeling pipeline (stdlib only). See LABELING.md.

    python3 research/full_catalog/labels.py batches  --sheet S --out-dir D --count 10
    python3 research/full_catalog/labels.py merge    --batches D --draft A --review B --out-dir M
    python3 research/full_catalog/labels.py finalize --merged M --adjudication J --labels L --receipt R
"""

import argparse
import json
from collections import Counter
from pathlib import Path

LABELS = (0, 1, 2, 3)
BASES = ("metadata_only", "recognized_problem")
CONFIDENCE = ("high", "medium", "low")
FIELDS = {"pair_id", "label", "deceptive", "basis", "confidence", "rationale", "abstain_reason"}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_new_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        handle.writelines(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def make_batches(sheet_rows, count):
    """Tier-1 pairs grouped by query; queries assigned greedily to the lightest batch."""
    by_query = {}
    for row in sheet_rows:
        if row["tier"] != 1:
            continue
        by_query.setdefault(row["query"]["key"], []).append({
            "pair_id": row["pair_id"],
            "query": {k: row["query"][k] for k in ("key", "title", "tags", "difficulty", "url")},
            "target": {k: row["target"][k] for k in ("key", "title", "tags", "difficulty", "url")},
        })
    batches = [[] for _ in range(count)]
    for query in sorted(by_query, key=lambda q: (-len(by_query[q]), q)):
        lightest = min(range(count), key=lambda i: (len(batches[i]), i))
        batches[lightest].extend(by_query[query])
    return [sorted(b, key=lambda r: (r["query"]["key"], r["pair_id"])) for b in batches]


def validate(rows, expected_ids):
    """Return a list of problems with a labeler's output for one batch."""
    errors = []
    ids = [r.get("pair_id") for r in rows]
    if sorted(ids) != sorted(expected_ids):
        missing, extra = set(expected_ids) - set(ids), set(ids) - set(expected_ids)
        errors.append(f"pair_ids differ: missing={len(missing)} extra={len(extra)} dup={len(ids) - len(set(ids))}")
    for r in rows:
        pid = r.get("pair_id")
        if set(r) != FIELDS:
            errors.append(f"{pid}: fields {sorted(set(r) ^ FIELDS)}")
            continue
        if r["label"] is None:
            if r["deceptive"] is not None or not r["abstain_reason"]:
                errors.append(f"{pid}: abstention needs deceptive=null and abstain_reason")
        else:
            if r["label"] not in LABELS or type(r["label"]) is not int:
                errors.append(f"{pid}: bad label")
            if r["deceptive"] not in (True, False) or (r["deceptive"] and r["label"] > 1):
                errors.append(f"{pid}: deceptive must be bool and only with label 0/1")
            if r["abstain_reason"] is not None:
                errors.append(f"{pid}: abstain_reason with a label")
        if r["basis"] not in BASES or r["confidence"] not in CONFIDENCE:
            errors.append(f"{pid}: bad basis/confidence")
        if not isinstance(r["rationale"], str) or len(r["rationale"].split()) > 25:
            errors.append(f"{pid}: rationale missing or too long")
    return errors


def resolve(a, b):
    """Apply LABELING.md resolution. Returns (final_label, deceptive, resolution) or None."""
    if a["label"] == b["label"] and a["deceptive"] == b["deceptive"]:
        return a["label"], a["deceptive"], "agreement"
    if (a["label"] is not None and b["label"] is not None and abs(a["label"] - b["label"]) == 1
            and a["deceptive"] == b["deceptive"]):
        return min(a["label"], b["label"]), a["deceptive"], "conservative_min"
    return None


def quadratic_kappa(pairs):
    """Quadratic-weighted Cohen's kappa over (a, b) integer label pairs."""
    n = len(pairs)
    if n == 0:
        return None
    k = len(LABELS)
    observed = Counter(pairs)
    ma, mb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    num = den = 0.0
    for i in LABELS:
        for j in LABELS:
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * observed[(i, j)] / n
            den += w * (ma[i] / n) * (mb[j] / n)
    return 1 - num / den if den else None


def merge(batch_dir, draft_dir, review_dir):
    merged, adjudicate, errors = [], [], {}
    for batch_path in sorted(batch_dir.glob("batch-*.jsonl")):
        batch = _read_jsonl(batch_path)
        ids = [r["pair_id"] for r in batch]
        passes = {}
        for name, directory in (("draft", draft_dir), ("review", review_dir)):
            path = directory / batch_path.name
            rows = _read_jsonl(path) if path.exists() else []
            problems = validate(rows, ids)
            if problems:
                errors[f"{name}/{batch_path.name}"] = problems[:20]
            passes[name] = {r["pair_id"]: r for r in rows}
        for item in batch:
            pid = item["pair_id"]
            a, b = passes["draft"].get(pid), passes["review"].get(pid)
            if a is None or b is None:
                continue
            outcome = resolve(a, b)
            record = {"pair_id": pid, "query": item["query"]["key"], "target": item["target"]["key"],
                      "draft": a, "review": b}
            if outcome is None:
                adjudicate.append({**item, "draft": _public(a), "review": _public(b)})
                record.update(label=None, deceptive=None, resolution="pending_adjudication")
            else:
                record.update(label=outcome[0], deceptive=outcome[1], resolution=outcome[2])
            merged.append(record)
    return merged, adjudicate, errors


def _public(r):
    return {k: r[k] for k in ("label", "deceptive", "basis", "confidence", "rationale", "abstain_reason")}


def agreement(merged):
    both = [(m["draft"]["label"], m["review"]["label"]) for m in merged]
    labelled = [(a, b) for a, b in both if a is not None and b is not None]
    return {
        "pairs": len(both),
        "exact_label_agreement": sum(a == b for a, b in both) / len(both) if both else None,
        "both_labelled": len(labelled),
        "within_one": sum(abs(a - b) <= 1 for a, b in labelled) / len(labelled) if labelled else None,
        "quadratic_weighted_kappa": quadratic_kappa(labelled),
        "abstain": {"draft": sum(a is None for a, _ in both), "review": sum(b is None for _, b in both)},
        "resolution": dict(Counter(m["resolution"] for m in merged)),
    }


def finalize(merged, adjudications):
    by_id = {r["pair_id"]: r for r in adjudications}
    final = []
    for m in merged:
        row = {"pair_id": m["pair_id"], "query": m["query"], "target": m["target"],
               "label_source": "ai_reviewed", "label": m["label"], "deceptive": m["deceptive"],
               "resolution": m["resolution"],
               "passes": {"draft": _public(m["draft"]), "review": _public(m["review"])}}
        if m["resolution"] == "pending_adjudication":
            adj = by_id.get(m["pair_id"])
            if adj is None:
                raise ValueError(f"missing adjudication {m['pair_id']}")
            row.update(label=adj["label"], deceptive=adj["deceptive"], resolution="adjudicated")
            row["passes"]["adjudication"] = _public(adj)
        final.append(row)
    return sorted(final, key=lambda r: (r["query"], r["pair_id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("batches")
    b.add_argument("--sheet", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--count", type=int, default=10)
    m = sub.add_parser("merge")
    for name in ("--batches", "--draft", "--review", "--out-dir"):
        m.add_argument(name, type=Path, required=True)
    f = sub.add_parser("finalize")
    for name in ("--merged", "--adjudication", "--labels", "--receipt"):
        f.add_argument(name, type=Path, required=True)
    f.add_argument("--labelers", required=True, help="e.g. 'draft:sonnet; review:opus; adjudication:opus'")
    args = parser.parse_args()

    if args.command == "batches":
        for i, batch in enumerate(make_batches(_read_jsonl(args.sheet), args.count), start=1):
            _write_new_jsonl(args.out_dir / f"batch-{i:02d}.jsonl", batch)
    elif args.command == "merge":
        merged, adjudicate, errors = merge(args.batches, args.draft, args.review)
        if errors:
            raise SystemExit(json.dumps(errors, indent=1))
        _write_new_jsonl(args.out_dir / "merged.jsonl", merged)
        for i in range(0, len(adjudicate), 200):
            _write_new_jsonl(args.out_dir / f"adjudicate-{i // 200 + 1:02d}.jsonl", adjudicate[i:i + 200])
        print(json.dumps(agreement(merged), indent=1))
    else:
        merged = _read_jsonl(args.merged / "merged.jsonl")
        adjudications = [r for p in sorted(args.adjudication.glob("adjudicate-*.jsonl")) for r in _read_jsonl(p)]
        pending = [m for m in merged if m["resolution"] == "pending_adjudication"]
        problems = validate(adjudications, [m["pair_id"] for m in pending])
        if problems:
            raise SystemExit("\n".join(problems[:30]))
        final = finalize(merged, adjudications)
        _write_new_jsonl(args.labels, final)
        receipt = {"schema": "pool-labels-receipt.v1", "label_source": "ai_reviewed", "labelers": args.labelers,
                   "guideline": "research/full_catalog/LABELING.md", "pool": "research/full_catalog/pool.v1.json",
                   "agreement_before_adjudication": agreement(merged),
                   "final": {"pairs": len(final),
                             "labels": dict(sorted(Counter(str(r["label"]) for r in final).items())),
                             "deceptive": sum(1 for r in final if r["deceptive"]),
                             "resolution": dict(Counter(r["resolution"] for r in final)),
                             "basis_recognized_any_pass": sum(1 for r in final if any(
                                 p["basis"] == "recognized_problem" for p in r["passes"].values()))}}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with open(args.receipt, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
