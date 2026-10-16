"""Shared row selection and input text for every Phase 3-5 method (standard library only).

Embeddings and baselines MUST build their inputs here, so that each method in a
variant sees exactly the same rows and text (spec P5-1).

Variants:
- A1      title only; eligible rows without encoder flags (MiniLM-comparable).
- A1-all  title only; all eligible rows, including non-Latin titles (TF-IDF diagnostic only).
- A2      "{title}. Topics: {tags}."; eligible LeetCode/Codeforces rows with tags, no encoder flags.
"""

import hashlib
import json

VARIANTS = ("A1", "A1-all", "A2")
IGNORED_TAGS = frozenset({"*special"})


def _a2_text(row):
    tags = [t for t in row["tags"] if t not in IGNORED_TAGS]
    return f"{row['title'].strip()}. Topics: {', '.join(tags)}."


def select(catalog_rows, variant):
    """Return (rows, texts) sorted by problem key for the given variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown_variant:{variant}")
    chosen = []
    for row in catalog_rows:
        if row["eligibility"] != "eligible":
            continue
        if variant != "A1-all" and row["encoder_flags"]:
            continue
        if variant == "A2":
            if row["platform"] not in ("leetcode", "codeforces"):
                continue
            if not [t for t in row["tags"] or [] if t not in IGNORED_TAGS]:
                continue
        chosen.append(row)
    chosen.sort(key=lambda r: r["problem_key"])
    texts = [_a2_text(r) if variant == "A2" else r["title"].strip() for r in chosen]
    return chosen, texts


def input_digest(rows, texts):
    """Hash of the exact (key, text) sequence a method receives."""
    payload = json.dumps([[r["problem_key"], t] for r, t in zip(rows, texts)], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_catalog(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
