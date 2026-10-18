"""Title language classification, natural parallel-title extraction and the frozen benchmark.

Pure functions, standard library only. A "parallel" title is written "Japanese (English)": the
English half is the platform's own wording, so it is a labeled translation pair that needs no
machine or human translation by us.
"""

from __future__ import annotations

import hashlib
import json
import re

JA = re.compile(r"[぀-ヿ一-鿿]")
KO = re.compile(r"[가-힯]")
RU = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")
PARALLEL = re.compile(
    r"^(?P<ja>[^()（）]*[぀-ヿ一-鿿][^()（）]*?)\s*[(（]\s*(?P<en>[A-Za-z][^()（）぀-ヿ一-鿿]*?)\s*[)）]\s*$")
CLASSES = ("en", "ja_pure", "ja_parallel", "ja_other", "ru", "ko")
SEED = 17
DISTRACTORS = 5000


def parse_parallel(title: str) -> tuple[str, str] | None:
    """Return (japanese half, english half) for 'Japanese (English)' titles, else None."""
    match = PARALLEL.match(title.strip())
    if not match:
        return None
    ja, en = match.group("ja").strip(), match.group("en").strip()
    if not ja or len(LATIN.findall(en)) < 2:
        return None
    return ja, en


def language_class(title: str) -> str:
    """One class per title, by script; Japanese is split by how much English it carries."""
    if JA.search(title):
        if parse_parallel(title):
            return "ja_parallel"
        return "ja_other" if LATIN.search(title) else "ja_pure"
    if KO.search(title):
        return "ko"
    if RU.search(title):
        return "ru"
    return "en"


def language_group(title: str) -> str:
    """Coarse group used for homophily: en, ja, ru, ko."""
    cls = language_class(title)
    return "ja" if cls.startswith("ja") else cls


def class_index(nodes: list[dict]) -> dict[str, list[int]]:
    out = {c: [] for c in CLASSES}
    for i, node in enumerate(nodes):
        out[language_class(node["title"])].append(i)
    return out


def build_benchmark(nodes: list[dict], seed: int = SEED, distractors: int = DISTRACTORS) -> dict:
    """Frozen translation benchmark: unique parallel (ja, en) text pairs plus English distractor titles."""
    by_pair: dict[tuple[str, str], list[int]] = {}
    for i, node in enumerate(nodes):
        parsed = parse_parallel(node["title"]) if JA.search(node["title"]) else None
        if parsed:
            by_pair.setdefault(parsed, []).append(i)
    pairs = [{"ja": ja, "en": en, "nodes": idx} for (ja, en), idx in sorted(by_pair.items())]
    pair_nodes = {i for p in pairs for i in p["nodes"]}
    pool = [i for i, n in enumerate(nodes) if language_class(n["title"]) == "en" and i not in pair_nodes]
    ordered = sorted(pool, key=lambda i: hashlib.sha256(f"lang-bias-v1:{seed}:{nodes[i]['problem_key']}".encode()).hexdigest())
    chosen = sorted(ordered[:distractors])
    return {
        "schema": "translation-benchmark.v1",
        "seed": seed,
        "pairs": pairs,
        "pair_node_indices": sorted(pair_nodes),
        "distractor_node_indices": chosen,
        "counts": {"pairs": len(pairs), "pair_nodes": len(pair_nodes), "distractors": len(chosen)},
        "sha256_of_pairs": hashlib.sha256(json.dumps(
            [[p["ja"], p["en"]] for p in pairs], ensure_ascii=False).encode()).hexdigest(),
    }
