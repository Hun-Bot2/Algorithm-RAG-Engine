"""Rebuild inputs for the selected representation (C2) and measure language bias in old versus new maps.

    .venv/bin/python -m research.language_bias.rebuild prepare --output private-learning/unified-map/full-v2
    .venv/bin/python -m research.language_bias.rebuild language-maps --receipt research/language_bias/receipts/language-maps.v1.json

`prepare` only reads cached C2 artifacts; `full-v1` is never written. The remaining pipeline steps reuse
research.unified_map.full_{project,quality,html} unchanged, pointed at the new run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research.language_bias import languages as L
from research.language_bias import metrics as M
from research.unified_map import full_quality as fq

ROOT = Path(__file__).resolve().parents[2]
LB = ROOT / "private-learning/unified-map/language-bias"
V1 = ROOT / "private-learning/unified-map/full-v1"
DEGRADED_UNK, UNRELIABLE_UNK = 0.10, 0.50
TIER_NOTES = {
    "reliable": "English title, or the platform's own English wording, embedded as written.",
    "degraded": "Machine-translated from Japanese (opus-mt-ja-en): on 599 reference pairs the translation has mean cosine "
                "0.70 to the platform's own English, with 21% exact matches. Neighbors may follow translation errors.",
    "unreliable": "Not translated (Russian or Korean) or no usable translation: embedded as written by an English-only model.",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reliability(source: str, unk_share: float) -> str:
    """Tier for the text that was actually embedded, from its provenance and unknown-token share."""
    if source in ("untranslated", "machine_empty_fallback") or unk_share >= UNRELIABLE_UNK:
        return "unreliable"
    if source == "machine" or unk_share >= DEGRADED_UNK:
        return "degraded"
    return "reliable"


def meta_rows(keys, classes, translations, unk_shares, token_counts) -> list[dict]:
    rows = []
    for key, cls, tr, unk, count in zip(keys, classes, translations, unk_shares, token_counts):
        rows.append({"problem_key": key, "vector_source": "c2_minilm_on_translated_title",
                     "unk_share": round(float(unk), 4), "reliability": reliability(tr["source"], unk),
                     "token_count": int(count), "embedded_source": tr["source"], "language_class": cls})
    return rows


def island_index(neighbors: np.ndarray, groups: np.ndarray, queries: np.ndarray, k: int = 10, share: float = 0.9) -> float:
    """Fraction of queries whose top-k neighbors are at least `share` in the query's own language group."""
    hit = total = 0
    for q in queries:
        nb = neighbors[q, :k]
        nb = nb[nb >= 0]
        if nb.size == 0:
            continue
        total += 1
        hit += (groups[nb] == groups[q]).mean() >= share
    return hit / total if total else float("nan")


def prepare(output: Path) -> dict:
    from research.language_bias.encoders import MiniLMEncoder
    nodes = fq.load_nodes()
    keys = [n["problem_key"] for n in nodes]
    titles = [n["title"] for n in nodes]
    classes = [L.language_class(t) for t in titles]
    lines = [json.loads(x) for x in (LB / "translations.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    header, translations = lines[0], lines[1:]
    if header["titles_sha256"] != hashlib.sha256(json.dumps(titles, ensure_ascii=False).encode()).hexdigest():
        raise SystemExit("translations were built from different titles")
    vectors = np.load(LB / "vectors/C2-minilm-translated.npy", allow_pickle=False)
    if vectors.shape != (len(keys), 384):
        raise SystemExit("unexpected C2 vector shape")
    tokenizer = MiniLMEncoder().encoder.tokenizer
    ids = tokenizer([t["text"] for t in translations], add_special_tokens=False)["input_ids"]
    unk = [(sum(1 for i in x if i == tokenizer.unk_token_id) / len(x)) if x else 1.0 for x in ids]
    meta = meta_rows(keys, classes, translations, unk, [len(x) for x in ids])
    output.mkdir(parents=True, exist_ok=False)
    with open(output / "vectors.npy", "xb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    (output / "keys.json").write_text(json.dumps(keys, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "vector_meta.jsonl").write_text("".join(json.dumps(m, sort_keys=True) + "\n" for m in meta), encoding="utf-8")
    (output / "embedded_text.jsonl").write_text("".join(json.dumps(
        {"problem_key": k, "text": t["text"], "source": t["source"]}, ensure_ascii=False, sort_keys=True) + "\n"
        for k, t in zip(keys, translations)), encoding="utf-8")
    evaluation = ROOT / "research/language_bias/receipts/evaluation.v2.json"
    selection = ROOT / "research/language_bias/receipts/selection.v2.json"
    representation = {
        "name": "C2", "description": "MiniLM-L6 on English-translated titles",
        "model": "sentence-transformers/all-MiniLM-L6-v2 @ 1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "translation": "platform English where the title is 'Japanese (English)', else Helsinki-NLP/opus-mt-ja-en @ 0770961a, beam 4; Russian and Korean untranslated",
        "input": "English-translated title (see embedded_text)", "tier_notes": TIER_NOTES,
        "evaluation_sha256": sha256_file(evaluation), "selection_sha256": sha256_file(selection)}
    (output / "representation.json").write_text(json.dumps(representation, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {"nodes": len(keys), "reliability": dict(sorted(Counter(m["reliability"] for m in meta).items())),
            "embedded_source": dict(sorted(Counter(m["embedded_source"] for m in meta).items())),
            "vectors_sha256": sha256_file(output / "vectors.npy")}


def language_maps(receipt: Path) -> dict:
    """Language homophily and fidelity by language class, for the v1 and v2 coordinates, same nodes and queries."""
    nodes = fq.load_nodes()
    rows = fq.make_rows(nodes)
    titles = [n["title"] for n in nodes]
    classes = np.array([L.language_class(t) for t in titles])
    groups = np.array([L.language_group(t) for t in titles])
    queries = {c: np.flatnonzero(classes == c) for c in ("ja_pure", "ja_parallel", "ja_other", "ru", "en")}
    out = {}
    for tag, run in (("v1_minilm_raw_title", V1), ("v2_minilm_translated_title", ROOT / "private-learning/unified-map/full-v2")):
        pres = np.load(run / "preservation.npz")
        any_idx = np.load(run / "neighbors/any-idx.npy")
        entry = {"high_dimensional": {c: M.homophily(any_idx, groups, q, 10) for c, q in queries.items()},
                 "high_dimensional_island_index_ja_pure": island_index(any_idx, groups, queries["ja_pure"]), "maps": {}}
        for method in ("pca", "umap", "tsne"):
            coords = np.load(run / f"projections/{method}-seed17.npy")
            lo, _ = fq.neighbor_arrays(rows, fq.euclidean_score_block(coords), 10, "all")
            entry["maps"][method] = {
                "homophily_2d": {c: M.homophily(lo, groups, q, 10) for c, q in queries.items()},
                "island_index_ja_pure_2d": island_index(lo, groups, queries["ja_pure"]),
                "fidelity_top10_by_class": {c: float(pres[method][q].mean() / 10) for c, q in queries.items()}}
        out[tag] = entry
    result = {"schema": "language-maps-receipt.v1", "k": 10, "island_definition": "share of pure-Japanese queries whose top-10 neighbors are at least 90% Japanese",
              "class_sizes": {c: int(len(q)) for c, q in queries.items()}, "runs": out}
    with open(receipt, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("language-maps")
    b.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.output), indent=1))
    else:
        language_maps(args.receipt)


if __name__ == "__main__":
    main()
