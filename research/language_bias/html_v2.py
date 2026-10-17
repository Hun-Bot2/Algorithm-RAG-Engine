"""Build the interactive page for the selected representation (full-v2).

Reuses research.unified_map.full_html unchanged and adds optional payload fields (language class, the text
actually embedded and its source, representation notes, a language-bias summary). Output is private.

    .venv/bin/python -m research.language_bias.html_v2 --run private-learning/unified-map/full-v2 \
        --quality research/language_bias/receipts/quality-v2.v1.json \
        --evaluation research/language_bias/receipts/evaluation.v2.json \
        --maps research/language_bias/receipts/language-maps.v1.json \
        --output private-learning/unified-map/full-v2/interactive.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research.language_bias import languages as L
from research.unified_map import full_html as fh
from research.unified_map import full_quality as fq

SOURCES = ("identity", "parenthetical", "machine", "machine_empty_fallback", "untranslated")


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def language_bias_summary(evaluation: dict, maps: dict) -> dict:
    """Structured rows for the page's language-bias panel (the page escapes every string)."""
    c0, c2 = evaluation["candidates"]["C0"], evaluation["candidates"]["C2"]
    v1 = maps["runs"]["v1_minilm_raw_title"]
    v2 = maps["runs"]["v2_minilm_translated_title"]
    hom = lambda c: c["homophily"]["ja_pure"]
    chance = hom(c0)["chance"]
    rows = [
        {"label": "Pure-Japanese titles: neighbors that are also Japanese (384-d)", "before": pct(hom(c0)["observed"]),
         "after": pct(hom(c2)["observed"])},
        {"label": "  language-blind chance level", "before": pct(chance), "after": pct(chance)},
        {"label": "Japanese half vs English half: top-10 neighbor overlap", "before": f"{c0['cross_lingual_neighbor_consistency']['overlap']['mean']:.3f}",
         "after": f"{c2['cross_lingual_neighbor_consistency']['overlap']['mean']:.3f}"},
        {"label": "Japanese half retrieves its English half (MRR)", "before": f"{c0['translation_retrieval']['mrr']['mean']:.3f}",
         "after": f"{c2['translation_retrieval']['mrr']['mean']:.3f}"},
        {"label": "English-title neighbor quality retained (tag lift ratio)", "before": "1.000",
         "after": f"{c2['english_quality_retention']:.3f}"},
    ]
    for method, name in (("umap", "UMAP"), ("tsne", "t-SNE")):
        rows.append({"label": f"{name}: pure-Japanese titles in a Japanese-only 2D neighborhood",
                     "before": pct(v1["maps"][method]["island_index_ja_pure_2d"]),
                     "after": pct(v2["maps"][method]["island_index_ja_pure_2d"])})
    return {"title": "Language bias (before = raw titles, after = English-translated titles)", "before_label": "Raw titles",
            "after_label": "Translated", "rows": rows,
            "caveats": ["Translation is machine output for 1,561 titles and is only tested on titles that already carry English.",
                        "Lower language clustering is not accepted as a gain by itself: the random control also scores near chance.",
                        "Russian and Korean titles are not translated and still cluster by language."]}


def extras(run: Path, evaluation: dict, maps: dict) -> dict:
    nodes = fq.load_nodes()
    classes = [L.language_class(n["title"]) for n in nodes]
    embedded = [json.loads(x) for x in (run / "embedded_text.jsonl").read_text(encoding="utf-8").splitlines()]
    if [e["problem_key"] for e in embedded] != [n["problem_key"] for n in nodes]:
        raise ValueError("embedded_text order differs from nodes")
    differing = {str(i): e["text"] for i, (n, e) in enumerate(zip(nodes, embedded)) if e["text"] != n["title"]}
    representation = json.loads((run / "representation.json").read_text(encoding="utf-8"))
    representation["banner"] = (
        "Positions and scores come from a MiniLM embedding of each problem's English title: the platform's own English wording "
        "or a machine translation where the title is Japanese. They show how alike two titles read, not that two problems share an "
        "algorithm, invariant or difficulty. Platform tags and the AI-reviewed taxonomy are separate evidence and were never used to "
        "place points or rank neighbors. AtCoder publishes no tags. Translations can be wrong; see the language-bias numbers.")
    return {"language": {"vocab": list(L.CLASSES), "codes": fh.pack_array([L.CLASSES.index(c) for c in classes], "u8")},
            "embedded": {"text": differing, "source": fh.pack_array([SOURCES.index(e["source"]) for e in embedded], "u8"),
                         "source_vocab": list(SOURCES)},
            "representation": representation, "language_bias": language_bias_summary(evaluation, maps)}


def build(run: Path, quality_path: Path, evaluation_path: Path, maps_path: Path, output: Path) -> None:
    nodes = sorted(fh._read_jsonl(fh.NODES), key=lambda n: n["problem_key"])
    meta = fh._read_jsonl(run / "vector_meta.jsonl")
    neighbors = run / "neighbors"
    coords = {n: np.load(run / "projections" / f"{n}-seed17.npy") for n in fh.PROJECTIONS}
    pres = np.load(run / "preservation.npz")
    aliases = {a["legacy_key"]: a["native_key"] for a in json.loads(fh.ALIASES.read_text(encoding="utf-8"))["aliases"]}
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    maps = json.loads(maps_path.read_text(encoding="utf-8"))
    run_info = {"vectors_sha256": hashlib.sha256((run / "vectors.npy").read_bytes()).hexdigest(),
                "quality_receipt_sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
                "model": "sentence-transformers/all-MiniLM-L6-v2 @ 1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                "input": "English-translated title (opus-mt-ja-en for Japanese-only titles)"}
    payload = fh.build_payload(
        nodes, meta, np.load(neighbors / "any-idx.npy"), np.load(neighbors / "any-score.npy"),
        np.load(neighbors / "cross-idx.npy"), np.load(neighbors / "cross-score.npy"), coords,
        {n: pres[n] for n in fh.PROJECTIONS}, json.loads(quality_path.read_text(encoding="utf-8")),
        fh._read_jsonl(fh.LABELS), json.loads(fh.HISTORY.read_text(encoding="utf-8")), aliases, run_info,
        extras=extras(run, evaluation, maps))
    html = fh.render_html(payload, fh.TEMPLATE.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "x", encoding="utf-8") as handle:
        handle.write(html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--run", "--quality", "--evaluation", "--maps", "--output"):
        parser.add_argument(name, type=Path, required=True)
    a = parser.parse_args()
    build(a.run, a.quality, a.evaluation, a.maps, a.output)


if __name__ == "__main__":
    main()
