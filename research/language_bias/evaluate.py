"""Evaluate every candidate representation on identical node sets, queries and benchmark.

    .venv/bin/python -m research.language_bias.evaluate --candidates C0,C1,random,shuffled \
        --output research/language_bias/receipts/evaluation-baseline-phase.v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform as host_platform
import time
from pathlib import Path

import numpy as np

from research.full_catalog.retrieve import dense_cosine
from research.language_bias import languages as L
from research.language_bias import metrics as M
from research.language_bias.encoders import (CentroidRemoval, MiniLMEncoder, Representation, SentenceTransformerEncoder,
                                             cached_vectors, normalize, random_hash_vectors)
from research.unified_map import full_quality as fq

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "private-learning/unified-map/full-v1"
LB = ROOT / "private-learning/unified-map/language-bias"
K = 10
SEED = 17


class Context:
    """Shared, frozen inputs: nodes, language classes, concepts, rows, benchmark, baseline vectors."""

    def __init__(self):
        self.nodes = fq.load_nodes()
        self.titles = [n["title"] for n in self.nodes]
        self.rows = fq.make_rows(self.nodes)
        self.classes = np.array([L.language_class(t) for t in self.titles])
        self.groups = np.array([L.language_group(t) for t in self.titles])
        self.platforms = np.array([{"atcoder": 0, "codeforces": 1, "leetcode": 2}[n["platform"]] for n in self.nodes])
        self.concepts = fq.concept_sets(self.nodes)
        self.bench = L.build_benchmark(self.nodes)
        self.v0 = np.load(RUN / "vectors.npy", allow_pickle=False)
        assert json.loads((RUN / "keys.json").read_text(encoding="utf-8")) == self.rows.keys
        self.queries = {c: np.flatnonzero(self.classes == c) for c in L.CLASSES}
        self.queries["ja_all"] = np.flatnonzero(np.char.startswith(self.classes.astype(str), "ja"))
        pairs = self.bench["pairs"]
        self.ja_texts = [p["ja"] for p in pairs]
        self.en_texts = [p["en"] for p in pairs]
        unique_en = sorted(set(self.en_texts))
        folded = {t.casefold() for t in unique_en}
        self.distractor_texts = [self.titles[i] for i in self.bench["distractor_node_indices"]
                                 if self.titles[i].casefold() not in folded]
        self.cand_texts = unique_en + self.distractor_texts
        position = {t: i for i, t in enumerate(unique_en)}
        self.true_cand = np.array([position[t] for t in self.en_texts])
        self.exclude = [p["nodes"] for p in pairs]


def build_candidates(ctx: Context, names: list[str], minilm: MiniLMEncoder) -> dict[str, Representation]:
    reps = {}
    if "C0" in names or "C1" in names or "shuffled" in names:
        c0 = Representation("C0", "MiniLM-L6 on the raw title (baseline)", ctx.v0,
                            lambda texts, groups: minilm.encode(texts), 22.7)
        reps["C0"] = c0
    if "C1" in names:
        model = CentroidRemoval(ctx.v0, ctx.groups)
        reps["C1"] = Representation("C1", "MiniLM-L6 with per-language centroid removal", model.transform(ctx.v0, ctx.groups),
                                    lambda texts, groups: model.transform(minilm.encode(texts), groups), 22.7)
    if "shuffled" in names:
        perm = np.random.default_rng(SEED).permutation(ctx.groups)
        model = CentroidRemoval(ctx.v0, perm)
        reps["shuffled"] = Representation("shuffled", "control: centroid removal with permuted language labels",
                                          model.transform(ctx.v0, perm),
                                          lambda texts, groups: model.transform(minilm.encode(texts), groups), 22.7)
    for tag, model_name, millions in (("C3", "e5-small", 118.0), ("C4", "paraphrase-multilingual", 118.0), ("C5", "labse", 471.0)):
        if tag in names:
            encoder = SentenceTransformerEncoder(model_name)
            vectors, meta = cached_vectors(LB / "vectors" / f"{tag}-{model_name}.npy", ctx.titles, encoder.encode)
            reps[tag] = Representation(tag, f"{model_name} on the raw title", vectors,
                                       lambda texts, groups, e=encoder: e.encode(texts), millions, notes=meta)
    if "C2" in names:
        from research.language_bias import translate as T
        mt = T.CachedMT(T.MarianTranslator())
        rows = T.cached_translations(LB / "translations.v1.jsonl", ctx.titles, [str(c) for c in ctx.classes], mt)
        vectors, meta = cached_vectors(LB / "vectors" / "C2-minilm-translated.npy", [r["text"] for r in rows], minilm.encode)
        meta = {**meta, "translation_provenance": T.provenance_counts(rows, [str(c) for c in ctx.classes]),
                "mt_quality_on_parallel_pairs": mt_quality(ctx, mt, minilm)}
        reps["C2"] = Representation(
            "C2", "MiniLM-L6 on English-translated titles (parenthetical English, else opus-mt-ja-en)", vectors,
            lambda texts, groups: minilm.encode(_translate_queries(mt, texts, groups)), 22.7 + 76.0, notes=meta)
    if "random" in names:
        reps["random"] = Representation("random", "control: random unit vector per exact title",
                                        random_hash_vectors(ctx.titles), lambda texts, groups: random_hash_vectors(texts), 0.0)
    return reps


def _translate_queries(mt, texts: list[str], groups: list[str]) -> list[str]:
    """Japanese halves go through MT; English texts are used as they are."""
    ja = [t for t, g in zip(texts, groups) if g == "ja"]
    done = dict(zip(ja, mt(ja))) if ja else {}
    return [done[t].strip() or t if g == "ja" else t for t, g in zip(texts, groups)]


def mt_quality(ctx: Context, mt, minilm: MiniLMEncoder) -> dict:
    """How close the machine translation of each Japanese half is to the platform's own English half."""
    import re
    out = mt(ctx.ja_texts)
    norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.casefold()).strip()
    exact = float(np.mean([norm(a) == norm(b) for a, b in zip(out, ctx.en_texts)]))
    a, b = minilm.encode(out), minilm.encode(ctx.en_texts)
    cos = np.sum(a * b, axis=1)
    shuffled = np.sum(a * b[np.random.default_rng(SEED).permutation(len(b))], axis=1)
    return {"pairs": len(out), "exact_match_after_normalization": exact, "mean_cosine_to_reference": float(cos.mean()),
            "median_cosine_to_reference": float(np.median(cos)), "mean_cosine_to_shuffled_reference": float(shuffled.mean())}


def tag_lift(idx: np.ndarray, ctx: Context) -> dict:
    obs = fq.tag_agreement(idx, ctx.concepts)
    base = fq.matched_random_tag_agreement(idx, ctx.concepts, ctx.platforms)
    lift = obs["share_with_concept"] / base["share_with_concept"] if base["share_with_concept"] else None
    return {"observed": obs["share_with_concept"], "platform_matched_random": base["share_with_concept"], "lift": lift,
            "tagged_slot_coverage": obs["tagged_slot_coverage"]}


def evaluate_rep(rep: Representation, ctx: Context, baseline_idx: np.ndarray | None, baseline_per: dict | None) -> tuple[dict, dict]:
    started = time.perf_counter()
    idx, _ = fq.neighbor_arrays(ctx.rows, dense_cosine(rep.index_vectors), K, "all")
    out = {"name": rep.name, "description": rep.description, "parameters_millions": rep.parameters_millions,
           "encoding": rep.notes}
    out["homophily"] = {c: M.homophily(idx, ctx.groups, ctx.queries[c], K)
                        for c in ("ja_pure", "ja_parallel", "ja_other", "ja_all", "ru", "en") if len(ctx.queries[c])}
    out["platform_composition_ja_pure"] = M.platform_composition(idx, ctx.platforms, ctx.queries["ja_pure"], K)
    out["tag_lift"] = tag_lift(idx, ctx)
    out["pure_japanese_concept_coherence"] = M.concept_coherence(idx, ctx.concepts, ctx.platforms, ctx.queries["ja_pure"], K)
    ja_vec = rep.encode_query(ctx.ja_texts, ["ja"] * len(ctx.ja_texts))
    en_vec = rep.encode_query(ctx.en_texts, ["en"] * len(ctx.en_texts))
    cand_vec = rep.encode_query(ctx.cand_texts, ["en"] * len(ctx.cand_texts))
    tr = M.translation_retrieval(ja_vec, cand_vec, ctx.true_cand)
    cons = M.neighbor_consistency(rep.index_vectors, ja_vec, en_vec, ctx.exclude, K)
    per = {"rr": tr.pop("per_query_rr"), "overlap": cons.pop("per_query_overlap")}
    every_pair_node = sorted(ctx.bench["pair_node_indices"])
    sens = {"neighbor_pool_without_any_parallel_node_k10": M.neighbor_consistency(
                rep.index_vectors, ja_vec, en_vec, [every_pair_node] * len(ja_vec), 10),
            "k5": M.neighbor_consistency(rep.index_vectors, ja_vec, en_vec, ctx.exclude, 5),
            "k20": M.neighbor_consistency(rep.index_vectors, ja_vec, en_vec, ctx.exclude, 20)}
    out["consistency_sensitivity"] = {k: {"overlap": v["overlap"], "chance": v["chance"]} for k, v in sens.items()}
    out["translation_retrieval"] = tr
    out["cross_lingual_neighbor_consistency"] = cons
    if baseline_per is not None:
        out["consistency_diff_vs_C0"] = M.paired_difference(per["overlap"], baseline_per["overlap"])
        out["mrr_diff_vs_C0"] = M.paired_difference(per["rr"], baseline_per["rr"])
    if baseline_idx is not None:
        en_q = ctx.queries["en"]
        out["english_neighbor_overlap_with_C0"] = float(fq.preservation_counts(baseline_idx[en_q], idx[en_q], K).mean() / K)
    out["seconds"] = round(time.perf_counter() - started, 1)
    return out, {"idx": idx, **per}


def run(names: list[str], output: Path, extra: dict[str, Representation] | None = None) -> dict:
    ctx = Context()
    minilm = MiniLMEncoder()
    reps = build_candidates(ctx, names, minilm)
    reps.update(extra or {})
    results, per = {}, {}
    order = ["C0"] + [n for n in reps if n != "C0"]
    for name in order:
        if name not in reps:
            continue
        base_idx = per["C0"]["idx"] if "C0" in per else None
        base_per = per["C0"] if "C0" in per else None
        results[name], per[name] = evaluate_rep(reps[name], ctx, base_idx, base_per)
        print("evaluated", name, results[name]["seconds"], "s")
    c0 = results["C0"]["tag_lift"]["lift"]
    for name, r in results.items():
        r["english_quality_retention"] = r["tag_lift"]["lift"] / c0 if c0 else None
    report = {
        "schema": "language-bias-evaluation.v1",
        "population": {"nodes": len(ctx.nodes), "class_counts": {c: int(len(v)) for c, v in ctx.queries.items()},
                       "benchmark": ctx.bench["counts"], "benchmark_pairs_sha256": ctx.bench["sha256_of_pairs"],
                       "translation_candidates": len(ctx.cand_texts)},
        "k": K, "seed": SEED, "bootstrap_resamples": M.BOOT,
        "language_chance_note": "homophily chance is the share of same-language nodes under language-blind neighbors",
        "candidates": results,
        "environment": {"numpy": np.__version__, "python": host_platform.python_version(), "machine": host_platform.machine()},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run([c for c in args.candidates.split(",") if c], args.output)


if __name__ == "__main__":
    main()
