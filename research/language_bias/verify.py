"""Independent validation of the language-bias study and the full-v2 map.

Recomputes claims from raw artifacts. Generic neighbor, projection, fidelity, payload and quarantine checks are
reused from research.unified_map.full_verify; v2-specific checks and the proof that the v1 artifacts are untouched
are added here.

    .venv/bin/python -m research.language_bias.verify --run private-learning/unified-map/full-v2 \
        --html private-learning/unified-map/full-v2/interactive.html \
        --output research/language_bias/receipts/validation.v1.json [--reproduce-projections]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from research.language_bias import languages as L
from research.language_bias import metrics as M
from research.language_bias import select_representation as S
from research.unified_map import full_quality as fq
from research.unified_map import full_verify as fv

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "private-learning/unified-map/full-v1"
LB = ROOT / "private-learning/unified-map/language-bias"
REC = ROOT / "research/language_bias/receipts"
V1_REC = ROOT / "research/unified_map/full-receipts"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_v1_untouched(report: fv.Report) -> None:
    emb = json.loads((V1_REC / "embeddings.v1.json").read_text(encoding="utf-8"))
    bad = [n for n, h in emb["outputs"].items() if sha256_file(V1 / n) != h]
    nb = json.loads((V1 / "neighbors/receipt.json").read_text(encoding="utf-8"))
    bad += [f"neighbors/{k}-{f}" for k in ("any", "cross") for f in ("idx", "score")
            if sha256_file(V1 / f"neighbors/{k}-{f}.npy") != nb[k][f"{f}_sha256"]]
    for receipt in sorted((V1 / "projections").glob("*.receipt.json")):
        r = json.loads(receipt.read_text(encoding="utf-8"))
        if sha256_file(V1 / "projections" / receipt.name.replace(".receipt.json", ".npy")) != r["output_sha256"]:
            bad.append(receipt.name)
    validation = json.loads((V1_REC / "validation.v1.json").read_text(encoding="utf-8"))
    if sha256_file(V1 / "interactive.html") != validation["html_sha256"]:
        bad.append("interactive.html")
    report.add("every full-v1 artifact is byte-identical to its recorded hash", not bad, changed=bad)


def check_artifacts(run: Path, report: Report_t) -> None:
    prep = [n for n in ("vectors.npy", "keys.json", "vector_meta.jsonl", "embedded_text.jsonl", "representation.json")]
    report.add("full-v2 inputs present", all((run / n).exists() for n in prep))
    nb = json.loads((run / "neighbors/receipt.json").read_text(encoding="utf-8"))
    bad = [f"{k}.{f}" for k in ("any", "cross") for f in ("idx", "score")
           if sha256_file(run / f"neighbors/{k}-{f}.npy") != nb[k][f"{f}_sha256"]]
    for receipt in sorted((run / "projections").glob("*.receipt.json")):
        r = json.loads(receipt.read_text(encoding="utf-8"))
        stem = receipt.name.replace(".receipt.json", "")
        if sha256_file(run / "projections" / f"{stem}.npy") != r["output_sha256"] or r["input_vectors_sha256"] != sha256_file(run / "vectors.npy"):
            bad.append(stem)
    report.add("v2 neighbor and projection arrays match their receipts and the v2 vectors", not bad, mismatched=bad)
    rep = json.loads((run / "representation.json").read_text(encoding="utf-8"))
    report.add("representation record binds the evaluation and selection receipts",
               rep["evaluation_sha256"] == sha256_file(REC / "evaluation.v2.json")
               and rep["selection_sha256"] == sha256_file(REC / "selection.v2.json"))


def check_vectors(run: Path, nodes, report: Report_t) -> np.ndarray:
    keys = json.loads((run / "keys.json").read_text(encoding="utf-8"))
    vectors = np.load(run / "vectors.npy", allow_pickle=False)
    node_keys = [n["problem_key"] for n in nodes]
    report.add("one v2 vector per catalog node, keys unique and sorted", keys == node_keys == sorted(keys) and vectors.shape == (24297, 384))
    norms = np.linalg.norm(vectors, axis=1)
    report.add("v2 vectors finite and unit-norm", bool(np.isfinite(vectors).all() and np.allclose(norms, 1.0, atol=1e-3)))
    cache = np.load(LB / "vectors/C2-minilm-translated.npy")
    report.add("v2 vectors equal the evaluated C2 vectors bit-for-bit", bool(np.array_equal(vectors, cache)))
    embedded = [json.loads(x) for x in (run / "embedded_text.jsonl").read_text(encoding="utf-8").splitlines()]
    meta = [json.loads(x) for x in (run / "vector_meta.jsonl").read_text(encoding="utf-8").splitlines()]
    report.add("embedded text and vector meta cover the catalog in key order",
               [e["problem_key"] for e in embedded] == node_keys == [m["problem_key"] for m in meta])
    bad_rules = 0
    for node, e in zip(nodes, embedded):
        cls = L.language_class(node["title"])
        if e["source"] == "identity" and (cls != "en" or e["text"] != node["title"]):
            bad_rules += 1
        elif e["source"] == "parenthetical" and (cls != "ja_parallel" or e["text"] != L.parse_parallel(node["title"])[1]):
            bad_rules += 1
        elif e["source"] == "machine" and (cls not in ("ja_pure", "ja_other") or not e["text"].strip()):
            bad_rules += 1
        elif e["source"] in ("untranslated", "machine_empty_fallback") and e["text"] != node["title"]:
            bad_rules += 1
    report.add("translation provenance obeys its rules for every node (identity/parenthetical/machine/untranslated)", bad_rules == 0, violations=bad_rules)
    tiers = {t: sum(1 for m in meta if m["reliability"] == t) for t in ("reliable", "degraded", "unreliable")}
    wrongly_reliable = sum(1 for m in meta if m["reliability"] == "reliable" and m["embedded_source"] in ("machine", "untranslated", "machine_empty_fallback"))
    report.add("no machine-translated or untranslated title is labelled reliable", wrongly_reliable == 0, tiers=tiers)
    return vectors, embedded


def check_reencode(embedded, vectors, report: Report_t, sample: int = 400) -> None:
    from research.language_bias.encoders import MiniLMEncoder
    rng = np.random.default_rng(31)
    idx = np.sort(rng.choice(len(embedded), size=sample, replace=False))
    redo = MiniLMEncoder().encode([embedded[i]["text"] for i in idx])
    dev = float(np.abs(redo - vectors[idx]).max())
    report.add(f"re-encoding {sample} embedded texts with MiniLM reproduces the stored v2 vectors", dev < 1e-5, max_abs_deviation=dev)


def check_receipts(run: Path, nodes, report: Report_t) -> None:
    evaluation = json.loads((REC / "evaluation.v2.json").read_text(encoding="utf-8"))
    selection = json.loads((REC / "selection.v2.json").read_text(encoding="utf-8"))
    report.add("selection receipt equals the pre-registered rule re-applied to the evaluation",
               json.loads(json.dumps(S.decide(evaluation), sort_keys=True))["winner"] == selection["winner"] == "C2"
               and selection["replaced"] is True)
    groups = np.array([L.language_group(n["title"]) for n in nodes])
    classes = np.array([L.language_class(n["title"]) for n in nodes])
    ja_pure = np.flatnonzero(classes == "ja_pure")
    for tag, directory in (("C0", V1), ("C2", run)):
        idx = np.load(directory / "neighbors/any-idx.npy")
        h = M.homophily(idx, groups, ja_pure, 10)
        saved = evaluation["candidates"][tag]["homophily"]["ja_pure"]["observed"]
        report.add(f"{tag}: saved pure-Japanese homophily equals a recomputation from the stored neighbors",
                   abs(h["observed"] - saved) < 1e-9, recomputed=h["observed"], saved=saved)
    maps = json.loads((REC / "language-maps.v1.json").read_text(encoding="utf-8"))
    for tag, directory in (("v1_minilm_raw_title", V1), ("v2_minilm_translated_title", run)):
        for method in ("umap", "tsne"):
            coords = np.load(directory / f"projections/{method}-seed17.npy").astype(np.float64)
            rows = fq.make_rows(nodes)
            hits = 0
            for q in ja_pure:
                d2 = ((coords - coords[q]) ** 2).sum(axis=1)
                allowed = np.flatnonzero(rows.representative != rows.representative[q])
                top = allowed[np.lexsort((allowed, d2[allowed]))][:10]
                hits += (groups[top] == groups[q]).mean() >= 0.9
            recomputed = hits / len(ja_pure)
            saved = maps["runs"][tag]["maps"][method]["island_index_ja_pure_2d"]
            report.add(f"{tag}/{method}: saved 2D island index equals an independent brute-force recomputation",
                       abs(recomputed - saved) <= 0.003, recomputed=recomputed, saved=saved)


Report_t = fv.Report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--quality", type=Path, default=REC / "quality-v2.v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reproduce-projections", action="store_true")
    args = parser.parse_args()
    report = fv.Report()
    nodes = fq.load_nodes()
    check_v1_untouched(report)
    check_artifacts(args.run, report)
    vectors, embedded = check_vectors(args.run, nodes, report)
    check_reencode(embedded, vectors, report)
    fv.check_neighbors(args.run, nodes, vectors, report)
    fv.check_projections(args.run, nodes, vectors, report, args.reproduce_projections, minimum=7)
    fv.check_fidelity_recompute(args.run, nodes, report)
    check_receipts(args.run, nodes, report)
    fv.check_html(args.run, args.html, nodes, report)
    fv.check_quarantine(args.run, report, args.quality)
    fv.check_existing_suite(report)
    result = {"schema": "language-bias-validation.v1", "passed": report.ok, "checks": report.checks,
              "html_sha256": sha256_file(args.html)}
    with open(args.output, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print("ALL PASSED" if report.ok else "FAILURES PRESENT")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
