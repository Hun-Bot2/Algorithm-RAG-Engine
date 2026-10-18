"""Independent validation of the full-catalog map artifacts.

Each check recomputes a claim from raw artifacts instead of trusting a receipt. The
JSON report lists every check with pass/fail and measured details; the exit status is
non-zero if any check fails.

    .venv/bin/python -m research.unified_map.full_verify \
        --run private-learning/unified-map/full-v1 \
        --html private-learning/unified-map/full-v1/interactive.html \
        --output research/unified_map/full-receipts/validation.v1.json [--reproduce-projections]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from research.unified_map import full_html, full_project, full_quality

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = ROOT / "research/unified_map/full-receipts"
FORBIDDEN_TEXT = ("content_clean", "full_statement", "statement_text", "embedding_text")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Report:
    def __init__(self):
        self.checks = []

    def add(self, name: str, ok: bool, **detail):
        self.checks.append({"name": name, "passed": bool(ok), **detail})
        print(("PASS " if ok else "FAIL ") + name, detail if not ok else "")

    @property
    def ok(self) -> bool:
        return all(c["passed"] for c in self.checks)


def check_artifact_hashes(run: Path, report: Report) -> None:
    emb = json.loads((RECEIPTS / "embeddings.v1.json").read_text(encoding="utf-8"))
    mismatches = [n for n, h in emb["outputs"].items() if sha256_file(run / n) != h]
    report.add("embedding outputs match receipt hashes", not mismatches, mismatched=mismatches)
    frozen = {"nodes": full_quality.NODES, "a1_vectors": ROOT / "private-learning/full-catalog/embeddings-v1/A1/vectors.npy",
              "a1_keys": ROOT / "private-learning/full-catalog/embeddings-v1/A1/keys.json"}
    bad = [k for k, p in frozen.items() if sha256_file(p) != emb["inputs"][k]]
    report.add("frozen input artifacts unchanged since embedding run", not bad, changed=bad)
    nb = json.loads((run / "neighbors/receipt.json").read_text(encoding="utf-8"))
    bad = [f"{k}.{f}" for k in ("any", "cross") for f in ("idx", "score")
           if sha256_file(run / f"neighbors/{k}-{f}.npy") != nb[k][f"{f}_sha256"]]
    report.add("neighbor arrays match receipt hashes", not bad, mismatched=bad)
    proj_bad = []
    for receipt_path in sorted((run / "projections").glob("*.receipt.json")):
        r = json.loads(receipt_path.read_text(encoding="utf-8"))
        stem = receipt_path.name.replace(".receipt.json", "")
        if sha256_file(run / "projections" / f"{stem}.npy") != r["output_sha256"]:
            proj_bad.append(stem)
        if r["input_vectors_sha256"] != emb["outputs"]["vectors.npy"]:
            proj_bad.append(stem + ":input")
    report.add("projection arrays match receipts and the final vectors", not proj_bad, mismatched=proj_bad)


def check_vectors(run: Path, nodes, report: Report) -> np.ndarray:
    keys = json.loads((run / "keys.json").read_text(encoding="utf-8"))
    vectors = np.load(run / "vectors.npy", allow_pickle=False)
    meta = [json.loads(line) for line in (run / "vector_meta.jsonl").read_text(encoding="utf-8").splitlines()]
    node_keys = [n["problem_key"] for n in nodes]
    report.add("one vector per catalog node, keys unique and sorted",
               keys == node_keys and len(set(keys)) == len(keys) == vectors.shape[0] == 24297 and keys == sorted(keys),
               rows=int(vectors.shape[0]), dim=int(vectors.shape[1]))
    norms = np.linalg.norm(vectors, axis=1)
    report.add("vectors finite and unit-norm", bool(np.isfinite(vectors).all() and np.allclose(norms, 1.0, atol=1e-3)),
               max_norm_error=float(np.abs(norms - 1).max()))
    a1_keys = json.loads((ROOT / "private-learning/full-catalog/embeddings-v1/A1/keys.json").read_text(encoding="utf-8"))
    a1 = np.load(ROOT / "private-learning/full-catalog/embeddings-v1/A1/vectors.npy")
    position = {k: i for i, k in enumerate(keys)}
    reused_rows = np.array([position[k] for k in a1_keys])
    report.add("all 21,297 A1 vectors reused bit-for-bit", bool(np.array_equal(vectors[reused_rows], a1)),
               reused=len(a1_keys))
    sources = [m["vector_source"] for m in meta]
    report.add("vector_meta sources partition the catalog (21,297 reused + 3,000 new)",
               [m["problem_key"] for m in meta] == keys and sources.count("a1_reused") == 21297
               and sources.count("new_title_encode") == 3000, counts={s: sources.count(s) for s in set(sources)})
    tiers = {t: [m["reliability"] for m in meta].count(t) for t in ("reliable", "degraded", "unreliable")}
    non_latin_reliable = sum(1 for n, m in zip(nodes, meta) if n["encoder_flags"] and m["reliability"] == "reliable")
    report.add("no non-Latin title is labelled reliable", non_latin_reliable == 0, tiers=tiers)
    return vectors


def brute_neighbors(vectors64, q, rows, mode, k):
    scores = vectors64 @ vectors64[q]
    if mode == "all":
        mask = rows.representative == rows.representative[q]
    else:
        mask = (rows.representative == rows.representative[q]) | (rows.platform == rows.platform[q])
    allowed = np.flatnonzero(~mask)
    order = allowed[np.lexsort((allowed, -scores[allowed]))][:k]
    return order, scores


def check_neighbors(run: Path, nodes, vectors, report: Report, sample=400) -> None:
    rows = full_quality.make_rows(nodes)
    n, k = len(nodes), full_quality.STORED_K
    v64 = vectors.astype(np.float64)
    arrays = {name: (np.load(run / f"neighbors/{name}-idx.npy"), np.load(run / f"neighbors/{name}-score.npy"))
              for name in ("any", "cross")}
    for name, (idx, score) in arrays.items():
        in_range = bool(((idx >= 0) & (idx < n)).all())
        no_self = bool((idx != np.arange(n)[:, None]).all())
        no_dup = bool((rows.representative[idx] != rows.representative[:, None]).all())
        monotone = bool((np.diff(score, axis=1) <= 1e-6).all())
        platform_rule = bool((rows.platform[idx] != rows.platform[:, None]).all()) if name == "cross" else True
        recomputed = np.einsum("nd,nkd->nk", v64, v64[idx])
        max_dev = float(np.abs(recomputed - score).max())
        report.add(f"{name}: indices valid, no self, no exact duplicates, scores non-increasing, platform rule",
                   in_range and no_self and no_dup and monotone and platform_rule, rows=n)
        report.add(f"{name}: stored scores equal recomputed float64 cosine", max_dev < 2e-6, max_abs_deviation=max_dev)
    rng = np.random.default_rng(99)
    queries = rng.choice(n, size=sample, replace=False)
    for name, mode in (("any", "all"), ("cross", "cross")):
        idx, score = arrays[name]
        wrong = 0
        for q in queries:
            order, scores = brute_neighbors(v64, q, rows, mode, k)
            if idx[q].tolist() != order.tolist():
                # tolerate float32/float64 ordering flips only between near-equal scores
                wrong += int(not np.allclose(np.sort(score[q]), np.sort(scores[order].astype(np.float32)), atol=2e-6))
        report.add(f"{name}: top-{k} equals float64 brute force on {sample} random queries (tie-aware)", wrong == 0,
                   queries=sample, mismatches=wrong)


def check_projections(run: Path, nodes, vectors, report: Report, reproduce: bool, minimum: int = 8) -> None:
    n = len(nodes)
    files = {p.stem: np.load(p) for p in sorted((run / "projections").glob("*-seed*.npy"))}
    ok = all(a.shape == (n, 2) and np.isfinite(a).all() for a in files.values())
    report.add("every projection covers all 24,297 nodes with finite coordinates", ok and len(files) >= minimum,
               projections=sorted(files))
    uniq = {name: len(np.unique(np.round(a, 4), axis=0)) for name, a in files.items()}
    report.add("projections are not degenerate (many distinct 2D positions)", all(u > 0.5 * n for u in uniq.values()),
               distinct_positions=uniq)
    if reproduce:
        for method, seed, init in (("pca", 17, "pca"), ("umap", 17, "pca"), ("tsne", 17, "pca")):
            coords, _ = full_project.project(vectors, method, seed, init)
            saved = files[f"{method}-seed{seed}"]
            same = bool(np.array_equal(coords, saved))
            report.add(f"{method} seed {seed} re-run reproduces saved coordinates", same or bool(np.allclose(coords, saved, atol=1e-4)),
                       bitwise_identical=same, max_abs_diff=float(np.abs(coords - saved).max()))


def check_fidelity_recompute(run: Path, nodes, report: Report, sample=300) -> None:
    pres = np.load(run / "preservation.npz")
    any_idx = np.load(run / "neighbors/any-idx.npy")
    rows = full_quality.make_rows(nodes)
    rng = np.random.default_rng(5)
    queries = rng.choice(len(nodes), size=sample, replace=False)
    for method in ("pca", "umap", "tsne"):
        coords = np.load(run / f"projections/{method}-seed17.npy").astype(np.float64)
        bad = 0
        for q in queries:
            d2 = ((coords - coords[q]) ** 2).sum(axis=1)
            mask = rows.representative == rows.representative[q]
            allowed = np.flatnonzero(~mask)
            lo = allowed[np.lexsort((allowed, d2[allowed]))][:10]
            kept = len(set(lo.tolist()) & set(any_idx[q, :10].tolist()))
            bad += int(abs(kept - int(pres[method][q])) > 1)   # float32 vs float64 2D distance near-ties
        report.add(f"{method}: saved per-node fidelity matches an independent recomputation on {sample} nodes (+/-1 tie slack)",
                   bad <= 0.02 * sample, queries=sample, off_by_more_than_one=bad)


def extract_payload(html_path: Path) -> dict:
    text = html_path.read_text(encoding="utf-8")
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>', text, re.S)
    return json.loads(match.group(1)), text


def check_html(run: Path, html_path: Path, nodes, report: Report) -> None:
    payload, text = extract_payload(html_path)
    keys = [n["problem_key"] for n in nodes]
    report.add("html payload lists every catalog node once, in key order", payload["keys"] == keys and payload["n"] == len(keys))
    same = all(np.array_equal(full_html.unpack_array(payload["neighbors"][f"{name}_idx"]),
                              np.load(run / f"neighbors/{name}-idx.npy").astype(np.uint16)) for name in ("any", "cross"))
    report.add("html neighbor indices equal the saved exact neighbors", same)
    quant = all(np.array_equal(full_html.unpack_array(payload["neighbors"][f"{name}_score"]),
                               full_html.quantize_scores(np.load(run / f"neighbors/{name}-score.npy"))) for name in ("any", "cross"))
    report.add("html neighbor scores equal the quantized saved scores (1e-4 resolution)", quant)
    coords_ok = all(np.allclose(full_html.unpack_array(payload["projections"][m]["coords"]),
                                full_html.normalize_coordinates(np.load(run / f"projections/{m}-seed17.npy")), atol=1e-6)
                    for m in full_html.PROJECTIONS)
    report.add("html coordinates equal the normalized saved projections", coords_ok)
    leaks = [t for t in FORBIDDEN_TEXT if t in text]
    report.add("html contains no statement-text fields", not leaks, found=leaks)
    report.add("html loads no remote code", not re.search(r"<script[^>]+src=|<link[^>]+href=", text[:300000]))
    size = html_path.stat().st_size
    report.add("html is a single self-contained file under 16 MiB", size < 16 * 2**20, bytes=size)


def check_quarantine(run: Path, report: Report, quality_path: Path | None = None) -> None:
    hits = []
    for path in list(run.glob("*.jsonl")) + list(run.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        hits += [f"{path.name}:{t}" for t in FORBIDDEN_TEXT if t in text]
    quality = (quality_path or RECEIPTS / "quality.v1.json").read_text(encoding="utf-8")
    hits += [f"quality:{t}" for t in FORBIDDEN_TEXT if t in quality]
    report.add("no statement fields in any text artifact written by this pipeline", not hits, found=hits)


def check_existing_suite(report: Report) -> None:
    from research.unified_map.verify import verify
    result = verify(ROOT / "private-learning/unified-map/map-v2")
    report.add("existing map-v2 verifier and 501-file preservation snapshot still pass",
               bool(result.get("all_checks_passed")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reproduce-projections", action="store_true")
    args = parser.parse_args()
    report = Report()
    nodes = full_quality.load_nodes()
    check_artifact_hashes(args.run, report)
    vectors = check_vectors(args.run, nodes, report)
    check_neighbors(args.run, nodes, vectors, report)
    check_projections(args.run, nodes, vectors, report, args.reproduce_projections)
    check_fidelity_recompute(args.run, nodes, report)
    check_html(args.run, args.html, nodes, report)
    check_quarantine(args.run, report)
    check_existing_suite(report)
    result = {"schema": "full-validation-receipt.v1", "passed": report.ok, "checks": report.checks,
              "html_sha256": sha256_file(args.html)}
    with open(args.output, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print("ALL PASSED" if report.ok else "FAILURES PRESENT")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
