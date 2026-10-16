"""Phase 5 baselines on the exact rows/texts of a variant: TF-IDF word, TF-IDF char,
tag-Jaccard via the CF<->LC crosswalk (A2 only) and a seeded random ranking.

All methods reuse retrieve.topk(), so masking, tie-breaking and output files are
identical to the embedding runs.

    .venv/bin/python research/full_catalog/baselines.py \
        --catalog private-learning/full-catalog/catalog-v1/catalog.v1.jsonl \
        --variant A1 --mode cross \
        --output private-learning/full-catalog/<new-run> \
        --receipt research/full_catalog/receipts/<new>.json
"""

import argparse
import hashlib
import json
import platform as host_platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

import inputs
import retrieve

HERE = Path(__file__).resolve().parent
CROSSWALK = HERE / "tag_crosswalk.v1.json"
RANDOM_SEED = 20261016
CONCEPT_RELATIONS = ("equivalent", "narrower")

TFIDF_PARAMS = {
    "tfidf_word": dict(lowercase=True, analyzer="word", token_pattern=r"(?u)\b\w+\b",
                       ngram_range=(1, 2), sublinear_tf=True, norm="l2", dtype=np.float32),
    "tfidf_char": dict(lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
                       sublinear_tf=True, norm="l2", dtype=np.float32),
}
METHODS_BY_VARIANT = {
    "A1": ("tfidf_word", "tfidf_char", "random"),
    "A1-all": ("tfidf_char", "random"),
    "A2": ("tfidf_word", "tfidf_char", "tag_jaccard", "random"),
}


def tfidf_matrix(texts, method):
    """Fit on the variant's own rows (transductive, deterministic) and return CSR rows."""
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS[method])
    matrix = vectorizer.fit_transform(texts).tocsr()
    return matrix, {"vocabulary": len(vectorizer.vocabulary_),
                    "zero_rows": int((matrix.getnnz(axis=1) == 0).sum())}


def sparse_cosine(matrix):
    transposed = matrix.T.tocsc()

    def block(start, stop):
        return (matrix[start:stop] @ transposed).toarray().astype(np.float32)

    return block


def concept_sets(rows, crosswalk):
    """Concepts in the Codeforces tag namespace; LeetCode tags map via equivalent/narrower."""
    lc_to_cf = {}
    for entry in crosswalk["cf_to_lc"]:
        for mapping in entry["lc"]:
            if mapping["relation"] in CONCEPT_RELATIONS:
                lc_to_cf.setdefault(mapping["tag"], set()).add(entry["cf"])
    concepts = []
    for row in rows:
        tags = [t for t in row["tags"] or [] if t not in inputs.IGNORED_TAGS]
        if row["platform"] == "codeforces":
            concepts.append(set(tags))
        elif row["platform"] == "leetcode":
            concepts.append(set().union(*(lc_to_cf.get(t, set()) for t in tags)))
        else:
            raise ValueError("tag_jaccard_requires_leetcode_or_codeforces")
    return concepts


def jaccard_block(concepts):
    vocab = {c: i for i, c in enumerate(sorted(set().union(*concepts)))}
    data_rows, data_cols = [], []
    for r, items in enumerate(concepts):
        for c in items:
            data_rows.append(r)
            data_cols.append(vocab[c])
    binary = sparse.csr_matrix((np.ones(len(data_rows), dtype=np.float32), (data_rows, data_cols)),
                               shape=(len(concepts), max(len(vocab), 1)))
    sizes = np.asarray(binary.sum(axis=1)).ravel()
    transposed = binary.T.tocsc()

    def block(start, stop):
        inter = (binary[start:stop] @ transposed).toarray()
        union = sizes[start:stop, None] + sizes[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(union > 0, inter / union, 0.0).astype(np.float32)

    return block, {"concepts": len(vocab), "empty_concept_rows": int((sizes == 0).sum())}


def random_block(n, seed=RANDOM_SEED):
    """Per-query seeded uniform scores; independent of block size."""
    def block(start, stop):
        return np.stack([np.random.default_rng([seed, q]).random(n, dtype=np.float32)
                         for q in range(start, stop)])

    return block


def method_block(method, rows, texts, crosswalk):
    if method in TFIDF_PARAMS:
        matrix, info = tfidf_matrix(texts, method)
        return sparse_cosine(matrix), info
    if method == "tag_jaccard":
        return jaccard_block(concept_sets(rows, crosswalk))
    if method == "random":
        return random_block(len(rows)), {"seed": RANDOM_SEED}
    raise ValueError(f"unknown_method:{method}")


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(params):
    return {k: (v.__name__ if isinstance(v, type) else list(v) if isinstance(v, tuple) else v)
            for k, v in params.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--variant", choices=inputs.VARIANTS, required=True)
    parser.add_argument("--mode", choices=("cross", "same"), default="cross")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True, help="new private directory")
    parser.add_argument("--receipt", type=Path, required=True, help="new aggregate receipt path")
    args = parser.parse_args()

    catalog_rows = inputs.load_catalog(args.catalog)
    rows, texts = inputs.select(catalog_rows, args.variant)
    universe = retrieve.Rows.from_catalog(catalog_rows, [r["problem_key"] for r in rows])
    crosswalk = json.loads(CROSSWALK.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)

    methods = {}
    for method in METHODS_BY_VARIANT[args.variant]:
        started = time.perf_counter()
        block, info = method_block(method, rows, texts, crosswalk)
        results = retrieve.topk(universe, block, max(retrieve.KS), args.mode, block_size=args.block_size)
        outputs, counts = retrieve.write_results(universe, results, args.output / method)
        methods[method] = {"outputs": outputs, "counts": counts, "info": info,
                           "seconds": round(time.perf_counter() - started, 3)}
        if method in TFIDF_PARAMS:
            methods[method]["params"] = _jsonable(TFIDF_PARAMS[method])
        if method == "tag_jaccard":
            methods[method]["params"] = {"crosswalk_relations": list(CONCEPT_RELATIONS)}

    per_platform = {}
    for row in rows:
        per_platform[row["platform"]] = per_platform.get(row["platform"], 0) + 1
    receipt = {
        "schema": "baseline-receipt.v1",
        "variant": args.variant,
        "mode": args.mode,
        "block_size": args.block_size,
        "rows": len(rows),
        "rows_by_platform": dict(sorted(per_platform.items())),
        "inputs": {"catalog": _sha256_file(args.catalog), "variant_input": inputs.input_digest(rows, texts),
                   "crosswalk": _sha256_file(CROSSWALK)},
        "methods": methods,
        "environment": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                        "python": host_platform.python_version(), "machine": host_platform.machine()},
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with open(args.receipt, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
