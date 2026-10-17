"""Build a provenance-preserving cross-platform unified problem map.

The map includes every native catalog row, but retrieval is limited to the
explicit cohorts for which the required evidence exists. Statement matches
are availability metadata only; statement vectors are never mixed with A1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.full_catalog.retrieve import Rows, dense_cosine, topk, write_results
from research.full_catalog.inputs import input_digest, load_catalog, select
from research.algorithm_system.taxonomy import load_taxonomy, normalize_tags

DEFAULT_CATALOG = ROOT / "private-learning/full-catalog/catalog-v1/catalog.v1.jsonl"
DEFAULT_EMBEDDING_DIR = ROOT / "private-learning/full-catalog/embeddings-v1/A1"
DEFAULT_STATEMENT_CORPUS = ROOT / "private-learning/algorithm-system/statement-run-v1/corpus.jsonl"
DEFAULT_APPROVAL = ROOT / "private-learning/verification-workbench/identity-mappings/user-approved-v1.json"
DEFAULT_APPROVAL_RECEIPT = ROOT / "research/identity_resolution/approval-receipt.v1.json"
DEFAULT_A1_RECEIPT = ROOT / "research/full_catalog/receipts/embeddings-A1.v1.json"
DEFAULT_A1_ALL_RECEIPT = ROOT / "research/full_catalog/receipts/baselines-A1-all-cross.v1.json"
DEFAULT_OUTPUT = ROOT / "private-learning/unified-map/map-v2"
KS = (10, 20, 50)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                          encoding="utf-8")


def load_statement_aliases(corpus_path: Path, approval_path: Path,
                           approval_receipt_path: Path) -> tuple[dict, dict]:
    """Validate the explicit approval chain and derive source/native aliases."""
    corpus = read_jsonl(corpus_path)
    receipt = json.loads(approval_receipt_path.read_text(encoding="utf-8"))
    approval_bytes_sha = sha256(approval_path)
    if receipt.get("mapping_sha256") != approval_bytes_sha:
        raise ValueError("approved_mapping_hash_mismatch")
    if receipt.get("mapping_path") != str(approval_path.relative_to(ROOT)):
        raise ValueError("approved_mapping_path_mismatch")
    if receipt.get("source_hashes_unchanged", {}).get(str(corpus_path)) != sha256(corpus_path):
        raise ValueError("statement_corpus_hash_mismatch")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("evidence_sha256") != receipt.get("evidence_sha256"):
        raise ValueError("approval_evidence_hash_mismatch")
    mappings = approval.get("mappings", {})
    if len(mappings) != 36 or receipt.get("approved") != 36:
        raise ValueError("approved_mapping_count_mismatch")
    if receipt.get("unresolved_publisher_id") != "p03510" or receipt.get("unresolved") != 1:
        raise ValueError("unresolved_identity_receipt_mismatch")
    by_source, by_native, mapping_by_native = {}, {}, {}
    for row in corpus:
        source_key = row.get("dataset_key") or row.get("problem_key")
        native = row.get("native_key")
        if source_key:
            by_source.setdefault(source_key, []).append(row)
        if native:
            by_native.setdefault(native, []).append(row)
    unresolved_keys = [key for key, value in mappings.items()
                       if value.get("publisher_title", "").startswith("p03510")]
    if unresolved_keys:
        raise ValueError("unresolved_p03510_must_not_be_approved")
    for source_key, mapping in mappings.items():
        matching = by_source.get(source_key, [])
        if len(matching) != 1:
            raise ValueError(f"approved_source_key_not_unique:{source_key}")
        if matching[0].get("content_sha256") != mapping.get("source_content_sha256"):
            raise ValueError(f"approved_source_content_hash_mismatch:{source_key}")
        if mapping.get("status") != "user_approved" or not mapping.get("native_key"):
            raise ValueError(f"invalid_approved_mapping:{source_key}")
        by_native.setdefault(mapping["native_key"], []).append(matching[0])
        mapping_by_native.setdefault(mapping["native_key"], []).append(mapping)
    return {"by_native": by_native, "by_source": by_source,
            "mapping_by_native": mapping_by_native}, {
        "statement_corpus_sha256": sha256(corpus_path),
        "approved_mapping_sha256": approval_bytes_sha,
        "approval_receipt_sha256": sha256(approval_receipt_path),
        "statement_corpus_rows": len(corpus), "approved_mapping_count": len(mappings),
        "unresolved_p03510_preserved": True,
        "abc084_c_quality_flags": receipt.get("abc084_c_quality_flags", []),
    }


def _concepts(normalized: dict) -> set[str]:
    """Only equivalent, high-confidence direct labels are comparable."""
    return {entry["concept_id"] for entry in normalized["mapped_concepts"]
            if entry.get("concept_id") and entry.get("relation") == "equivalent"
            and entry.get("confidence", 0) >= .85}


def tag_features(rows: list[dict], taxonomy: dict) -> list[set[str] | None]:
    output = []
    for row in rows:
        raw = row.get("tags")
        if raw is None:
            output.append(None)
        else:
            output.append(_concepts(normalize_tags(row["platform"], raw, taxonomy)))
    return output


def jaccard_block(features: list[set[str] | None]):
    from scipy.sparse import csr_matrix
    vocabulary = sorted({concept for values in features if values is not None for concept in values})
    columns = {concept: index for index, concept in enumerate(vocabulary)}
    data, row_ids, col_ids = [], [], []
    for i, values in enumerate(features):
        for concept in values or ():
            row_ids.append(i)
            col_ids.append(columns[concept])
            data.append(1.0)
    matrix = csr_matrix((np.asarray(data, dtype=np.float32), (row_ids, col_ids)),
                        shape=(len(features), len(vocabulary)), dtype=np.float32)
    sizes = np.asarray(matrix.sum(axis=1)).ravel()
    def score(start: int, stop: int):
        intersection = (matrix[start:stop] @ matrix.T).toarray()
        union = sizes[start:stop, None] + sizes[None, :] - intersection
        block = np.zeros_like(intersection, dtype=np.float32)
        np.divide(intersection, union, out=block, where=union > 0)
        return block
    return score


def _statement_records(aliases: dict, key: str) -> list[dict]:
    return aliases["by_native"].get(key, [])


def enrich_catalog(catalog: list[dict], embedding_keys: list[str], aliases: dict,
                   taxonomy: dict) -> list[dict]:
    a1 = set(embedding_keys)
    output = []
    for source in catalog:
        row = dict(source)
        key = row["problem_key"]
        row["source_tags"] = row.get("tags")
        row["a1_embedding_available"] = key in a1
        row["normalized_taxonomy"] = normalize_tags(row["platform"], row.get("tags"), taxonomy)
        records = _statement_records(aliases, key)
        if records:
            overlays = aliases.get("mapping_by_native", {}).get(key, [])
            row["statement_availability"] = {
                "status": "publisher_statement_available",
                "source_keys": sorted({r.get("dataset_key") or r.get("problem_key") for r in records}),
                "content_sha256": sorted({r["content_sha256"] for r in records if r.get("content_sha256")}),
                "identity_basis": sorted({r.get("identity_basis", "native_key_alias") for r in records}),
                "quality_flags": sorted({flag for r in records for flag in r.get("quality_flags", [])}
                                         | {flag for m in overlays for flag in m.get("quality_flags", [])}),
                "identity_overlay": "user_approved_hash_validated" if overlays else "source_corpus_native_alias",
            }
        else:
            row["statement_availability"] = {"status": "not_available_in_admitted_corpus",
                                                "source_keys": [], "content_sha256": [],
                                                "quality_flags": []}
        output.append(row)
    return output


def _dump_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    path.chmod(0o600)
    return sha256(path)


def _subset_rows(catalog_by_key: dict, keys: list[str]) -> Rows:
    return Rows.from_catalog([catalog_by_key[k] for k in keys], keys)


def _run_rankings(rows: Rows, score, folder: Path, block_size: int = 128):
    folder.parent.mkdir(parents=True, exist_ok=True)
    hashes, counts = write_results(rows, topk(rows, score, 50, "cross", block_size=block_size), folder)
    return {"outputs": hashes, **counts}


def _rrf(rows: Rows, semantic, tags, folder: Path, block_size: int = 128, rrf_k: int = 60):
    """Equal-weight reciprocal rank fusion over the union of exact top-50 lists."""
    folder.parent.mkdir(parents=True, exist_ok=True)
    sem_rank, tag_rank, sem_score, tag_score = {}, {}, {}, {}
    for index, ranked in topk(rows, semantic, 50, "cross", block_size=block_size):
        sem_rank[index] = {target: rank for rank, (target, _) in enumerate(ranked, 1)}
        sem_score[index] = {target: score for target, score in ranked}
    for index, ranked in topk(rows, tags, 50, "cross", block_size=block_size):
        positive = [(target, score) for target, score in ranked if score > 0]
        tag_rank[index] = {target: rank for rank, (target, _) in enumerate(positive, 1)}
        tag_score[index] = {target: score for target, score in positive}
    fused = []
    sidecar = []
    for query in range(len(rows)):
        candidates = set(sem_rank[query]) | set(tag_rank[query])
        scored = []
        for target in candidates:
            sr, tr = sem_rank[query].get(target), tag_rank[query].get(target)
            value = (1.0 / (rrf_k + sr) if sr else 0.0) + (1.0 / (rrf_k + tr) if tr else 0.0)
            scored.append((target, value, sr, tr))
        scored.sort(key=lambda item: (-item[1], item[0]))
        fused.append((query, [(target, score) for target, score, _, _ in scored[:50]]))
        sidecar.extend({"query": rows.keys[query], "target": rows.keys[target],
                        "rrf_score": round(score, 9), "semantic_rank": sr, "tag_rank": tr,
                        "semantic_cosine": round(sem_score[query][target], 6)
                            if target in sem_score[query] else None,
                        "tag_jaccard": round(tag_score[query][target], 6)
                            if target in tag_score[query] else None,
                        "tag_supported": target in tag_score[query]}
                       for target, score, sr, tr in scored[:50])
    hashes, counts = write_results(rows, iter(fused), folder)
    sidecar_path = folder / "score-components.jsonl"
    with sidecar_path.open("x", encoding="utf-8") as handle:
        for item in sidecar:
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")
    sidecar_path.chmod(0o600)
    return {"outputs": hashes, "score_components_sha256": sha256(sidecar_path), **counts,
            "fusion": {"method": "reciprocal_rank_fusion", "k": rrf_k,
                       "weights": {"semantic": 1, "taxonomy": 1},
                       "candidate_pool": "union of semantic top50 and positive-Jaccard taxonomy top50; fusion is bounded, not exhaustive",
                       "zero_jaccard_candidates_receive_no_taxonomy_rank_or_weight": True}}


def build(catalog_path: Path = DEFAULT_CATALOG,
          embedding_dir: Path = DEFAULT_EMBEDDING_DIR,
          statement_corpus_path: Path = DEFAULT_STATEMENT_CORPUS,
          approval_path: Path = DEFAULT_APPROVAL,
          approval_receipt_path: Path = DEFAULT_APPROVAL_RECEIPT,
          output: Path = DEFAULT_OUTPUT, block_size: int = 128,
          a1_receipt_path: Path = DEFAULT_A1_RECEIPT,
          a1_all_receipt_path: Path = DEFAULT_A1_ALL_RECEIPT) -> dict:
    if output.exists():
        raise ValueError("fresh_immutable_output_required")
    started = time.perf_counter()
    catalog = load_catalog(catalog_path)
    keys = json.loads((embedding_dir / "keys.json").read_text(encoding="utf-8"))
    vectors_path = embedding_dir / "vectors.npy"
    a1_receipt = json.loads(a1_receipt_path.read_text(encoding="utf-8"))
    a1_rows, a1_texts = select(catalog, "A1")
    if a1_receipt.get("inputs", {}).get("catalog") != sha256(catalog_path):
        raise ValueError("a1_catalog_receipt_mismatch")
    if a1_receipt.get("inputs", {}).get("variant_input") != input_digest(a1_rows, a1_texts):
        raise ValueError("a1_variant_input_receipt_mismatch")
    if [row["problem_key"] for row in a1_rows] != keys:
        raise ValueError("a1_keys_do_not_match_catalog_selection")
    if a1_receipt.get("variant") != "A1" or a1_receipt.get("outputs", {}).get("keys.json") != sha256(embedding_dir / "keys.json"):
        raise ValueError("a1_keys_receipt_mismatch")
    if a1_receipt.get("outputs", {}).get("vectors.npy") != sha256(vectors_path):
        raise ValueError("a1_vectors_receipt_mismatch")
    if a1_receipt.get("rows") != len(keys):
        raise ValueError("a1_receipt_row_count_mismatch")
    a1_all_receipt = json.loads(a1_all_receipt_path.read_text(encoding="utf-8"))
    a1_all_top50 = ROOT / "private-learning/full-catalog/baselines-v1/A1-all/tfidf_char/top50.jsonl"
    expected_a1_all_hash = a1_all_receipt.get("methods", {}).get("tfidf_char", {}).get("outputs", {}).get("top50.jsonl")
    if not expected_a1_all_hash or sha256(a1_all_top50) != expected_a1_all_hash:
        raise ValueError("a1_all_lexical_artifact_hash_mismatch")
    vectors = np.load(vectors_path, allow_pickle=False, mmap_mode="r")
    if len(keys) != len(vectors) or keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("a1_key_vector_alignment_invalid")
    if not set(keys) <= {r["problem_key"] for r in catalog if r["eligibility"] == "eligible"}:
        raise ValueError("a1_keys_outside_eligible_catalog")
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("invalid_a1_vectors")
    aliases, statement_receipt = load_statement_aliases(statement_corpus_path, approval_path,
                                                         approval_receipt_path)
    taxonomy = load_taxonomy()
    output.mkdir(parents=True, exist_ok=False)
    (output / "implementation.py").write_bytes(Path(__file__).read_bytes())
    nodes = enrich_catalog(catalog, keys, aliases, taxonomy)
    node_digest = _dump_jsonl(output / "nodes.jsonl", nodes)
    by_key = {r["problem_key"]: r for r in catalog}
    semantic_rows = _subset_rows(by_key, keys)
    semantic_metrics = _run_rankings(semantic_rows, dense_cosine(vectors), output / "semantic", block_size)

    candidates_with_tags = [r for r in catalog if r["eligibility"] == "eligible"
                           and r["platform"] in {"leetcode", "codeforces"} and r.get("tags")]
    tagged_keys = [row["problem_key"] for row, concepts in
                   zip(candidates_with_tags, tag_features(candidates_with_tags, taxonomy))
                   if concepts]
    tagged_catalog = [by_key[k] for k in tagged_keys]
    tag_rows = _subset_rows(by_key, tagged_keys)
    tag_feats = tag_features(tagged_catalog, taxonomy)
    tag_metrics = _run_rankings(tag_rows, jaccard_block(tag_feats), output / "diagnostics/tag_only", block_size)

    matched_keys = [k for k in tagged_keys if k in set(keys)]
    matched_catalog = [by_key[k] for k in matched_keys]
    key_index = {key: i for i, key in enumerate(keys)}
    matched_vectors = np.asarray(vectors[[key_index[k] for k in matched_keys]], dtype=np.float32)
    matched_rows = _subset_rows(by_key, matched_keys)
    matched_tags = tag_features(matched_catalog, taxonomy)
    semantic_matched = dense_cosine(matched_vectors)
    tags_matched = jaccard_block(matched_tags)
    matched_semantic_metrics = _run_rankings(matched_rows, semantic_matched,
                                             output / "matched_semantic", block_size)
    matched_tag_metrics = _run_rankings(matched_rows, tags_matched, output / "tag", block_size)
    hybrid_metrics = _rrf(matched_rows, semantic_matched, tags_matched,
                          output / "hybrid", block_size)

    # Explicitly summarize every coverage state instead of treating missing evidence as zero.
    counts = Counter((r["platform"], r["eligibility"], r["a1_embedding_available"],
                      r["normalized_taxonomy"]["tag_status"], r["statement_availability"]["status"])
                     for r in nodes)
    receipt = {
        "schema": "unified-map-receipt.v1", "catalog_sha256": sha256(catalog_path),
        "a1_keys_sha256": sha256(embedding_dir / "keys.json"),
        "a1_variant_input_sha256": input_digest(a1_rows, a1_texts),
        "a1_vectors_sha256": sha256(vectors_path), "taxonomy_sha256": sha256(ROOT / "research/algorithm_system/taxonomy.v1.json"),
        "a1_embedding_receipt_sha256": sha256(a1_receipt_path),
        **statement_receipt, "nodes_sha256": node_digest,
        "counts": {"catalog_nodes": len(nodes), "eligible": sum(r["eligibility"] == "eligible" for r in nodes),
                   "a1_embedding_rows": len(keys), "tag_only_rows": len(tagged_keys),
                   "matched_tagged_a1_rows": len(matched_keys), "coverage_cells": [
                       {"platform": p, "eligibility": e, "a1_embedding_available": a1,
                        "tag_status": t, "statement_status": s, "rows": n}
                       for (p, e, a1, t, s), n in sorted(counts.items())]},
        "methods": {"semantic_a1_cross": semantic_metrics, "tag_only_full_tagged_cross": tag_metrics,
                    "matched_semantic_cross": matched_semantic_metrics,
                    "matched_tag_cross": matched_tag_metrics,
                    "hybrid_cross": hybrid_metrics},
        "algorithm_similarity": {"tag_metric": "Jaccard over equivalent normalized concepts only",
                                 "minimum_mapping_confidence": .85,
                                 "missing_or_unmapped_tag_behavior": "unknown_or_empty remains explicit; no imputation",
                                 "vector_spaces_averaged": False,
                                 "hybrid": "equal-weight reciprocal rank fusion, k=60, on semantic top50 union positive-Jaccard taxonomy top50; zero-overlap ties contribute no taxonomy rank",
                                 "tag_only_diagnostic": "exact top50 includes deterministic zero-Jaccard ties when fewer than 50 positive-overlap targets exist; score=0 means no evidence",
                                 "statement_retrieval": "not run; source availability only"},
        "a1_all_lexical_reference": {
            "receipt_path": str(a1_all_receipt_path.relative_to(ROOT)),
            "receipt_sha256": sha256(a1_all_receipt_path),
            "rows": a1_all_receipt.get("rows"),
            "methods": {"tfidf_char": {
                "top50_path": "private-learning/full-catalog/baselines-v1/A1-all/tfidf_char/top50.jsonl",
                "top50_sha256": expected_a1_all_hash,
                "actual_top50_sha256": sha256(ROOT / "private-learning/full-catalog/baselines-v1/A1-all/tfidf_char/top50.jsonl")}},
            "use": "frozen lexical diagnostic for encoder-flagged rows; no MiniLM vectors or new retrieval generated"},
        "duration_seconds": round(time.perf_counter() - started, 3),
        "output_sha256": {str(p.relative_to(output)): sha256(p)
                          for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"},
        "evaluation_corpus_required": "A runnable labeled corpus with problem_key/platform/native_difficulty/source_tags; current statement evaluation corpus does not cover all native catalog nodes.",
    }
    write_json(output / "manifest.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--statement-corpus", type=Path, default=DEFAULT_STATEMENT_CORPUS)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--approval-receipt", type=Path, default=DEFAULT_APPROVAL_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--a1-receipt", type=Path, default=DEFAULT_A1_RECEIPT)
    parser.add_argument("--a1-all-receipt", type=Path, default=DEFAULT_A1_ALL_RECEIPT)
    args = parser.parse_args()
    receipt = build(args.catalog, args.embedding_dir, args.statement_corpus, args.approval,
                    args.approval_receipt, args.output, args.block_size,
                    args.a1_receipt, args.a1_all_receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
