"""Pack the full-catalog map into one self-contained, offline HTML file.

The output embeds private problem metadata (titles, tags, URLs), so it is written under
private-learning/ only. It reads no statement text: only the statement-availability
status flag of each node. The page template (full_map_template.html) holds no data.

    .venv/bin/python -m research.unified_map.full_html \
        --run private-learning/unified-map/full-v1 \
        --quality research/unified_map/full-receipts/quality.v1.json \
        --output private-learning/unified-map/full-v1/interactive.html
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from research.unified_map.build import _concepts

ROOT = Path(__file__).resolve().parents[2]
NODES = ROOT / "private-learning/unified-map/map-v2/nodes.jsonl"
LABELS = ROOT / "research/full_catalog/labels.v1.jsonl"
HISTORY = ROOT / "research/relationship-judgments.v1.json"
ALIASES = ROOT / "assets/problem_identity_aliases.v1.json"
TEMPLATE = Path(__file__).with_name("full_map_template.html")
PLATFORMS = ("atcoder", "codeforces", "leetcode")
TIERS = ("reliable", "degraded", "unreliable")
RELATIONS = ("equivalent", "narrower", "broader", "related", "unmapped")
PROJECTIONS = ("pca", "umap", "tsne")
HISTORY_TYPES = ("similarity", "prerequisite", "transfer")
DTYPES = {"u8": np.uint8, "i16": np.int16, "u16": np.uint16, "u32": np.uint32, "i32": np.int32,
          "f32": np.float32}
SCORE_SCALE = 10000
MISSING = 255
PLACEHOLDER = "__PAYLOAD_JSON__"


def pack_array(array, dtype: str) -> dict:
    """Little-endian base64 array record that the page decodes into a typed array."""
    data = np.ascontiguousarray(np.asarray(array).astype(DTYPES[dtype]).astype(
        np.dtype(DTYPES[dtype]).newbyteorder("<"), copy=False))
    return {"dtype": dtype, "shape": list(data.shape), "b64": base64.b64encode(data.tobytes()).decode("ascii")}


def unpack_array(record: dict) -> np.ndarray:
    dtype = np.dtype(DTYPES[record["dtype"]]).newbyteorder("<")
    return np.frombuffer(base64.b64decode(record["b64"]), dtype=dtype).reshape(record["shape"])


def quantize_scores(scores: np.ndarray) -> np.ndarray:
    """Cosine in [-1, 1] -> int16 at 1e-4 resolution; NaN is rejected."""
    if not np.isfinite(scores).all():
        raise ValueError("non_finite_scores")
    return np.rint(np.clip(scores, -1.0, 1.0) * SCORE_SCALE).astype(np.int16)


def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] on each axis independently by min/max (shape-preserving per axis)."""
    coords = np.asarray(coords, dtype=np.float64)
    low, high = coords.min(axis=0), coords.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    return ((coords - low) / span).astype(np.float32)


_CF_KEY = re.compile(r"codeforces:(\d+)([A-Za-z0-9]+)")


def derive_url(node: dict) -> str | None:
    key, platform = node["problem_key"], node["platform"]
    if platform == "leetcode":
        return f"https://leetcode.com/problems/{key.split(':', 1)[1]}/"
    if platform == "codeforces":
        match = _CF_KEY.fullmatch(key)
        return f"https://codeforces.com/problemset/problem/{match.group(1)}/{match.group(2)}" if match else None
    if node.get("contest"):
        return f"https://atcoder.jp/contests/{node['contest']}/tasks/{key.split(':', 1)[1]}"
    return None


def title_groups(nodes: list[dict]) -> np.ndarray:
    """Group id for nodes sharing a normalized title (any platform); -1 when unique."""
    members = collections.defaultdict(list)
    for index, node in enumerate(nodes):
        members[node["title_normalized"]].append(index)
    ids = np.full(len(nodes), -1, dtype=np.int32)
    for group, indices in enumerate(sorted((v for v in members.values() if len(v) > 1), key=lambda v: v[0])):
        ids[indices] = group
    return ids


def duplicate_groups(nodes: list[dict]) -> tuple[np.ndarray, list[list[int]]]:
    members = collections.defaultdict(list)
    for index, node in enumerate(nodes):
        if node["duplicate_candidate_group"]:
            members[node["duplicate_candidate_group"]].append(index)
    ordered = sorted(members.items())
    ids = np.full(len(nodes), -1, dtype=np.int32)
    for group, (_, indices) in enumerate(ordered):
        ids[indices] = group
    return ids, [indices for _, indices in ordered]


def judged_pairs(labels: list[dict], index: dict) -> list[list[int]]:
    """[query, target, label|255, deceptive|255] for frozen AI-reviewed pair records."""
    out = []
    for row in labels:
        label = MISSING if row["label"] is None else int(row["label"])
        deceptive = MISSING if row["deceptive"] is None else int(bool(row["deceptive"]))
        out.append([index[row["query"]], index[row["target"]], label, deceptive])
    return out


def history_pairs(history: dict, aliases: dict, index: dict) -> tuple[list[list], int]:
    """Historical relationship judgments; legacy keys resolve through the reviewed alias file."""
    rows, unresolved = [], 0
    for item in history["judgments"]:
        source = aliases.get(item["source"], item["source"])
        target = aliases.get(item["target"], item["target"])
        if source not in index or target not in index:
            unresolved += 1
            continue
        rows.append([index[source], index[target], HISTORY_TYPES.index(item["type"]), int(item["relevance"]),
                     item["evidence"], int(source != item["source"] or target != item["target"])])
    return rows, unresolved


def build_payload(nodes, meta, any_idx, any_score, cross_idx, cross_score, coords, preservation,
                  quality, labels, history, aliases, run_info, extras: dict | None = None) -> dict:
    n = len(nodes)
    index = {node["problem_key"]: i for i, node in enumerate(nodes)}
    if [m["problem_key"] for m in meta] != [node["problem_key"] for node in nodes]:
        raise ValueError("meta_order_mismatch")
    tag_vocab = sorted({t for node in nodes for t in (node["tags"] or [])})
    tag_index = {t: i for i, t in enumerate(tag_vocab)}
    offsets, tag_ids = [0], []
    for node in nodes:
        tag_ids.extend(tag_index[t] for t in node["tags"] or [])
        offsets.append(len(tag_ids))
    concept_vocab = sorted({e["concept_id"] for node in nodes for e in node["normalized_taxonomy"]["mapped_concepts"]
                            if e.get("concept_id")})
    concept_index = {c: i for i, c in enumerate(concept_vocab)}
    mapped = [[[concept_index[e["concept_id"]], RELATIONS.index(e["relation"]), int(round(e["confidence"] * 100))]
               for e in node["normalized_taxonomy"]["mapped_concepts"] if e.get("concept_id")] for node in nodes]
    reasons = sorted({r for node in nodes for r in node["exclusion_reasons"]})
    excluded = {i: [reasons.index(r) for r in node["exclusion_reasons"]]
                for i, node in enumerate(nodes) if node["exclusion_reasons"]}
    url_exceptions = {i: node["url"] for i, node in enumerate(nodes) if node["url"] != derive_url(node)}
    dup_ids, dup_members = duplicate_groups(nodes)
    judgment_rows, unresolved = history_pairs(history, aliases, index)
    top1 = any_score[:, 0]
    tenth = any_score[:, 9]
    payload = {
        "schema": "full-map-payload.v1",
        "run": run_info,
        "n": n,
        "platforms": list(PLATFORMS),
        "keys": [node["problem_key"] for node in nodes],
        "titles": [node["title"] for node in nodes],
        "platform": pack_array([PLATFORMS.index(node["platform"]) for node in nodes], "u8"),
        "eligible": pack_array([node["eligibility"] == "eligible" for node in nodes], "u8"),
        "excluded_reasons": {"vocab": reasons, "by_node": {str(k): v for k, v in excluded.items()}},
        "reliability": pack_array([TIERS.index(m["reliability"]) for m in meta], "u8"),
        "unk_percent": pack_array([int(round(m["unk_share"] * 100)) for m in meta], "u8"),
        "vector_source": pack_array([m["vector_source"] != "a1_reused" for m in meta], "u8"),
        "difficulty": [node["native_difficulty"] for node in nodes],
        "contest": [node["contest"] if node["platform"] != "leetcode" else None for node in nodes],
        "url_exceptions": {str(k): v for k, v in url_exceptions.items()},
        "tags": {"vocab": tag_vocab, "offsets": pack_array(offsets, "u32"), "ids": pack_array(tag_ids, "u16")},
        "concepts": {"vocab": concept_vocab, "relations": list(RELATIONS), "mapped": mapped},
        "statement_available": pack_array([node["statement_availability"]["status"] == "publisher_statement_available"
                                           for node in nodes], "u8"),
        "title_group": pack_array(title_groups(nodes), "i32"),
        "duplicate_group": pack_array(dup_ids, "i32"),
        "duplicate_members": dup_members,
        "neighbors": {"k": int(any_idx.shape[1]), "score_scale": SCORE_SCALE,
                      "any_idx": pack_array(any_idx, "u16"), "any_score": pack_array(quantize_scores(any_score), "i16"),
                      "cross_idx": pack_array(cross_idx, "u16"),
                      "cross_score": pack_array(quantize_scores(cross_score), "i16")},
        "projections": {name: {"coords": pack_array(normalize_coordinates(coords[name]), "f32"),
                               "preserved_top10": pack_array(preservation[name], "u8")} for name in PROJECTIONS},
        "quality": quality,
        "judged_pairs": judged_pairs(labels, index),
        "history": judgment_rows,
        "history_unresolved": unresolved,
        "score_context": {"top1_any_median": float(np.median(top1)), "top1_any_p10": float(np.percentile(top1, 10)),
                          "top1_any_p90": float(np.percentile(top1, 90)),
                          "rank10_any_median": float(np.median(tenth))},
    }
    clash = set(extras or {}) & set(payload)
    if clash:
        raise ValueError(f"extras would overwrite payload fields: {sorted(clash)}")
    payload.update(extras or {})
    return payload


def render_html(payload: dict, template: str) -> str:
    if template.count(PLACEHOLDER) != 1:
        raise ValueError("template_placeholder_count")
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    text = text.replace("</", "<\\/").replace("<!--", "<\\!--").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return template.replace(PLACEHOLDER, text)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    nodes = sorted(_read_jsonl(NODES), key=lambda n: n["problem_key"])
    meta = _read_jsonl(run / "vector_meta.jsonl")
    neighbors = run / "neighbors"
    coords = {name: np.load(run / "projections" / f"{name}-seed17.npy") for name in PROJECTIONS}
    pres = np.load(run / "preservation.npz")
    aliases = {a["legacy_key"]: a["native_key"] for a in json.loads(ALIASES.read_text(encoding="utf-8"))["aliases"]}
    quality = json.loads(args.quality.read_text(encoding="utf-8"))
    run_info = {"vectors_sha256": hashlib.sha256((run / "vectors.npy").read_bytes()).hexdigest(),
                "quality_receipt_sha256": hashlib.sha256(args.quality.read_bytes()).hexdigest(),
                "model": "sentence-transformers/all-MiniLM-L6-v2 @ 1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                "input": "problem title only"}
    payload = build_payload(
        nodes, meta, np.load(neighbors / "any-idx.npy"), np.load(neighbors / "any-score.npy"),
        np.load(neighbors / "cross-idx.npy"), np.load(neighbors / "cross-score.npy"), coords,
        {name: pres[name] for name in PROJECTIONS}, quality, _read_jsonl(LABELS),
        json.loads(HISTORY.read_text(encoding="utf-8")), aliases, run_info)
    html = render_html(payload, TEMPLATE.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        handle.write(html)


if __name__ == "__main__":
    main()
