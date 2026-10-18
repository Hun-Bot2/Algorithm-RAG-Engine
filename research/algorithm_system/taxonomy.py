"""Versioned, provenance-preserving normalization for catalog-native tags."""

from __future__ import annotations

import json
import re
from pathlib import Path

TAXONOMY_PATH = Path(__file__).with_name("taxonomy.v1.json")
RELATIONS = {"equivalent", "narrower", "broader", "related", "unmapped"}


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_taxonomy(taxonomy: dict) -> list[str]:
    errors = []
    if taxonomy.get("schema") != "algorithm-taxonomy.v1":
        errors.append("unsupported_schema")
    concepts = taxonomy.get("concepts", [])
    ids = [c.get("id") for c in concepts]
    known = set(ids)
    if len(ids) != len(known) or None in known:
        errors.append("duplicate_or_missing_concept_id")
    graph = {c.get("id"): c.get("parents", []) for c in concepts}
    for cid, parents in graph.items():
        for parent in parents:
            if parent not in known:
                errors.append(f"unknown_parent:{cid}:{parent}")
    visiting, visited = set(), set()

    def visit(node):
        if node in visiting:
            errors.append(f"cycle:{node}")
            return
        if node in visited or node not in graph:
            return
        visiting.add(node)
        for parent in graph[node]:
            visit(parent)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)
    seen = set()
    for entry in taxonomy.get("tag_mappings", []):
        key = (entry.get("platform"), entry.get("tag"))
        if key in seen:
            errors.append(f"duplicate_tag_mapping:{key[0]}:{key[1]}")
        seen.add(key)
        if entry.get("relation") not in RELATIONS:
            errors.append(f"invalid_relation:{key[0]}:{key[1]}")
        if entry.get("relation") == "unmapped":
            if entry.get("concept_id") is not None:
                errors.append(f"unmapped_has_concept:{key[0]}:{key[1]}")
        elif entry.get("concept_id") not in known:
            errors.append(f"unknown_concept:{key[0]}:{key[1]}")
        confidence = entry.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"invalid_confidence:{key[0]}:{key[1]}")
        if not entry.get("rationale"):
            errors.append(f"missing_rationale:{key[0]}:{key[1]}")
    coverage = taxonomy.get("platform_coverage", {})
    for platform, record in coverage.items():
        expected = set(record.get("tag_document_frequency", {}))
        actual = {tag for p, tag in seen if p == platform}
        if expected != actual:
            errors.append(f"coverage_mismatch:{platform}")
        if record.get("native_tag_status") == "unavailable" and expected:
            errors.append(f"unavailable_platform_has_tags:{platform}")
    return errors


def _ancestor_ids(concept_id: str, concepts: dict[str, dict]) -> list[str]:
    result, seen, pending = [], set(), list(concepts[concept_id].get("parents", []))
    while pending:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        result.append(current)
        pending.extend(concepts[current].get("parents", []))
    return result


def normalize_tags(platform: str, rawtags, taxonomy: dict | None = None) -> dict:
    """Return raw tags, mapped concepts, separate ancestors, and missingness.

    ``None`` denotes unavailable source tags; an empty list denotes an observed
    empty tag list. Unknown native tags remain explicit and are never discarded.
    """
    tax = taxonomy or load_taxonomy()
    mappings = {(m["platform"], m["tag"]): m for m in tax["tag_mappings"]}
    concepts = {c["id"]: c for c in tax["concepts"]}
    if rawtags is None:
        return {"taxonomy_version": tax["version"], "original_tags": None,
                "mapped_concepts": [], "ancestors": [], "unknown_tags": [],
                "tag_status": "unavailable"}
    originals = list(rawtags)
    mapped, ancestors, unknown = [], [], []
    for tag in originals:
        entry = mappings.get((platform, tag))
        if entry is None:
            unknown.append(tag)
            continue
        if entry["relation"] == "unmapped":
            mapped.append({"tag": tag, "concept_id": None, "relation": "unmapped",
                           "confidence": entry["confidence"], "rationale": entry["rationale"]})
            continue
        mapped.append({"tag": tag, "concept_id": entry["concept_id"],
                       "relation": entry["relation"], "confidence": entry["confidence"],
                       "rationale": entry["rationale"]})
        # A related tag overlaps a concept; it does not entail that concept's
        # broader categories. Only equivalent/narrower source labels do.
        if entry["relation"] in {"equivalent", "narrower"}:
            ancestors.extend(_ancestor_ids(entry["concept_id"], concepts))
    return {"taxonomy_version": tax["version"], "original_tags": originals,
            "mapped_concepts": mapped, "ancestors": list(dict.fromkeys(ancestors)),
            "unknown_tags": unknown,
            "tag_status": "unknown_tags" if unknown else ("empty" if not originals else "present")}


def slug(value: str) -> str:
    value = value.casefold().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "tag"
