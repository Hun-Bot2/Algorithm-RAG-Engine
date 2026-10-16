"""Allowlisted profile construction and deterministic candidate scorers."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import math
import re
from pathlib import Path
from .research_contracts import NamespaceMismatchError, ResearchError, _clean_text, _tokens, json_bytes, sha256_bytes, read_json

@dataclass(frozen=True)
class Profile:
    problem_key: str
    platform: str
    native_difficulty: str
    concepts: tuple[str, ...]
    title: str
    concept_rationale: str

    @property
    def title_text(self) -> str:
        return self.title

    @property
    def rationale_text(self) -> str:
        return self.concept_rationale

    @property
    def concept_text(self) -> str:
        return " ".join(self.concepts)

    @property
    def text(self) -> str:
        return " ".join(part for part in (self.title, self.concept_rationale, self.concept_text) if part)

def build_profiles(curriculum: dict, annotations: dict) -> dict[str, Profile]:
    """Build profiles from the fixed, text-minimal input whitelist."""
    if curriculum.get("schema_version") != 1 or not isinstance(curriculum.get("problems"), list):
        raise ResearchError("unsupported curriculum document")
    annotation_map = annotations.get("annotations", annotations)
    if not isinstance(annotation_map, dict):
        raise ResearchError("annotation map required")
    result: dict[str, Profile] = {}
    for problem in curriculum["problems"]:
        if not isinstance(problem, dict):
            raise ResearchError("problem record must be an object")
        key = problem.get("problem_key")
        if not isinstance(key, str) or not re.fullmatch(r"(leetcode|atcoder|codeforces|baekjoon):[A-Za-z0-9][A-Za-z0-9_.-]*", key):
            raise ResearchError("platform-qualified problem key required")
        if key in result:
            raise ResearchError(f"duplicate problem key: {key}")
        annotation = annotation_map.get(key, {})
        if not isinstance(annotation, dict):
            raise ResearchError(f"annotation must be an object: {key}")
        raw_concepts = problem.get("concepts", [])
        if not isinstance(raw_concepts, list) or any(not isinstance(item, str) for item in raw_concepts):
            raise ResearchError(f"concepts must be a string list: {key}")
        result[key] = Profile(
            problem_key=key,
            platform=key.split(":", 1)[0],
            native_difficulty=_clean_text(problem.get("native_difficulty")),
            concepts=tuple(sorted({item.strip().casefold() for item in raw_concepts if item.strip()})),
            # No problem statement, solution, or unlisted annotation field is read.
            title=_clean_text(annotation.get("title")),
            concept_rationale=_clean_text(annotation.get("concept_rationale")),
        )
    return result

def catalog_hash(profiles: dict[str, Profile]) -> str:
    payload = {
        key: {
            "title": item.title,
            "concept_rationale": item.concept_rationale,
            "concepts": list(item.concepts),
            "platform": item.platform,
            "native_difficulty": item.native_difficulty,
        }
        for key, item in sorted(profiles.items())
    }
    return sha256_bytes(json_bytes(payload))

def feature_vector(source: Profile, target: Profile, text_scores: dict[str, float]) -> dict[str, float]:
    left, right = set(source.concepts), set(target.concepts)
    union = left | right
    return {
        "concept_jaccard": len(left & right) / len(union) if union else 0.0,
        "same_platform": float(source.platform == target.platform),
        "same_native_difficulty": float(bool(source.native_difficulty) and source.native_difficulty == target.native_difficulty),
        "title_tfidf": text_scores.get("title", 0.0),
        "rationale_tfidf": text_scores.get("rationale", 0.0),
        "concept_text_tfidf": text_scores.get("concept_text", 0.0),
    }

class TfidfIndex:
    """A fixed Unicode-token TF-IDF corpus with smoothed IDF and cosine scores."""

    def __init__(self, profiles: dict[str, Profile]):
        self.documents: dict[str, dict[str, Counter[str]]] = {}
        self.weights: dict[str, dict[str, dict[str, float]]] = {}
        for field, getter in (
            ("all", lambda p: p.text),
            ("title", lambda p: p.title_text),
            ("rationale", lambda p: p.rationale_text),
            ("concept_text", lambda p: p.concept_text),
        ):
            docs = {key: Counter(_tokens(getter(profile))) for key, profile in profiles.items()}
            document_frequency = Counter(token for counts in docs.values() for token in counts)
            size = len(docs)
            idf = {token: math.log((size + 1) / (frequency + 1)) + 1.0 for token, frequency in document_frequency.items()}
            weights = {key: {token: (1.0 + math.log(count)) * idf[token] for token, count in counts.items()} for key, counts in docs.items()}
            self.documents[field] = docs
            self.weights[field] = weights

    def cosine(self, source: str, target: str, field: str = "all") -> float:
        left, right = self.weights[field].get(source, {}), self.weights[field].get(target, {})
        if not left or not right:
            return 0.0
        dot = math.fsum(left[token] * right.get(token, 0.0) for token in left)
        left_norm = math.sqrt(math.fsum(value * value for value in left.values()))
        right_norm = math.sqrt(math.fsum(value * value for value in right.values()))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def pair_features(self, source: str, target: str) -> dict[str, float]:
        return {
            "title": self.cosine(source, target, "title"),
            "rationale": self.cosine(source, target, "rationale"),
            "concept_text": self.cosine(source, target, "concept_text"),
        }

class MetadataProvider:
    name = "metadata"

    def __init__(self, profiles: dict[str, Profile]):
        self.profiles = profiles

    def score(self, source: str, target: str, index: TfidfIndex | None = None) -> float:
        left, right = self.profiles[source], self.profiles[target]
        overlap = feature_vector(left, right, {})["concept_jaccard"]
        platform = float(left.platform == right.platform)
        difficulty = float(bool(left.native_difficulty) and left.native_difficulty == right.native_difficulty)
        return 0.60 * overlap + 0.25 * platform + 0.15 * difficulty

class TfidfProvider:
    name = "tfidf"

    def __init__(self, profiles: dict[str, Profile]):
        self.profiles = profiles
        self.index = TfidfIndex(profiles)

    def score(self, source: str, target: str, index: TfidfIndex | None = None) -> float:
        return self.index.cosine(source, target)

class NeighborProvider:
    """Read-only lookup of one precomputed namespace; never computes vectors."""

    name = "precomputed_neighbors"

    def __init__(self, document: dict, *, expected_namespace: str, expected_catalog_hash: str):
        manifest = document.get("manifest")
        neighbors = document.get("neighbors")
        required = {"schema_version", "namespace", "revision", "license_reference", "dimensions", "normalization", "metric", "preprocessing", "input_sha256", "ordered_keys", "ordered_keys_sha256"}
        if not isinstance(manifest, dict) or not required.issubset(manifest) or not isinstance(neighbors, dict):
            raise ResearchError("neighbor artifact manifest or rows are incomplete")
        if manifest["schema_version"] != 1:
            raise ResearchError("unsupported neighbor artifact schema")
        if manifest["namespace"] != expected_namespace:
            raise NamespaceMismatchError("neighbor artifact namespace mismatch")
        if manifest["input_sha256"] != expected_catalog_hash:
            raise ResearchError("neighbor artifact catalog input hash mismatch")
        if not isinstance(manifest["dimensions"], int) or manifest["dimensions"] <= 0:
            raise ResearchError("invalid neighbor artifact dimensions")
        if not manifest["revision"] or not manifest["license_reference"] or not manifest["normalization"] or not manifest["metric"] or not manifest["preprocessing"]:
            raise ResearchError("neighbor artifact provenance is incomplete")
        keys = manifest["ordered_keys"]
        if (not isinstance(keys, list) or any(not isinstance(key, str) for key in keys) or
                len(keys) != len(set(keys)) or sha256_bytes(json_bytes(keys)) != manifest["ordered_keys_sha256"]):
            raise ResearchError("ordered neighbor key manifest mismatch")
        self.manifest = manifest
        self.neighbors: dict[str, dict[str, float]] = {}
        self.directions: dict[str, dict[str, str]] = {}
        for source, rows in neighbors.items():
            if not isinstance(source, str) or not isinstance(rows, list):
                raise ResearchError("invalid neighbor list")
            result = {}
            directions = {}
            for row in rows:
                if not isinstance(row, dict) or set(row) - {"problem_key", "score", "direction"} or not isinstance(row.get("problem_key"), str):
                    raise ResearchError("invalid neighbor row")
                score = row.get("score")
                if type(score) not in {int, float} or not math.isfinite(score):
                    raise ResearchError("invalid neighbor score")
                target = row["problem_key"]
                if target in result:
                    raise ResearchError("duplicate neighbor key")
                result[target] = float(score)
                if "direction" in row:
                    if row["direction"] not in {"source_to_target", "target_to_source", "none"}:
                        raise ResearchError("invalid neighbor direction")
                    directions[target] = row["direction"]
            self.neighbors[source] = result
            self.directions[source] = directions

    def score(self, source: str, target: str, index: TfidfIndex | None = None) -> float:
        return self.neighbors.get(source, {}).get(target, 0.0)

    def direction(self, source: str, target: str) -> str | None:
        return self.directions.get(source, {}).get(target)

def load_neighbor_provider(path: str | Path | None, *, expected_namespace: str | None, expected_catalog_hash: str, profiles: dict[str, Profile]) -> NeighborProvider | None:
    if path is None or not Path(path).is_file():
        return None
    if not expected_namespace:
        raise ResearchError("expected namespace is required with a neighbor artifact")
    document = read_json(path)
    provider = NeighborProvider(document, expected_namespace=expected_namespace, expected_catalog_hash=expected_catalog_hash)
    if provider.manifest["ordered_keys"] != sorted(profiles):
        raise ResearchError("neighbor ordered keys do not match catalog")
    unknown = (set(provider.neighbors) | {target for row in provider.neighbors.values() for target in row}) - set(profiles)
    if unknown:
        raise ResearchError("neighbor artifact contains keys outside the input catalog")
    return provider
