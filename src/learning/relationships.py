"""Offline relationship candidates backed by one explicit scoring namespace."""
from __future__ import annotations

from pathlib import Path

from .research import (
    MetadataProvider,
    NamespaceMismatchError,
    ResearchError,
    TfidfProvider,
    build_profiles,
    catalog_hash,
    load_neighbor_provider,
)


class RelationshipProvider:
    """Deterministic baseline or one compatible precomputed-neighbor list."""

    def __init__(self, curriculum: dict, annotations: dict, *, method: str = "metadata",
                 neighbor_path: str | Path | None = None, expected_namespace: str | None = None):
        self.profiles = build_profiles(curriculum, annotations)
        self.input_hash = catalog_hash(self.profiles)
        self.neighbor = load_neighbor_provider(neighbor_path, expected_namespace=expected_namespace,
                                               expected_catalog_hash=self.input_hash, profiles=self.profiles)
        if self.neighbor is not None and method != "metadata":
            raise ResearchError("a precomputed namespace cannot be blended with another scorer")
        if self.neighbor is not None:
            self.scorer = self.neighbor
            self.status = "precomputed_namespace"
            self.namespace = self.neighbor.manifest["namespace"]
        elif method == "metadata":
            self.scorer = MetadataProvider(self.profiles)
            self.status = "metadata_fallback_missing_artifact" if neighbor_path else "metadata_baseline"
            self.namespace = "metadata-concept-platform-difficulty-v1"
        elif method == "tfidf":
            self.scorer = TfidfProvider(self.profiles)
            self.status = "tfidf_baseline"
            self.namespace = "authored-text-tfidf-v1"
        else:
            raise ResearchError("method must be metadata or tfidf")

    def candidates(self, problem_key: str, *, limit: int = 10) -> list[dict]:
        if problem_key not in self.profiles:
            raise ResearchError("unknown source problem key")
        if type(limit) is not int or limit < 1 or limit > 500:
            raise ResearchError("candidate limit must be from 1 through 500")
        missing_query = self.neighbor is not None and problem_key not in self.neighbor.neighbors
        scorer = MetadataProvider(self.profiles) if missing_query else self.scorer
        namespace = "metadata-concept-platform-difficulty-v1" if missing_query else self.namespace
        status = "metadata_fallback_missing_query" if missing_query else self.status
        targets = (self.neighbor.neighbors.get(problem_key, {}) if self.neighbor and not missing_query else
                   {key: None for key in self.profiles if key != problem_key})
        ranked = sorted(targets, key=lambda target: (-scorer.score(problem_key, target), target))
        results = []
        for key in ranked[:limit]:
            source, target = self.profiles[problem_key], self.profiles[key]
            results.append({
                "problem_key": key,
                "score": scorer.score(problem_key, key),
                "components": {
                    "shared_concepts": sorted(set(source.concepts) & set(target.concepts)),
                    "source_platform": source.platform,
                    "target_platform": target.platform,
                    "source_native_difficulty": source.native_difficulty,
                    "target_native_difficulty": target.native_difficulty,
                },
                "direction": self.neighbor.direction(problem_key, key) if self.neighbor and not missing_query else None,
                "namespace": namespace,
                "status": status,
            })
        return results


__all__ = ["RelationshipProvider", "NamespaceMismatchError", "ResearchError"]
