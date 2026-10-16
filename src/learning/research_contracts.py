"""Shared version, errors, hashing, input, and run provenance contracts."""
from __future__ import annotations
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

RESEARCH_VERSION = "relationship-research-v1"

TOKENIZER_VERSION = "unicode-words-casefold-v1"

FEATURES = (
    "concept_jaccard",
    "same_platform",
    "same_native_difficulty",
    "title_tfidf",
    "rationale_tfidf",
    "concept_text_tfidf",
)

REVIEWED_STATES = {"accepted", "adjudicated", "approved"}

RELATION_TYPES = {"similarity", "prerequisite", "transfer"}

class ResearchError(ValueError):
    """Invalid or incompatible research input."""

class NamespaceMismatchError(ResearchError):
    """Precomputed neighbors belong to another embedding space."""

def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ResearchError(f"JSON object required: {path}")
    return value

def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)

def _clean_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""

def provenance(*, input_paths: list[str | Path], config: dict, elapsed_seconds: float, peak_rss: int | None = None, artifact_paths: list[str | Path] | None = None) -> dict:
    if peak_rss is None:
        try:
            import resource
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except (ImportError, AttributeError):
            peak_rss = None
    unit = "bytes" if sys.platform == "darwin" else "kibibytes"
    sizes = {}
    for path in artifact_paths or []:
        item = Path(path)
        sizes[str(item)] = item.stat().st_size if item.is_file() else None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "peak_rss": {"value": peak_rss, "unit": unit, "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss"},
        "artifact_sizes_bytes": sizes,
        "input_sizes_bytes": {str(Path(path)): Path(path).stat().st_size if Path(path).is_file() else None for path in input_paths},
        "cash_cost": None,
        "cash_cost_boundary": "No billing receipt is available; cost is not measured.",
        "inputs_sha256": {str(Path(path)): sha256_file(path) for path in input_paths},
        "config": config,
        "config_sha256": sha256_bytes(json_bytes(config)),
    }
