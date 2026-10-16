"""Public facade for deterministic offline relationship research helpers."""
from .research_contracts import (FEATURES, RELATION_TYPES, RESEARCH_VERSION, REVIEWED_STATES,
                                 TOKENIZER_VERSION, NamespaceMismatchError, ResearchError,
                                 json_bytes, provenance, read_json, sha256_bytes, sha256_file)
from .research_profiles import (MetadataProvider, NeighborProvider, Profile, TfidfIndex,
                                TfidfProvider, build_profiles, catalog_hash, feature_vector,
                                load_neighbor_provider)
from .research_metrics import (concept_evidence_overlap, paired_bootstrap, reviewed_judgments,
                               score_judgments, validate_judgments, _query_metric)
from .research_training import (fit_feature_scorer, promotion_gate, run_config, scorer_score,
                                solve_ridge, training_gate, training_snapshot_sha256)
