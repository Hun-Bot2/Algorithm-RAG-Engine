# Design, task breakdown and validation

## Research and decision boundary

Reuse the existing full catalog, pinned MiniLM, exact retrieval engine and A1/A2/
A1-all outputs. Add a new versioned research namespace research/algorithm_system/
and private captures under private-learning/algorithm-system/. The previous report
found metadata gaps and interpreted dataset rights conservatively. The new review
will assess the expressly published CodeContests non-code CC-BY-4.0 scope and actual
CodeNet dataset license separately; retain third-party conditions and attribution.
This does not authorize direct platform scraping, redistribution or LC unquarantine.

## Sequential implementation

1. CP1 audit + taxonomy + native difficulty availability, followed by independent
   semantic/statistical review and an actual-coverage checkpoint report.
2. CP2 source-specific bounded ingestion with pinned revisions, source/data-license
   receipts, hashes and authoritative ID joins; native topic-conditioned ECDF
   baseline, query-independent bootstrap and frozen split contracts. Mark unavailable
   supervised difficulty calibration rather than fitting on semantic-only labels.
3. CP3 matched statement corpus and deterministic duplicate/review queues; original
   384D local MiniLM with token-aware full coverage, exact lexical/tag/hybrid
   rankings, honest pooled-label evaluation and additional blind review candidates.
4. CP4 compression after the baseline, notebook/figures/report and adaptive engine
   proposal. Record blocked branches and continue independent authorized work.

## Data and interfaces

Keep native catalog IDs separate from publisher dataset row IDs. For unresolved
AtCoder IDs retain dataset source identity and missing original task ID; do not
fuzzy-join by title as verified. Each statement retains raw/document hashes,
language/detection basis, source/publisher revision, license/use scope, translation
flag, section/public-sample availability and normalized text. Duplicates are
annotations, not deletion. Tag mappings preserve originals and separate direct
concepts, ancestors and related-only candidates. Difficulty values and confidence
remain platform-specific; quantiles are provisional relative ranks, not equal skill.

Use HTTP range/column projection for CodeContests Parquet if supported; reject
servers ignoring ranges rather than downloading multi-GB solution columns. CodeNet
may use its small description archive and metadata only, not the submission dump.
All outputs are new run directories. MiniLM loads only the existing verified local
snapshot; no inference server or OCI model workload. Report raw and deduplicated
corpus membership and never expose statement text in public figures or queues.

## Validation and risks

Run focused fixture tests, source identity/schema/hash checks, duplicate family
checks and actual HTTP receipts. Review implementations before accepting findings.
Compare exact blockwise scores against brute force on bounded samples. Bootstrap
by query/family, preserve missing outcomes and withhold cross-platform equality
claims when difficulty ground truth is absent. Topic overlap may be circular with
tag-based methods; label pool favors contributing baselines. Statement languages,
unknown IDs, Premium uncertainty and sparse AtCoder tags can block a three-platform
matched comparison; report subsets explicitly rather than filling gaps.
