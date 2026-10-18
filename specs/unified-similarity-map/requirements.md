# Unified similarity map requirements

- The research SHALL preserve existing datasets, embeddings, judgments and runs.
- Every native catalog node SHALL have explicit coverage and eligibility metadata.
  Excluded or unavailable representations SHALL not be invented.
- Semantic title, semantic statement and algorithm-tag evidence MUST remain
  distinguishable. Shared broad tags SHALL not establish algorithm equivalence.
- Reuse pinned embeddings without comparing incompatible representations directly.
- User-approved identity mappings SHALL be hash-validated; p03510 stays unresolved
  and abc084_c retains its quality flags. Original source records SHALL not change.
- Retrieval SHALL be cross-platform, deterministic and exact where specified, with
  Top10/20/50 outputs and explicit duplicate exclusions.
- Comparisons SHALL use matched populations. Missing labels are unknown, not negative.
  Historical AI labels SHALL not become human gold or algorithmic truth.
- PCA/SVD compression SHALL be compared to original384 on identical eligible rows.
  UMAP/t-SNE SHALL report sample coverage, parameters, seeds and local-neighbor loss;
  visual clusters SHALL not be used as proof of algorithmic similarity.
- Record actual coverage, hashes, runtime, environment, quality metrics and limitations.
  No deployment, training, new statement collection or automatic model promotion.
