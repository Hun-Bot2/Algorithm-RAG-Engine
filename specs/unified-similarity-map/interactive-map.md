# Full-catalog interactive similarity map — spec v1

Extends `specs/unified-similarity-map/` (map-v2, reduction-v2). Existing outputs,
frozen labels and experiments are read-only; everything new is written to fresh
versioned paths.

## Research summary (measured 2026-10-17)

- Native nodes: 24,297 (`map-v2/nodes.jsonl`). Compatible MiniLM title vectors
  (A1): 21,297. **3,000 nodes have no vector**: 2,157 eligible non-Latin titles
  (2,141 AtCoder, 16 Codeforces) and 843 excluded rows (740 Latin, 103 non-Latin).
- Of 2,260 non-Latin titles, the pinned English MiniLM tokenizer maps 23.8% of
  tokens to `[UNK]`; 579 titles are at least half unknown. They can be embedded but
  carry degraded information.
- Existing 2D coordinates cover only a 3,000-problem sample (reduction-v2).
- Tags: 14,269 nodes have source tags; all 9,787 AtCoder nodes have none.
- Statement text is never read here. Only `statement_availability` status flags are shown.

## Requirements

- R1. THE SYSTEM SHALL produce one 384-d title vector per catalog node (24,297),
  reusing the 21,297 A1 vectors unchanged and encoding only the missing 3,000 with
  the same pinned model and settings, offline, no download, no API.
- R2. WHEN reuse is applied, THE SYSTEM SHALL re-encode a seeded sample of A1 rows
  and report the maximum deviation from the stored vectors.
- R3. Every node SHALL carry an embedding-reliability tier (`reliable`, `degraded`,
  `unreliable`) derived from its tokenizer-unknown share, never hidden or dropped.
- R4. THE SYSTEM SHALL compute PCA, UMAP and t-SNE 2D coordinates for **all** nodes
  with fixed seeds, and record parameters, seeds, duration and artifact hashes.
- R5. THE SYSTEM SHALL measure each projection against exact full-catalog cosine
  neighbors: Top-10 neighbor preservation (any-platform and cross-platform),
  trustworthiness and continuity at k=10 (seeded query sample), platform mixing,
  seed stability, and descriptive tag agreement against a random baseline.
- R6. Nearest neighbors SHALL be exact blockwise cosine (no N×N matrix), with
  deterministic key tie-breaking, for any-platform and other-platform targets.
- R7. THE MAP SHALL let a user filter by platform, reliability/eligibility, native
  tag and taxonomy concept; search by title/key; compare the three projections side
  by side with linked selection; and select a node to see its neighbors.
- R8. For a selected node THE MAP SHALL show cosine scores with their definition,
  and keep three evidence channels visibly separate: (a) semantic title similarity,
  (b) platform tags and AI-reviewed taxonomy concepts (never used to place points
  or rank semantic neighbors), (c) historical AI-reviewed pair judgments and
  statement availability status. Missing tags SHALL display as unavailable.
- R9. THE SYSTEM SHALL NOT state or imply that neighbors solve the same algorithmic
  problem. Scores are title-embedding similarity; the page SHALL say so.
- R10. Private titles/URLs/tags SHALL stay under `private-learning/`; Git receives
  only code, tests, aggregate receipts, figures with no problem text and docs.

## Design

```text
research/unified_map/full_embed.py     coverage audit + missing-vector encode + reuse check
research/unified_map/full_project.py   PCA/UMAP/t-SNE on all nodes (+ seed repeats)
research/unified_map/full_quality.py   exact neighbors, preservation/trust/continuity, agreement
research/unified_map/full_html.py      payload packing + single-file interactive HTML
research/unified_map/test_full_*.py    fixture tests (no private data)
private-learning/unified-map/full-v1/  vectors, neighbors, coords, interactive.html
research/unified_map/full-map-receipt.v1.json   aggregate receipts (committed)
```

Data flow: nodes.jsonl + A1 vectors -> full vectors -> exact neighbors ->
projections -> quality metrics -> packed payload -> self-contained HTML (no
external scripts; works offline). Per-node, per-projection preservation counts are
packed so the map can color by local fidelity.

Tradeoffs: titles are weak inputs, so the map documents title similarity, not
algorithms. The any-platform mode (self and exact duplicates excluded only) lives in
`full_quality.py` and reuses `retrieve.exclusion_mask`/`rank_row`: the existing
`research/full_catalog/retrieve.py` is pinned by hash in the preservation snapshot and
stays byte-identical. Duplicate-candidate members are shown with a flag, not hidden.
Projections run single-threaded for reproducibility. scikit-learn's t-SNE with
`init="pca"` ignores the seed, so stability is measured with random-init t-SNE runs.

## Tasks

- [x] 1. Coverage audit and reliability tiers; encode missing vectors; reuse check.
- [x] 2. Exact neighbors for all nodes (any and other-platform); any-mode tests.
- [x] 3. Full PCA/UMAP/t-SNE with seed repeats; quality metrics vs brute force.
- [x] 4. Pack payload and build the interactive HTML; fixture tests.
- [x] 5. Verify in a real browser (load, filter, select, compare, console clean).
- [x] 6. Notebook, report, README/AGENTS updates; run the full suites.

## Validation plan

Fixture unit tests (round-trip packing, brute-force equality for neighbors and
metrics, `trustworthiness` against scikit-learn on small data, schema and
no-statement-field checks); an independent verifier (`full_verify.py`) that recomputes
vector reuse, neighbors, projection reproducibility, a fidelity sample and HTML payload
equality from raw artifacts; the existing suites and the 501-file preservation snapshot
still pass; browser smoke test with console and DOM assertions.

## Risks

Non-Latin titles produce weak vectors (flagged, not hidden). 2D maps distort
distances; preservation numbers are reported per projection and no projection is
treated as truth. Tag agreement is descriptive and partly circular for tag
rankers. Labels cover 3,647 pairs and are sparse, so they are evidence for few
neighbors only. UMAP/t-SNE are stochastic; seeds and stability are reported.

## Result

Completed 2026-10-17; see `research/unified_map/FULL-MAP-REPORT.md`. Measured coverage: 24,297 nodes with vectors
(21,297 reused, 3,000 new; 22,030 reliable, 1,684 degraded, 583 unreliable). Top-10 title-neighbor preservation in 2D:
PCA 3.5%, UMAP 28.9%, t-SNE 48.3%. 32 independent validation checks pass. Deviations from this spec: seed repeats use
random-init t-SNE because PCA-init t-SNE ignores the seed; the any-platform search lives in `full_quality.py`.
