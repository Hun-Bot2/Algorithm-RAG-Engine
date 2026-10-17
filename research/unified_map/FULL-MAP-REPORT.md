# Full-catalog interactive similarity map — 2026-10-17

All 24,297 catalog problems (LeetCode 3,053, Codeforces 11,457, AtCoder 9,787) now have a title vector and a
position in PCA, UMAP and t-SNE maps, with an interactive explorer. **Nothing here establishes that two problems
share an algorithm.** Positions and scores come from a MiniLM embedding of each problem's *title*; platform tags,
the AI-reviewed taxonomy and historical pair judgments are shown as separate evidence and never influenced the
vectors, neighbors or coordinates. No statement text was read, no data was collected, no model was downloaded or
promoted, and no existing dataset or experiment was modified (the 501-file preservation snapshot still passes).

Spec: [interactive-map.md](../../specs/unified-similarity-map/interactive-map.md). Notebook: [full_map.ipynb](full_map.ipynb).
Receipts: [full-receipts/](full-receipts/). Figures: [figures/](figures/) (points only, no problem text).

## Coverage gap and how it was closed

The earlier map had compatible title vectors for 21,297 of 24,297 nodes. The 3,000 uncovered nodes were 2,157
eligible non-Latin titles (2,141 AtCoder, 16 Codeforces) and 843 excluded rows (740 Latin, 103 non-Latin).

- The 21,297 A1 vectors were **reused bit-for-bit** (verified); only the 3,000 missing titles were encoded with the
  same pinned model and settings (offline, CPU, one thread, batch 64): 3.2 s. A re-encode of 500 sampled A1 titles in
  a different batch composition deviated by at most 1.8e-7 per component (minimum cosine to the stored vector 0.9999998).
- Every node carries a reliability tier from the share of its tokens the English tokenizer cannot represent:
  **22,030 reliable, 1,684 degraded, 583 unreliable**. A non-Latin title is capped at *degraded*, because Cyrillic
  titles tokenize into known pieces and the unknown-share alone would have called them reliable (found and fixed
  during this work). Romanized non-English Latin titles cannot be detected and stay *reliable*: a limitation.
- Non-Latin titles are mean 31% unknown tokens. Their vectors exist but carry little meaning; they are flagged in
  the map, never hidden or dropped, and excluded rows stay on the map with their exclusion reason.

## Projection quality (all 24,297 nodes, versus exact 384-d title neighbors)

| Projection | Top-10 neighbors kept (any platform) | Cross-platform | Trustworthiness | Continuity | Fit time |
|---|---:|---:|---:|---:|---:|
| PCA (exact SVD) | 3.5% | 2.7% | 0.700 | 0.840 | 1.6 s |
| UMAP (cosine, 15, 0.1) | 28.9% | 24.7% | 0.916 | 0.935 | 29 s |
| t-SNE (cosine, perplexity 30) | 48.3% | 35.6% | 0.974 | 0.931 | 203 s |

"Top-10 kept" is the share of a problem's exact top-10 title neighbors that are also its 10 nearest points in 2D,
computed for all nodes. Trust and continuity are estimated on 4,000 seeded query nodes against all 24,297. PCA is
poor at local structure by design; t-SNE preserves the most neighborhoods and UMAP sits between them. Global layout
fidelity was not measured. Fidelity is similar across platforms and reliability tiers for UMAP and t-SNE (UMAP
25–30%, t-SNE 47–49%); PCA's 22% on *unreliable* titles reflects those titles collapsing together.

**Seed stability.** scikit-learn's t-SNE with `init="pca"` ignores the seed: seeds 17 and 18 gave a *bit-identical*
layout, so that repeat measures nothing. Random-init t-SNE (seeds 18, 19) differs moderately (Procrustes disparity
0.21, 71.7% shared 2D top-10) and has the same fidelity. UMAP seeds differ more (disparity 0.49–0.69, about 29%
shared 2D neighbors) at the same fidelity. Read the shape of any single UMAP layout as one sample.

Titles also cluster by platform style: 60.2% of exact top-10 title neighbors are on the query's own platform, against
40.0% expected if platform were unrelated to the title. The maps inherit this, and the 2D same-platform share is
50.7% (PCA), 56.8% (UMAP) and 59.4% (t-SNE).

## What the neighbor scores show

Scores are cosine similarities of title vectors. The top-1 any-platform neighbor has median 0.78 (10th neighbor 0.60),
and **11.9% of problems have a top-1 neighbor with a near-identical title** (cosine ≥ 0.999): many titles are generic
("Sum", "Game") and are reused across platforms. A high score often means shared wording ("Two Sum" ↔ "Second Sum",
"Double Sum" on AtCoder). The page says so beside every score and flags identical normalized titles and same-title
duplicate-candidate groups; a shared title is not evidence of an identical problem.

**Tag agreement (descriptive only, not validation).** Among tagged neighbor slots, the share that share an equivalent
high-confidence taxonomy concept with the query:

| Neighbor list | Observed | Random tagged nodes on the same platforms |
|---|---:|---:|
| Title neighbors, any platform | 52.6% | 40.5% |
| Title neighbors, other platforms | 33.9% | 20.2% |
| t-SNE / UMAP / PCA 2D neighbors | 52.2% / 50.2% / 41.0% | 40.0% / 39.2% / 38.1% |

Title neighbors share concepts more often than chance (1.30× any platform, 1.68× other platforms) and t-SNE and UMAP
keep most of that; PCA keeps little. Coverage is partial: only 68% (any-platform) and 23% (other-platform) of neighbor
slots are tagged on both sides, because AtCoder has no tags and many LeetCode tags map to no concept. Broad shared
tags (greedy, math) do not establish the same invariant or technique. An earlier, naive all-pairs baseline made
cross-platform neighbors look *worse* than chance (33.9% vs 35.3%); that baseline mixed same-platform pairs and was
replaced by the platform-matched one above.

## The interactive map

`private-learning/unified-map/full-v1/interactive.html` (11.4 MB, one file, no external scripts or fonts). It is
private because it embeds titles, tags and URLs of the private catalog; it is not committed. Rebuild it with the
commands in [README.md](README.md). Tested over loopback HTTP in the in-app browser: layout, compare mode, filters,
search, hover, click, zoom, reset and Escape on an earlier build at desktop 1500 px and 375 px phone (light and
dark); the final build was re-checked for load, selection and the fidelity panel and is covered by the 32 payload
checks below. Not tested as a `file://` page.

- **Compare:** single projection or PCA/UMAP/t-SNE side by side with linked selection and a per-panel
  "k/10 of this problem's title neighbors stay among its 10 nearest points here" readout.
- **Explore by platform** (toggle), **tag** (native platform tag or taxonomy concept, dim or hide others),
  **coverage** (show/hide excluded problems and degraded/unreliable-embedding titles) and **search** (title or key).
- **Color by** platform, embedding reliability/eligibility, or per-problem local neighborhood fidelity in that map.
- **Select a problem** for: provenance (reused/new vector, reliability, exclusion reasons, statement-record availability
  flag only), **tag evidence** (native tags and mapped concepts; AtCoder shown as *unavailable*, not empty), exact top-30
  **title neighbors** (any platform or other platforms) with score, shared concepts or "tag overlap unavailable",
  identical-title and duplicate-candidate flags and judged-pair badges, and **judged evidence** (3,647 pooled AI-reviewed
  pair judgments and the 180 earlier curated relationship judgments, with legacy AtCoder keys resolved through the
  reviewed alias file). Absent judgments read "unknown, not negative".

## Validation

- 29 new fixture tests plus the existing suites: `research/unified_map` 41, `research/full_catalog` 48, `tests/` 78; all pass.
  Trustworthiness and continuity match scikit-learn exactly on random data; neighbor ranking equals brute force
  across block sizes and settings.
- [validation.v1.json](full-receipts/validation.v1.json): **32 independent checks pass**, recomputed from raw
  artifacts: receipt hashes; one vector per node; all 21,297 reused vectors identical; no non-Latin title marked
  reliable; every stored neighbor valid (no self, no exact duplicates, non-increasing, platform rule), scores equal to
  recomputed float64 cosine, and top-30 equals a float64 brute force on 400 random queries per list; PCA, UMAP and
  t-SNE re-runs reproduce the saved coordinates; per-node fidelity matches an independent recomputation on 300 nodes
  (±1 slack for 2D distance near-ties); HTML payload equals the saved neighbors (scores to 1e-4) and coordinates;
  no statement-text fields in any written artifact; and the existing map-v2 verifier with the 501-file preservation
  snapshot still passes.
- Browser checks: loads with 24,297 nodes and the expected counts per platform; compare mode; tag/concept filters
  (2,547 `dp`, 3,996 `greedy` matches), platform toggle (14,510 without AtCoder), search, hover, click selection, wheel
  zoom, reset and Escape all behave as intended; only console message is the static server's favicon 404.

## Limits and what is not claimed

- Title similarity is a weak proxy for algorithmic similarity and the page makes no equivalence claim.
- 2D distances are distorted; fidelity above is the quantitative guide. PCA is not suitable for neighborhood reading.
- UMAP layouts vary across seeds; only one seed (17) is shown. Trust/continuity use a 4,000-query sample.
- 2,260 non-Latin titles (mostly Japanese AtCoder) are embedded by an English-only model and are low-meaning; no
  multilingual model was downloaded. Romanized non-English titles are undetected.
- Judged pairs are a small, selected, AI-reviewed set (not human gold); tag coverage excludes AtCoder.
- Not done: statement-based maps (statements remain outside this pipeline), difficulty calibration across platforms,
  learner outcomes, or any promotion of the map into recommendations.

## Process notes

Three mistakes were caught and fixed before the results above: an edit to `research/full_catalog/retrieve.py`
(pinned by hash in the preservation snapshot) was reverted and the any-platform search moved into this module;
the cross-platform tag baseline was replaced as described; and a t-SNE seed check that tested nothing was replaced by
random-init runs.
