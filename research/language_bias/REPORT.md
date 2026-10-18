# Language-driven clustering in the unified similarity map — 2026-10-17

**Problem.** Japanese AtCoder titles formed isolated clusters of unrelated problems in the 24,297-problem map: an
English-only encoder reads them by script and shared characters, not meaning.

**Result.** Embedding **English-translated titles** with the same MiniLM (the platform's own English wording where a
title is "Japanese (English)", a pinned local machine translation otherwise) is the best-supported fix among eight
candidates. It was chosen by a rule fixed before any new result existed. In the rebuilt maps, the share of pure-Japanese
titles sitting in a Japanese-only 2D neighborhood falls from 97.7% to 0.4% (UMAP) and 2.2% (t-SNE), local fidelity is
unchanged, and English-title neighbors are unaffected. **This does not prove the new neighbors are algorithmically
right**: no algorithm labels exist for Japanese titles, translation is noisy, and the evidence is cross-language
consistency plus a weak tag-coherence signal. Details and limits below.

Spec: [language-bias.md](../../specs/unified-similarity-map/language-bias.md). Notebook: [language_bias.ipynb](language_bias.ipynb).
Receipts: [receipts/](receipts/). Figures: [figures/](figures/) (points only, no problem text). Existing experiments are
untouched: every `full-v1` artifact, `retrieve.py` and the 501-file preservation snapshot verified byte-identical.

## Baseline: how strong is the bias?

Nodes by title script: English/Latin 22,037; Japanese 2,216 (1,421 pure, 655 "Japanese (English)", 140 mixed); Russian 43;
Korean 1. For the 1,421 pure-Japanese titles, **98.8% of top-10 neighbors are Japanese** against 9.1% expected if
language were ignored (10.8x), and 99.9% come from AtCoder. Japanese halves retrieve their own English halves with MRR
0.039 (chance 0.0017), and a title's neighbors from its Japanese versus English half overlap 0.016. Russian titles
cluster the same way (0.93 same-language neighbors for 43 nodes).

## What was compared (identical nodes, queries and benchmark)

Candidates: C0 raw-title MiniLM (baseline); C1 per-language centroid removal (no new model); **C2 MiniLM on English-translated titles**;
C3 multilingual-e5-small; C4 paraphrase-multilingual-MiniLM-L12-v2; C5 LaBSE (downloaded by the user). Controls: random
unit vectors per title, and centroid removal with permuted language labels. bge-m3 and the other translators were not usable
(no safetensors weights on their main branch) and NLLB is non-commercial; none were tested.

**Benchmark.** 599 unique "Japanese (English)" titles (655 nodes) give a labeled translation pair with no translation by us.
Queries are the Japanese half, candidates the English halves plus 5,000 distractor titles (5,551 candidates).

| Candidate | Same-language share, pure-Japanese (chance 9.1%) | Translation MRR | Neighbor consistency@10 (95% interval) | English quality retained | Concept-coherence lift (coverage) |
|---|---:|---:|---:|---:|---:|
| C0 raw baseline | 98.8% | 0.039 | 0.016 [0.011, 0.020] | 1.000 | unassessable (0.1%) |
| C1 centroid removal | 96.9% | 0.043 | 0.021 [0.015, 0.026] | 1.011 | unassessable (1.2%) |
| **C2 English-translated** | **25.2%** | 0.664 | **0.506 [0.473, 0.540]** | 1.008 | 1.105 [1.063, 1.147] (40%) |
| C3 multilingual-e5-small | 88.8% | 0.579 | 0.107 [0.092, 0.121] | 0.986 | unassessable (4.5%) |
| C4 paraphrase-multilingual | 62.4% | 0.557 | 0.394 [0.362, 0.421] | 1.000 | 1.097 [1.036, 1.160] (20%) |
| C5 LaBSE | 35.7% | **0.740** | 0.410 [0.384, 0.431] | 0.974 | 1.088 [1.043, 1.137] (35%) |
| control: random vectors | 10.5% | 0.001 | 0.000 [0.000, 0.001] | 0.773 | 1.009 [0.977, 1.049] (57%) |
| control: permuted-label centroids | 98.7% | 0.043 | 0.016 [0.012, 0.021] | 1.007 | unassessable |

Neighbor consistency is the overlap of a title's top-10 neighbors when queried by its Japanese half versus its English half
in the same index (chance 0.0004). "English quality retained" is the tag-agreement lift of English-title neighbors against a
platform-matched random baseline, relative to C0. Coherence lift asks whether a pure-Japanese title's tagged neighbors agree
with each other more than platform-matched random nodes do; it is only assessable when enough neighbor slots are tagged.

**How the random control is used.** Random vectors have the lowest language bias (10.5%) and no meaning: MRR 0.001,
consistency 0, English quality 0.77, coherence 1.0. So less language clustering or more platform mixing cannot be a selection
criterion, and the rule below never uses it. C2's low same-language share comes with high alignment and above-chance coherence;
random's does not.

**Pre-registered rule** (written before any new result, applied mechanically by `select_representation.py`): a candidate must keep
English quality (>= 0.95), beat 0.25 translation MRR and 10x the random control, and not fall below chance coherence when
assessable; the winner has the highest neighbor consistency and must beat C0 by at least 0.10 with an interval excluding zero.
C1 was ineligible (MRR 0.043). C2, C3, C4 and C5 passed the guards (C3's coherence is unassessable and its guard waived). **C2 won**:
consistency 0.506 against 0.410 for the next candidate, gain over C0 0.490 [0.458, 0.524].

**Robustness.** The ranking C2 > C5 ~ C4 > C3 >> C1 ~ C0 holds at k=5 (C2 0.498), k=20 (0.524) and when every parallel title
is removed from the neighbor pool (C2 0.504, C5 0.404, C4 0.391, C3 0.117, C0 0.006). That last check removes an asymmetry that
could have favored C2, whose parallel nodes are indexed by their English half.

## Translation quality and provenance

Texts embedded: 22,037 used as written (English), 655 the platform's own English, 1,561 machine-translated, 44 untranslated
(Russian and Korean: the translator reads only Japanese). On the 599 reference pairs, machine translations match the platform's
English exactly 21% of the time (after normalization); mean cosine to the reference is 0.70 (median 0.77) against 0.16 for
shuffled references. Tiers in the rebuilt map: 22,685 reliable, 1,564 degraded (machine-translated), 48 unreliable (untranslated
or mostly unknown tokens). Failures are real: place names are mistranslated, and numbered tutorial titles can be garbled
(a "1.00. Introduction" page came out as an unrelated sentence). Such nodes are flagged and their neighbors should not be trusted.

## Rebuilt maps (all 24,297 nodes, fixed seeds)

| Map | v1 raw titles: top-10 kept / trust / continuity | v2 English titles: top-10 kept / trust / continuity |
|---|---|---|
| PCA | 3.5% / 0.700 / 0.840 | 3.4% / 0.693 / 0.834 |
| UMAP | 28.9% / 0.916 / 0.935 | 30.2% / 0.915 / 0.922 |
| t-SNE | 48.3% / 0.974 / 0.931 | 48.6% / 0.974 / 0.927 |

Cross-platform neighbors kept improved (UMAP 24.7% to 27.2%, t-SNE 35.6% to 39.0%). Tag agreement of title neighbors is
unchanged (any platform 53.1% against 40.5% matched chance; other platforms 34.5% against 20.4%).

| Pure-Japanese titles (1,421) | v1 UMAP / t-SNE | v2 UMAP / t-SNE |
|---|---|---|
| In a Japanese-only 2D neighborhood (top-10 at least 90% Japanese) | 97.7% / 97.7% | 0.4% / 2.2% |
| Same-language share of 2D top-10 | 98.8% / 99.1% | 19.0% / 22.1% |
| Local fidelity (top-10 kept) | 0.244 / 0.484 | 0.280 / 0.471 |

Seed stability: t-SNE (random init, seeds 18/19) Procrustes disparity 0.04, 77% shared 2D neighbors; UMAP (seeds 17/18/19)
0.75 to 0.89 and about 30%. As before, scikit-learn t-SNE with PCA init ignores the seed. The interactive map is rebuilt for v2
(`private-learning/unified-map/full-v2/interactive.html`, private) with a language color mode, the text actually embedded and its
source for each title, and a before/after language-bias panel.

## Limits and what is not claimed

- **Not proven algorithmic.** No algorithm labels exist for Japanese titles. Consistency measures cross-language stability;
  coherence lift is about 1.1 and only 40% of neighbor slots are tagged. A consistent, aligned map can still be wrong about algorithms.
- **Benchmark bias.** The 599 test titles already carry English, so alignment on them may overstate gains on the 1,421 pure-Japanese titles.
- **Translation noise.** 1,561 titles rest on machine translation (21% exact match). Errors become arbitrary neighbors; they are flagged *degraded*.
- **Untranslated scripts.** Russian (43) and Korean (1) titles still cluster by language (0.93 same-language neighbors in t-SNE and UMAP).
- **Not all remaining Japanese clustering is bias.** Some is real, such as same-title tutorial and heuristic-contest pages.
- **Romanized non-English titles** in Latin script are undetected and treated as English.
- **Low homophily is not the goal.** C5 and C4 also lower it; the choice rests on alignment and consistency, with the random control as the guardrail.
- The choice is for title-only retrieval; statement-based similarity is untouched. No hosted service, paid API or OCI inference was used.

## Acquisition and safety record

Approved by the user on 2026-10-17: multilingual-e5-small (449 MB), paraphrase-multilingual-MiniLM-L12-v2 (449 MB) and the
opus-mt-ja-en translator (303 MB); LaBSE (1.8 GB) was declined and then downloaded by the user. All were pinned to a commit,
bounded by their published sizes and verified against the Hub's full SHA-256, then used offline. The translator ships only pickle weights:
they were kept in a quarantine folder, read once with torch's weights-only unpickler with the network blocked, re-saved as safetensors
and checked tensor by tensor. The PyPI package `sentencepiece` 0.2.1 (1.25 MB wheel, SHA-256 matched to PyPI) was installed into the research
`.venv` on approval; it is not added to the production requirements.

## Validation

- 48 new fixture tests; `research/unified_map` 43, `research/full_catalog` 48 and `tests/` 78 all pass.
- [validation.v1.json](receipts/validation.v1.json): **41 independent checks pass**: all `full-v1` artifacts byte-identical; v2 vectors equal the
  evaluated C2 vectors and a 400-text re-encode; translation provenance rules hold for every node; no machine-translated title is called reliable; exact
  neighbors equal a float64 brute force; PCA, UMAP and t-SNE re-runs reproduce the saved coordinates; per-node fidelity and the 2D island index match
  independent recomputations; headline homophily matches the stored neighbors; the selection rule re-applied to the evaluation gives the same winner;
  the HTML payload equals the saved arrays and holds no statement text; the existing map verifier and preservation snapshot pass.
- Browser checks (loopback, in-app browser): the v2 page loads 24,297 nodes, shows the language color mode, the embedded-text line and the language-bias panel,
  and the compare view and selection work.

## Process notes

An early centroid-removal run failed on a one-node language group (a zero vector) and now skips groups under 30 nodes. Several of my test expectations
were wrong and were corrected against hand calculations, not the code. An interrupted command had already deleted two superseded receipts of mine, so the
final evaluation is `evaluation.v2.json`. The earlier evaluation phases are kept as `evaluation-baseline-phase.v1.json` and
`evaluation-multilingual-phase.v1.json`; their shared numbers match the final run exactly.
