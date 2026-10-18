# Language-driven clustering: investigation and correction — spec v1

Extends `interactive-map.md`. `full-v1` (vectors, neighbors, projections, HTML) and every earlier artifact stay
byte-identical; new work writes to `full-v2/` and `language-bias/`.

## Research summary (measured 2026-10-17, before any new model)

- Language groups by title script: English/Latin 22,037 (11,415 CF, 7,569 AtCoder, 3,053 LC), Japanese 2,216 (all
  AtCoder), Russian 43, Korean 1. Of the Japanese titles, 1,421 have no Latin letters, 655 are written
  "Japanese (English)", and 140 are mixed tutorial/other strings.
- The pinned English MiniLM tokenizer maps 24% of non-Latin tokens to `[UNK]`; Japanese vectors mainly encode which
  unknown/rare pieces occur, so Japanese titles group by script and shared characters, not meaning.
- The 655 natural parallel titles are a labeled cross-language benchmark that needs no translation by us.
- AtCoder has no tags, so no algorithm label exists for Japanese nodes; evaluation must not pretend otherwise.

## Requirements

- R1. THE SYSTEM SHALL compare, on the **same 24,297 nodes and the same query sets**, these candidates:
  C0 MiniLM on the raw title (baseline); C1 MiniLM with per-language centroid removal (no new model);
  C2 MiniLM on English-translated titles; C3-C5 pinned multilingual encoders on the raw title; and two controls:
  a random-vector control and a shuffled-language control. Candidate models SHALL be local, offline, pinned by
  revision and file hash, safetensors only, with no paid API, no OCI inference and no network at inference time.
- R2. English translation SHALL use, in order: the natural English parenthetical when the title has one, otherwise a
  pinned local MT model. Which source produced each translation SHALL be recorded. No translation by hand or by a
  hosted service.
- R3. Evaluation SHALL report, per candidate: (a) **translation retrieval** (Japanese half retrieves its English half;
  Recall@1/10, MRR, against chance); (b) **cross-lingual neighbor consistency@10** (a title's top-10 neighbors computed
  from its Japanese half versus from its English half, in the same index); (c) **language homophily** with its
  language-blind chance level, for Japanese, pure-Japanese, Russian and English queries; (d) **English quality
  retention** via tag-agreement lift against a platform-matched random baseline; (e) **neighbor concept coherence
  lift** for pure-Japanese queries with its tagged-slot coverage; (f) descriptive platform composition of neighbors.
- R4. Reduced language homophily or increased cross-platform mixing SHALL NOT count as improvement. The random-vector
  control SHALL be run through every metric to show what mixing without meaning scores.
- R5. Uncertainty SHALL be shown by seeded query bootstrap (200 resamples). Metrics with too little coverage
  SHALL be reported as unassessable, not as zero.
- R6. THE SYSTEM SHALL rebuild PCA, UMAP and t-SNE (all nodes, fixed seeds), exact neighbors and fidelity metrics for
  the selected representation, and record fidelity separately for Japanese and English nodes.
- R7. All model files SHALL come from an explicitly user-approved list (name, source, revision, size) and be
  hash-verified; a refusal or failure to obtain a model SHALL remove that candidate and be reported.

## Pre-registered selection rule (fixed before any new result)

Population: pure-Japanese queries (1,421) are the problem population; the 655 parallel titles are the labeled test.

1. A candidate is eligible only if all guards hold: **G1** English quality retention (tag-agreement lift divided by
   C0's lift) is at least 0.95; **G2** translation-retrieval MRR exceeds 0.25 and exceeds the random control's by
   10 times; **G3** pure-Japanese neighbor concept-coherence lift is at least 1.0 when assessable (tagged-slot
   coverage at least 0.20; otherwise reported unassessable and G3 is waived, noted in the report).
2. Among eligible candidates, the winner has the highest cross-lingual neighbor consistency@10; ties (within the
   bootstrap interval) go to the higher translation-retrieval MRR, then to the smaller model.
3. The winner replaces C0 only if its consistency exceeds C0's by at least 0.10 absolute with a bootstrap interval
   excluding zero. Otherwise C0 is retained and the report says no tested alternative was supported.
4. Language homophily and platform mixing are reported, never used in the selection.

## Design

```text
research/language_bias/languages.py   script classification, parallel-title extraction, benchmark builder
research/language_bias/models.py      pinned downloader (allowlist, hash check) and offline loaders
research/language_bias/translate.py   parenthetical-first translation with local MT, per-title provenance
research/language_bias/metrics.py     homophily, translation retrieval, consistency, coherence, bootstrap
research/language_bias/evaluate.py    runs all candidates and controls, writes receipts
research/language_bias/test_*.py      fixture tests (no private data, no models)
private-learning/unified-map/language-bias/   benchmark, translations, per-candidate vectors
private-learning/unified-map/full-v2/         selected representation: vectors, neighbors, maps, HTML
research/language_bias/receipts/              aggregate receipts (committed)
```

Existing `full_*` modules are reused for neighbors, projections, fidelity and HTML; the page template gains
optional fields (language group, text actually embedded, representation name) and still renders v1 payloads.

## Tasks

- [x] 1. Measure baseline language bias; build and freeze the parallel benchmark (no downloads).
- [x] 2. Obtain user approval for the model list, then download, pin and verify.
- [x] 3. Translate non-English titles with provenance; embed all candidates and controls.
- [x] 4. Evaluate all candidates on identical sets; apply the selection rule.
- [x] 5. Rebuild maps, neighbors, fidelity and the interactive page for the selected representation.
- [x] 6. Validate independently; notebook, report, docs; preservation check.

## Validation plan

Fixture tests for classification, parallel extraction, translation provenance, every metric (hand-computed cases,
chance levels, bootstrap determinism), the downloader's allowlist/hash refusal and the selection rule (including the
cases where the baseline must be retained). An independent verifier recomputes headline metrics from raw vectors.
Existing suites and the 501-file preservation snapshot must still pass; the v1 map files must be byte-identical.

## Risks

Parallel titles are a biased subset (they already contain English), so alignment on them may overstate gains on
pure-Japanese titles. Machine translation of short titles is noisy. No algorithm labels exist for Japanese nodes, so
concept coherence is a weak, partly unassessable proxy. A better aligned model can still be a worse algorithm
retriever; guards G1/G3 and the random control exist to catch that. Larger multilingual models are slower and may
not help.

## Result

Completed 2026-10-17; see `research/language_bias/REPORT.md`. Selected representation: **C2, MiniLM on English-translated titles** (platform
English where available, pinned local opus-mt-ja-en otherwise). Neighbor consistency 0.506 against 0.016 for the baseline; pure-Japanese
same-language neighbors 98.8% to 25.2% (chance 9.1%); English quality retained (1.008). Rebuilt maps: pure-Japanese titles in a Japanese-only 2D
neighborhood fall from 97.7% to 0.4% (UMAP) and 2.2% (t-SNE) with unchanged fidelity. 41 independent checks pass.

Deviations from this spec: LaBSE was downloaded by the user instead of by the pinned downloader; `sentencepiece` 0.2.1 was installed on a
second approval because the Marian tokenizer requires it; a one-node language group (Korean) is left out of centroid removal; robustness checks (k=5/20,
neighbor pool without parallel nodes) were added before the final run. Remaining limits: no algorithm labels for Japanese titles, noisy machine
translation, untranslated Russian/Korean, benchmark titles already carry English.
