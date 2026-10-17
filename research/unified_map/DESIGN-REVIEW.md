# Unified map independent design review

Reviewed 2026-10-17 from local source and frozen artifacts. This is an AI research
review, not an independent human validation or a model promotion decision.

## Recommended evidence model

Keep one catalog identity per native key, with separate source-record aliases.
Apply only hash-bound approved overlays from
`private-learning/verification-workbench/identity-mappings/user-approved-v1.json`.
The overlay approves 36 mappings by explicit user instruction; it explicitly says
`independent_adjudication: false`. Preserve unresolved p03510 and the abc084_c
English/Japanese constraint and sample flags. Identity approval does not establish
statement fidelity or permission for public redistribution.

The catalog map can cover records lacking statements. It must expose separate
availability for title embeddings, native tags, statement neighbors and difficulty.
A missing modality is unknown, not a zero score. Retain excluded and non-Latin rows
as catalog records even if they have no MiniLM vector. Do not substitute title
neighbors for statement neighbors under the same label.

Use existing `research/full_catalog/inputs.py` A1 row selection and exact text for
reusing existing MiniLM embeddings. A1-all includes encoder-flagged non-Latin rows
and was a TF-IDF diagnostic, not a claim of equivalent MiniLM coverage. Any expanded
semantic embedding cohort needs its own explicit eligibility and language receipt.
Do not read quarantined LeetCode content or its derived legacy vectors.

## Classification provenance

`research/algorithm_system/taxonomy.v1.json` is
`ai_reviewed_not_human_gold`: mappings are editorial normalizations of native tags,
not algorithm labels derived from statements. `taxonomy.py` preserves raw tags,
relation, confidence, unknown tags and separate ancestor expansion. For primary
concept comparison use equivalent mappings with confidence >= 0.85, as in
`evaluate.py`; avoid treating related/broader mappings or ancestors as equivalent
techniques. In the frozen source snapshot all 9,787 AtCoder native tag fields are
null. Their tags remain unavailable; neither titles nor statement neighbors supply
verified classifications. Preserve null versus observed empty lists.

A hybrid should be inspectable and availability-aware. Fixed, documented rank
fusion is preferable to adding incomparable raw cosine and Jaccard scores. If tags
are unavailable, explicitly identify a semantic-only fallback. Freeze weights
before evaluation; existing labels are too sparse for reliable tuning and testing
on the same pairs. Report the tag contribution separately because evaluating a
tag-based ranker by tag consistency is circular.

## Reusable evaluation and its limits

The actual frozen labels are `research/full_catalog/labels.v1.jsonl`, with provenance
in `research/full_catalog/receipts/labels.v1.json` and the evidence rules in
`research/full_catalog/LABELING.md`. Local inspection confirms 3,647 directed pair
records over 150 queries: 2,886 non-null judgments, 761 abstentions; labels 2/3 total
44 positives over only 23 queries. Every pair is cross-platform. Preserve direction:
reverse pairs are not automatically judged. Same-platform results have no evidence
from this pool. Labels are historical AI-reviewed metadata/recollection judgments;
self-reported recognition is not statement-grounded validation or human gold.

Reuse `research/algorithm_system/evaluate.py` for explicitly qualified diagnostics:

- At K=10,20,50 report retrieved slots, judged slots, judged coverage, queries with
  judgments, observed precision among judged slots (positive means label >= 2),
  and lower/upper precision bounds assigning all unknowns negative/positive.
- Report common-judged nDCG only on the same judged-target intersection per query
  and K across compared methods, including eligible-query and common-slot counts.
  This removes unknowns before discounting: call it conditional nDCG, never ordinary
  complete-catalog nDCG@K. All-zero-gain or empty support remains undefined.
- Report query bootstrap intervals with seed/resample count and the existing
  caveat that duplicate-family independence is not established. Compare paired
  per-query deltas if claiming method differences; overlapping intervals alone do
  not establish equivalence or superiority.
- Stratify query and retrieved-slot coverage by directed platform pair, modality
  availability, and label evidence basis. Report catalog-level hubness, distinct
  targets and target-platform diversity as descriptive properties, not relevance.
- Preserve historical deceptive flags as AI flags among judged slots; they do not
  estimate independently validated error rates. Do not use unjudged/abstained pairs
  as negatives. Recall over the entire catalog is unavailable from this sparse pool.

Use two evaluation cohorts: (1) identical title/tag-eligible native catalog queries
and candidate universe for semantic, tag and hybrid methods; (2) a separately
reported statement-eligible identity-resolved cohort for statement comparisons.
Within cohort 2 use matched candidate universes when comparing rankers. A statement
run may omit LeetCode entirely; its intersection with the old pool can therefore
be tiny. Report zero support honestly rather than borrowing aggregate historical
metrics. Do not present a tiny intersection as full-map quality. Keep all-catalog
retrieval diagnostics separate from this controlled comparison.

## Verification gates

Require deterministic ordering/ties, finite vectors/scores, exact key-vector row
alignment and hashes, no native self/alias neighbors, and no duplicate native
targets. Record per-method cohort hashes, candidate filtering, encoder identity,
input text digest, taxonomy/overlay hashes and score components. Existing evaluator
checks duplicate native targets but does not replace an explicit native-self/alias
check in the map builder.

No calibrated cross-platform difficulty distance follows from native ratings or
percentiles. Report only within-platform source units, and expose missing values.
Export no private statement text, local learner data or model weights. Scientific
claims remain provisional until new blinded statement-grounded judgments with
frozen sampling support the intended cross-platform task; map construction alone
establishes neither personalization nor learning outcomes.
