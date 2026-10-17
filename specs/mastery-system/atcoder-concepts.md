# AtCoder concept classification: full catalog, metadata only — spec v1

Extends `design.md` section 8 (AI judgments) and section 6 (related practice). The 120-task pilot (MS-603,
`assets/mastery/atcoder-pilot-concepts.v1.json`) stays byte-identical; this work writes new versions.

## Why the pilot stopped at 120 (finding, verified 2026-10-17)

No technical limit. `specs/mastery-system/design.md` section 15 records the reasons: "cost and weak title-only evidence",
plus the fact that both judges are Claude models whose recognition comes from overlapping training memory. The pilot
therefore sampled 120 tasks (ABC C-F, three contest ranges) to learn the abstention rate before paying for 9,6xx tasks.
Measured at the pilot: draft judge abstained on 43/120 items, review judge on 33/120; 77/120 items kept an
ai_reviewed concept. Nothing measured how often a judge claims to recognize a task that does not exist, so the
pilot could not say whether the other 77 labels were memory or title guessing. That gap is what this spec closes
before scaling.

Downstream consequence (verified in `related.mjs` and `planner.mjs`): a problem with no trusted concept is never
proposed as related practice (`if (!shared.length && !edge) continue`) and is excluded from every exploit/explore
pool (`trustedConcepts(p).size > 0`). Only 120 of 9,613 eligible AtCoder problems carry a concept, so AtCoder is
effectively invisible to concept-driven selection.

## Requirements

- R1. THE SYSTEM SHALL classify using only catalog metadata (problem key, title, task letter) read from the
  immutable catalog-v3. No statement text, no network, no web search, no difficulty shown to the judges.
- R2. THE SYSTEM SHALL judge every eligible AtCoder problem in a seeded-shuffle order, in fixed-size batches, with two
  independent judge roles (draft, review) that never see each other's output. Ineligible problems get no label.
- R3. Each judgment SHALL carry concepts from `concepts.v1.json` (0-3), a basis (`recognized`, `title_inference`,
  `unknown`) and a non-empty rationale. `unknown` SHALL carry no concepts.
- R4. Merge rule. A (problem, concept) pair is `ai_reviewed` only when both judges include it, neither is `unknown`
  and at least one basis is `recognized`. Agreement that rests only on title inference is `proposed` (visible,
  never trusted: two readings of one title are not independent evidence). A concept only one judge includes is
  `abstained`. Items where either judge says `unknown` and no pair qualifies keep no trusted label; the item
  status (`labeled`, `proposed_only`, `abstained`, `unknown`) is recorded in the asset.
- R5. Controls. Each batch SHALL contain blind decoys: (a) fabricated tasks (invented title, contest number beyond
  the catalog's latest) and (b) real keys carrying another task's title. They SHALL be removed before merging and
  reported: false-recognition rate on (a); title-following rate on (b).
- R6. Reliability SHALL be reported with: inter-judge agreement beyond chance, abstention by contest family and
  letter, decoy rates, test-retest agreement on the 120 pilot items (fresh judges, new prompt), and a
  title-leakage check (are `recognized` rationales more specific than the title?). No gold labels exist; nothing here
  is called accurate.
- R7. THE SYSTEM SHALL distinguish evidence classes everywhere they surface: `approved` = explicit user
  confirmation, `ai_reviewed`, `proposed`, `abstained`, `rejected` (explicit user rejection). AI output SHALL never
  create `approved` or `rejected`. A user decision recorded through the `confirm` command SHALL be validated
  (concept in the vocabulary, problem in the catalog, `decided_by` = `user`, dated, append-only) and a rejected pair
  SHALL suppress an `ai_reviewed` row for the same pair in the next catalog build.
- R8. Semantic fallback. WHEN an eligible problem has no trusted concept, THE SYSTEM SHALL still consider it for
  related practice through the title-neighbor channels, labelled `Semantic title neighbor: concept not classified
  (title similarity only, not an algorithm match)`. The candidate SHALL carry a `semantic_only` component, require
  reliable titles on both sides, never feed concept statistics or exploit targeting, and rank by its inspectable
  score components. Candidates with trusted concepts keep their existing labels and ranks.
- R9. A new catalog version (catalog-v4) SHALL be built by offline replay; catalog-v1..v3 SHALL NOT be modified.
- R10. Cost SHALL be measured on a calibration slice before the full run and reported (tokens per item, per judge,
  projected and actual totals). The full run SHALL stop if the calibration gates below fail.

## Pre-registered calibration gates (fixed before any new result)

On batches 000-002 (450 items; batches are built from a seeded shuffle, so they are a random sample), controls included:

- G1 decoy false-recognition (fabricated tasks) at most 5% for each judge; otherwise `recognized` is not trusted
  for that judge and the run stops for review.
- G2 title-following rate on swapped-title items reported only (no pass/fail), because a real task may share a
  concept with another by chance.
- G3 draft/review abstention is reported, never tuned: forcing labels is rejected.

## Design

```text
research/mastery_catalog/atcoder_concepts.py   inputs, decoys, batch export, merge, confirm, evaluate
research/mastery_catalog/prompts/atcoder-full.v1.txt
private-learning/mastery/judgments/atcoder-full-v1/{in,out}/   batches and raw judge output (private titles)
assets/mastery/atcoder-concepts.v1.json                        merged rows + item statuses, no titles
hosted/mastery/related.mjs, planner.mjs                        semantic fallback (R8)
research/mastery_catalog/receipts/atcoder-concepts-eval.v1.json aggregate evaluation
```

## Validation plan

Fixture tests: decoy generation and removal, merge rule (every R4 branch), confirm/reject precedence, batch coverage
and determinism. Node tests: semantic-only candidates appear only for unlabeled problems, carry the label and
component, never create concept statistics; every existing mastery test still passes. Catalog-v4 replay equals
catalog-v3 except for the new concept rows. Recommendation impact: before/after counts of AtCoder problems
reachable by concept pools and related candidates.

## Risks

Judge recognition is training memory, not verification; two Claude judges share it. Title-only evidence for obscure
tasks is weak and the judges may confidently generalize from a title. Decoys catch invention, not subtle errors.
Japanese-only titles are translated before neighbors are computed (language-bias study), and machine translation
errors become neighbor errors; those nodes are `degraded` and excluded from semantic fallback.

## Result and deviations (2026-10-17)

See `atcoder-concepts-report.md`. R2 (every eligible problem) was not met: only the 450-problem calibration sample was judged. After the
calibration gate passed, the user chose ABC/ARC/AGC (about 5M tokens), then stopped the first wave and declined the spend, so the
remaining batches are exported but unjudged. R6 test-retest is unassessable (2 pilot items in the sample). Gate G1 passed on a
point estimate over 12 decoys, which is too few to certify 5%. R8 and R9 are implemented and tested (88 Node tests, 30 Python tests).
