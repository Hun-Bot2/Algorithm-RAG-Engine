# AtCoder concept classification: calibration result — 2026-10-17

**Scope actually run: a 450-problem random sample, not the full 9,613.** Extending to the whole catalog was measured and
then declined on cost. 4,308 further ABC/ARC/AGC problems (about 5.0M tokens) were exported and one wave of judges was
started, then stopped by the user before any output was written. Everything below describes the sample plus what it implies.
Spec: [atcoder-concepts.md](atcoder-concepts.md). Receipts: `research/mastery_catalog/receipts/atcoder-concepts-eval.v1.json`,
`atcoder-concepts-impact.v1.json`, `catalog-v4-refresh.json`. Asset: `assets/mastery/atcoder-concepts.v1.json` (keys and concepts only).

## Why the pilot stopped at 120

A scoping choice recorded in `design.md` section 15: cost, weak title-only evidence, and judges that share training memory.
The pilot also could not tell memory from guessing because it had no controls. This run adds them.

## Coverage (450 random eligible problems, two blind judges)

| Item status | Count | Meaning |
|---|---:|---|
| labeled (`ai_reviewed`) | 136 (30%) | both judges include a concept, neither unknown, at least one recognized |
| proposed only | 3 | agreement rests on title inference alone; visible, never trusted |
| abstained | 56 | one judge only, or disagreement |
| unknown | 255 (57%) | no label. Left unclassified, not forced |

By family the labeled rate is very uneven: ABC 90 of 122 (74%), AGC 9/20 (45%), ARC 12/44 (27%), JOI 3/24, other 22/171,
AtCoder Welcome Contest 0/42, PAST 0/14. By AtCoder difficulty: below 400 about 69% labeled, 400-1199 75%, 1200+ 43%, no
difficulty model 6%. Of the 450, 76 have machine-translated Japanese titles (degraded): 6 labeled. Projection from these
rates: about 2,650 labeled problems if ABC/ARC/AGC were completed, about 3,100 for the whole catalog. These are
extrapolations from 450 items; family cells are small (AGC 20, ARC 44).

## Reliability (no gold labels exist; nothing here is accuracy)

- Judges disagree on whether they know a task as often as they agree: kappa 0.79 on known vs unknown. Draft knew 150, review 194.
  When both knew (149), mean concept overlap was 0.82 and 139 shared a concept.
- **Invented-task decoys:** 0 of 12 recognized by either judge (95% interval 0 to 24%). The gate (at most 5%) passed on the
  point estimate, but 12 decoys cannot establish a rate below 5%.
- **Wrong-title decoys:** 3 of 12 real keys given another task's title still drew a "recognized" answer from one judge
  (interval 9% to 53%), none from both, so none became a label. A judge can follow the key and ignore a mismatched title or follow
  the title; this is a weakness that the two-judge rule absorbed, not removed.
- **Test-retest against the pilot:** only 2 pilot items fell in the sample (both agreed). Unassessable.
- Both judges are Claude models with shared training memory, so their agreement is not independent verification.

## Cost

About 1,150 tokens per problem for two judges and about 3,800 tokens per labeled problem (518k tokens for the 450-item sample,
474 shown with decoys). Projected: about 5.0M for the 4,308 remaining ABC/ARC/AGC problems, about 11.1M for the full catalog, where
most of the extra spend returns unknown.

## Recommendation impact (catalog-v3 to catalog-v4, 1,307 Codeforces/LeetCode source problems, top-5 AtCoder related)

| Condition | AtCoder problems with a trusted concept | Sources with an AtCoder candidate | Distinct AtCoder problems reached |
|---|---:|---:|---:|
| catalog-v3, concept only (before) | 77 | 93.4% | 76 |
| catalog-v4, concept only | 211 | 99.3% | 170 |
| catalog-v4 + semantic fallback | 211 | 100% | 568 |

Presence barely changed, since the same few labeled problems were recommended to everyone. The gain is breadth: the reachable AtCoder
set grew 7.5 times, 10% of candidates are semantic-only, and they rank below concept matches by design (-1.5 penalty, an unvalidated
choice). Semantic candidates are title similarity only, require reliable titles on both sides, and never feed concept statistics.
catalog-v4 was built by offline replay: neighbors identical to v3, only `problem_concepts` gained rows (+186 ai_reviewed,
+137 abstained, +4 proposed, none removed). Whether these recommendations help learners is unmeasured.

## User-confirmed vs AI-reviewed

`approved` and `rejected` exist only through `python3 -m research.mastery_catalog.atcoder_concepts confirm`, an append-only user
action validated against the catalog and vocabulary. AI judgments can only produce `ai_reviewed`, `proposed` or `abstained`.
The record is a self-attestation, as elsewhere in this repository. No user decisions exist yet.

## Limits

Sample only; 255 of 450 sampled problems stay unclassified, and the other 9,163 are unjudged. Labels reflect judge memory and can
be wrong with confidence. The invented-task gate is underpowered. Japanese-titled problems are mostly unlabeled and, when
they get semantic recommendations, are excluded unless the title is reliable. The 1.5 penalty and 5% gate are choices, not
validated values.
