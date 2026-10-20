# Mastery system requirements v1

Status: revised 2026-10-17, before implementation. This extends
`specs/adaptive-learning/` and does not replace it. ADR 003 (local SQLite
canonical, JS core as the only learner-state engine) is accepted. Decisions D-A,
D-C and D-D in `critique.md` are still open; requirements that depend on them say
so.

## Scope

- S1 The system SHALL extend the existing JS core (`hosted/core`) running locally
  through `hosted/local.mjs` on `node:sqlite`. It MUST NOT add a second
  learner-state store or policy engine.
- S2 Python SHALL be limited to offline batches (catalog refresh, embeddings,
  neighbors, AI-assisted evaluation, research). Those batches SHALL emit
  versioned, read-only artifacts with receipts.
- S3 Daily use MUST NOT require a paid API, LLM, network connection or cloud
  service. AI MAY be used only in offline batches whose outputs are versioned data
  (ADR 001 no-AI core).
- S4 Human review MUST NOT be a prerequisite for any feature. Human review MAY
  upgrade a record's status.
- S5 All policy behavior SHALL be versioned under one policy bundle
  (`mastery-policy-vN`) whose component versions are recorded in every snapshot.

## R1. Evidence ledger and append-only consistency

- R1.1 Learner evidence SHALL live in append-only ledger tables: existing
  `events` and `contest_participation`, plus new `git_observations`,
  `practice_feedback` and `catalog_switches`.
- R1.2 Every ledger table MUST share one contract:
  - `sequence` (autoincrement)
  - `entry_id` (unique)
  - `learner_id`, `schema_version`, `occurred_at`, `recorded_at`, `source`
  - `supersedes` (same table and same learner only)
  - `canonical` (canonical JSON)
  - no-update and no-delete triggers
- R1.3 Re-inserting an existing `entry_id` with the same canonical value SHALL be a
  no-op. With a different canonical value it MUST fail.
- R1.4 Derived entry IDs (Git, provider imports) MUST be deterministic functions
  of their source identity.
- R1.5 An ingestion batch and its watermark advance MUST commit in one
  transaction. A watermark MUST NOT advance past evidence that was not written.
- R1.6 Every snapshot (mission, recommendation, weekly report) MUST record its
  `event_cutoff` time and the maximum `sequence` per ledger table. Replay for
  reproduction MUST use those sequence high-water marks.
- R1.7 Replay SHALL merge ledger tables in a deterministic order:
  `occurred_at`, then table rank, then `entry_id`.
- R1.8 Derived state (ability estimates, concept status, metrics) MUST NOT be
  stored as truth. Snapshots store it only as an audit copy.
- R1.9 The v1 event envelope, its validator and existing tables MUST stay
  unchanged. Any migration MUST be additive.

## R2. Evidence capture

- R2.1 Attempts SHALL come from manual entry, Git commits (convention per D-C) and
  opt-in CF/AtCoder sync (existing).
- R2.2 Git ingestion SHALL walk all commits after the stored watermark. Rewritten
  history MUST stop ingestion for that repository and be reported.
- R2.3 A Git outcome counts as independent only with an explicit independent/no-hints
  trailer pair, valid schema-1/2 Markdown evidence, or an explicit O in a resolved daily row.
  Unmarked legacy files are `reported_completion`; daily `-` creates no attempt.
- R2.4 Self-reported minutes and confirmed timer intervals MUST be reported
  separately and never summed.
- R2.5 The system MUST NOT call LeetCode `/api/`, `/graphql`, `/submissions` or
  any other robots-disallowed path, and MUST NOT use session cookies.

## R3. Platform-specific difficulty calibration

- R3.1 The system SHALL estimate, per platform, the probability that the learner
  solves a problem independently. Codeforces uses a logistic model on the native
  rating scale. AtCoder uses AtCoder Problems IRT parameters (`difficulty`,
  `discrimination`). LeetCode uses Beta-binomial estimates per native level.
- R3.2 It MUST NOT convert one platform's difficulty into another's. Each platform
  has its own learner ability estimate.
- R3.3 Practice ability SHALL be estimated from practice outcomes only. Contest
  ratings MAY seed the prior and MUST NOT be overwritten by practice estimates.
- R3.4 Every recommendation SHALL store the predicted probability it used.
  Outcome metrics MUST use that stored prediction, not a recomputed one.
- R3.5 Calibration quality (Brier score against a constant-rate baseline) SHALL be
  reported per platform once at least 30 predictions have outcomes.
- R3.6 A problem without a difficulty value or model SHALL have predicted
  probability `unknown`. It MAY appear only in exploration slots.

## R4. Exploration and exploitation

- R4.1 Each practice day SHALL be planned in this order: due reviews, then exploit
  slots, then at most one explore slot. Explore minutes MUST NOT exceed 25% of the
  week's practice minutes.
- R4.2 Exploit slots SHALL target the concept with the highest learning need (a
  Beta posterior over independent outcomes) and a problem whose predicted
  probability is in the target zone (default 0.60-0.80).
- R4.3 Explore slots SHALL rotate on a fixed calendar schedule that does not
  depend on learner outcomes. The kinds are:
  - uncertain concept
  - stretch probe (predicted probability 0.35-0.55)
  - related-practice channel
  - newly added catalog problem
- R4.4 Every slot MUST record its kind, explore sub-kind, score components,
  predicted probability and policy version.
- R4.5 A concept in remediation (a failed or assisted review) SHALL replace the
  explore slot with remediation.
- R4.6 A cold start (fewer than 6 known-outcome practice attempts on a platform)
  SHALL use a placement pattern spanning low, target and stretch difficulty.

## R5. Related cross-platform practice

- R5.1 Candidates SHALL be scored from inspectable components: concept overlap,
  reviewed relationship, title-embedding rank, calibrated difficulty fit and
  exposure recency.
- R5.2 A candidate with zero concept overlap and no relationship MUST NOT be
  proposed from an embedding channel alone.
- R5.3 Candidates MUST be labeled by evidence status:
  `reviewed relationship`, `AI-reviewed concept match` or
  `unverified candidate`. They MUST NOT be described as algorithmically
  equivalent.

## R6. Spaced review and duplicate prevention

- R6.1 Existing review intervals and remediation rules SHALL apply unchanged.
- R6.2 New-practice candidates MUST exclude problems attempted, imported or
  exposed within the cooldown (default 30 days), resolving identity aliases first.
- R6.3 A virtual contest MUST NOT be proposed if any of its problems has been
  attempted or imported.

## R7. Weekend contests

- R7.1 The system SHALL list fitting live CF and AtCoder contests in the learner's
  weekend window (Asia/Seoul), each labeled with a fit reason. It SHALL also
  propose one virtual contest with score components.
- R7.2 Proposals SHALL stay proposals until the learner accepts them (existing
  `suitable` semantics).
- R7.3 Missing tag data (all AtCoder) SHALL make weakness coverage `unknown`, not
  zero.

## R8. Measurable learning outcomes

- R8.1 The system SHALL compute these metrics (definitions in `design.md`
  section 7): independent solve rate, difficulty-adjusted excess solve rate,
  ability trajectory per platform, transfer success rate and transfer lift,
  review retention per interval stage, and contest performance (rating change,
  solved versus expected solved, upsolve completion).
- R8.2 Each metric MUST report its numerator, denominator, window and a 95% Wilson
  interval or equivalent. Below its minimum sample it MUST show
  `insufficient_evidence`.
- R8.3 Metrics MUST NOT be presented as causal effects of the system.

## R9. AI-assisted and solving-based evaluation

- R9.1 AI MAY draft and independently review concept crosswalks, AtCoder concept
  assignments, relatedness judgments and archive `practice_for` links. Agreeing
  reviews yield status `ai_reviewed`. Disagreement yields `abstained`.
- R9.2 `ai_reviewed` records MAY be used by policy. They MUST be labeled as such
  and MUST NOT be called human-verified or gold.
- R9.3 AI MUST NOT resolve problem identity (`same_problem`) and MUST NOT read
  quarantined LeetCode statements.
- R9.4 Every AI judgment MUST record its model, run date, prompt hash, input hash,
  role and rationale. Empty rationales MUST be rejected.
- R9.5 Solving outcomes and learner verdicts SHALL be the primary evaluation
  signal. AI judgments are secondary. An evaluation MUST NOT use labels derived
  from the same signal it evaluates.

## R10. Weekly report

- R10.1 Every Sunday, or on demand, the system SHALL produce a deterministic,
  immutable weekly report (Monday-Sunday, Asia/Seoul) covering:
  - activity
  - review
  - outcome metrics (4-week rolling plus the current week)
  - calibration
  - weaknesses
  - contests
  - evaluation tallies
  - data quality
- R10.2 It SHALL be writable as Markdown into a learner-chosen repository. The
  system never commits or pushes.

## R11. Incremental catalog updates

- R11.1 A weekly opt-in batch SHALL fetch CF and AtCoder problem, contest and
  problem-model metadata through documented endpoints only, with existing rate
  gaps (decision D-D).
- R11.2 Each run SHALL produce catalog version N+1 from a delta against version N.
  The delta is keyed by problem key and normalized-row hash, and lists added,
  changed, removed and unchanged rows with a receipt. Version N MUST NOT be
  modified.
- R11.3 Embeddings SHALL be computed only for added or changed input text, reusing
  a cache keyed by (text hash, model revision).
- R11.4 Neighbor tables in version N+1 MUST equal an exact full recomputation.
  Any incremental shortcut MUST be tested for equality with full recomputation.
- R11.5 Upstream removals SHALL be kept as `removed_upstream` rows so historical
  evidence still resolves.
- R11.6 Switching the active catalog version SHALL be an explicit, logged
  `catalog_switches` entry.
- R11.7 LeetCode rows SHALL be added only from learner-authored entries
  (`provenance = user_entered`).

## R12. Provenance-aware exam and interview archive

- R12.1 Archive records SHALL be kept apart from platform catalogs, in versioned
  learner-authored files compiled into the catalog database.
- R12.2 Each record SHALL carry:
  - `origin`: organization, exam, year, session, question label
  - `source`: kind, citation, URL, access date
  - `acquisition` method
  - `rights_status`
  - `confidence`
  - `recorded_by` (learner, or AI draft)
  - optional private copy hash
- R12.3 Third-party statement text MUST NOT be stored unless `rights_status` is
  `own_work` or `licensed`. Learner notes are allowed and marked `own_work`.
- R12.4 Corrections SHALL supersede records in a new archive version. Earlier
  versions MUST remain readable.
- R12.5 Links default to `practice_for` or `concept_only`. A `same_problem` link
  MUST cite two verifiable sources and MUST NOT be created by AI alone.

## R13. Filtering

- R13.1 Filters SHALL cover platform, concept (with evidence status), native tag,
  native difficulty, predicted probability, learner status, similarity
  (neighbor of a key by channel and rank), source, archive category and
  provenance, and date.
- R13.2 Each date filter MUST name its basis: attempt time, contest start or
  first-seen catalog version.
- R13.3 Filters SHALL be read-only SQL views plus a CLI before any UI.

## Non-goals

- Online-judge execution, AI-generated problems and learned mastery models.
- Multi-user support.
- Any statement collection rejected on 2026-10-16.


## Daily study-log compatibility

- Daily date-named Markdown files SHALL be interpreted as multi-problem tables, never as problem IDs.
- An explicit O/X per row SHALL be independent/failed evidence respectively; an unrecorded `-` SHALL NOT create an attempt.
- Problem identity SHALL resolve from exact catalog titles or supported original URLs, scoped by platform where available. Unknown or ambiguous identities SHALL be flagged and skipped.
- Repository/canonical-problem/date SHALL identify one daily attempt. Memo edits SHALL be idempotent; changed outcomes SHALL be flagged without rewriting prior evidence.
- CLI and local polling SHALL use the active verified catalog and the same ingestion path. Legacy formats SHALL remain compatible.
