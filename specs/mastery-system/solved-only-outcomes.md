# Solved-only outcomes (policy mastery-policy-v3, 2026-10-28)

Decision record: [ADR 005](../adaptive-learning/adrs/005-solved-only-outcomes.md).

## Requirements

- SO1 THE SYSTEM SHALL classify an attempt as solved WHEN its stored outcome is `independent`, `assisted` or
  `reported_completion`, and as unsolved WHEN it is `failed`, regardless of hints.
- SO2 The classification MUST happen in one place (`attemptStream`); planner, calibration, progression, reviews,
  weekly metrics, mock contests and contest fit MUST NOT re-read raw outcomes.
- SO3 Stored events, Git observations, daily-log rows and backups MUST NOT be rewritten. Each projected attempt
  SHALL keep `recorded_outcome` and `recorded_hints`.
- SO4 The UI, CLI help text and recording forms SHALL offer solved / unsolved only and MUST NOT display or ask
  about hints or help.
- SO5 The policy bundle SHALL change to `mastery-policy-v3`, so new plans are distinguishable from v2 plans.

## Design

`attemptStream` maps the stored outcome through `SOLVED = {independent, assisted, reported_completion}` to
`outcome: 'independent', hints: false`. Failed attempts stay `outcome: 'failed'`, `hints: null`. All existing
success checks (`outcome==='independent' && hints===false`) therefore mean "solved". Raw-table readers that build
the superseded-history display in the journal apply the same mapping for display.

Recording: the attempt editor sends `independent`/`hints:false` for solved and `failed`/`hints:null` for unsolved.
The daily log keeps its format (`1`, `0`, legacy `O`/`X`); ingestion is unchanged.

## Tasks

- [x] T1 Projection in `attemptStream` with `recorded_outcome`/`recorded_hints`; policy bundle v3.
- [x] T2 UI: solved/unsolved labels, symbols, analytics, editor without hints; contest-fit wording.
- [x] T3 Tests updated to v3 semantics; new projection test.
- [x] T4 Docs: AGENTS.md, runbook.

## Validation

`node --test hosted/test/*.test.mjs`; the running local server is restarted and today's `1` rows are checked to
appear as solved.
