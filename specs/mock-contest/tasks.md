# Weekly LeetCode mock contest: tasks (2026-10-28, draft)

Each task is independently testable. Tests use synthetic catalogs and `:memory:` or temporary databases only (R7.2).
Run `node --test hosted/test/*.test.mjs` after every task.

- [x] T1 Pure composition: `composeLevels({abilities, pools, count})` in `hosted/mastery/mock-contest.mjs`.
  Tests: prior-only worked example gives (5,1,0); evidence example gives (1,4,1); tie prefers harder; pool limits
  respected; impossible pools throw; count outside [5,7] throws.
  Done 2026-10-28: `hosted/test/mastery-mock-contest.test.mjs` (6 tests); full suite 220/220 passed.
- [x] T2 Pool and selection: `selectProblems(...)` with R2 filters, weakness score, tag-set diversity, FNV seeded
  tie-break. Tests: attempted/alias/cooldown/premium/ineligible exclusion; identical tag sets rejected then relaxed
  with a note; unclassified coverage reported as unknown; byte-identical output for identical input; output does
  not change when only titles change (R2.4).
  Done 2026-10-28: `candidatePool`, `selectProblems`, `fnv1a`; 4 tests, full suite 224/224. Read-only check on
  catalog-v4 with a synthetic learner: pools 735/1,498/716, prior-only set Easy 5 + Medium 1, no relaxation notes.
- [x] ~~T3 Placement~~ (removed in v2, 2026-10-28: mock contests carry a date and solving time only): `placeStart(...)` in Asia/Seoul. Tests: window start when free; shift after an overlapping
  contest; no fit gives `start_at: null` with reason; non-weekend date gives date only; KST midnight boundary.
  Done 2026-10-28: `placeStart`; 5 tests (half-open boundaries, unknown durations, invalid input), full suite 229/229.
- [x] T4 Record builder `mockContest(...)` combining T1-T3 with policy version, ability status and inputs.
  Tests: `insufficient_evidence` label (R3.3); no Codeforces/AtCoder evidence used; no Node-only imports
  (static import check like the existing core/worker boundary tests).
  Done 2026-10-28: `mockContest`; ability from LeetCode practice observations only, record adds `learner_id`,
  `evidence_label` and `inputs` (attempt IDs, exposure dates, pool sizes, exclusions, busy contests). 4 tests, full suite 233/233.
- [x] T5 CLI `mock-contest create|show`: default next `virtual` day, `--date/--count/--minutes/--json`, stores
  `recommendation_snapshots` kind `mock_contest`; second create returns stored record with `changed: false`.
  Tests in `mastery-cli.test.mjs` with a temporary DB: immutability, attempts table unchanged (R6.1).
  Done 2026-10-28: exposures = plan snapshots + earlier mock contests; weak concepts from the weekly report;
  contests from the provider cache. `show` returns 1 when absent. 1 CLI test, full suite 234/234. Runbook updated.
- [x] T6 Planner exposure (R6.4): `plan` includes mock contest problems in prior exposures.
  Test: a problem in a mock contest dated within cooldown is never recommended by `planDay` afterwards.
  Done 2026-10-28: `mockExposurePlans` feeds CLI `plan` and the journal planner; `snapshotExposures` makes
  `mock-contest create` count every snapshot kind (CLI plans, journal recommendations, mock contests; not archives).
  Found while wiring: the journal treated every snapshot row as a plan and iterated `slots`, so a stored mock contest
  would have broken `/api/journal`. The journal now keeps mock rows out of plan history. 3 tests, full suite 237/237.
- [x] T7 Journal and calendar projection: journal exposes `mock_contests`; `workspaceEvents` adds layer `mock`
  with slots and presentation-only result matching (R6.2). Tests in `workspace-data.test.mjs`: layer
  separation, superseded attempts ignored, attempts before the mock date ignored.
  Done 2026-10-28: journal returns `mock_contests`; `workspaceEvents` adds one `mock` event per contest with
  per-problem `result` (earliest active attempt on/after the date) and `eventCategory` = `mock`. The layer stays
  hidden until T8 adds its toggle, because `filterCalendarEvents` shows only enabled categories. 2 tests, suite 238/238.
- [x] T8 UI: render the mock layer distinctly in `workspace.mjs`/`styles.css` with links, level, reason and
  start time; add any new browser module to the `local.mjs` allowlist. Tests: `ui-contract.test.mjs`;
  targeted browser check with a disposable synthetic learner (screenshots private).
  Done 2026-10-28: `mock` layer toggle (on by default), calendar chip with start time, agenda/day sections, mobile dot,
  detail dialog (mix, expected solved, evidence notice, LeetCode links, matched results, per-problem record button).
  No new browser module, so the `local.mjs` allowlist is unchanged. Browser check: synthetic `browser-smoke` learner in a
  scratchpad DB on port 8790 with the read-only catalog-v4. Desktop and 375px views were checked; there was no
  horizontal scroll and no console errors. The real catalog run exposed 61 untagged JS/Pandas/SQL LeetCode rows, so R2.1
  now requires a native tag (`excluded.untagged`). UI contract test added; suite passes.
- [x] T9 Backup round trip: a `mock_contest` snapshot survives backup-v2 export/import unchanged.
  Done 2026-10-28: import wraps it as an inert `archive-...` row (original_kind `mock_contest`); re-export is byte-identical.
  The journal now lists archived mock contests with `archived: true` (calendar id `mock-archive:<date>`, archived notice)
  and leaves them out of exposures, matching archived plans (R7.4). 3 tests, suite 237/237.
- [x] T10 Docs: runbook commands, AGENTS.md section (gotchas found during T1-T9), this file's checkboxes.
  Done 2026-10-28.

## Validation plan

- Unit: T1-T4 pure-module tests, deterministic fixtures with hand-computed expectations.
- Integration: T5, T6, T9 against temporary SQLite with migrations applied.
- Presentation: T7 data tests, T8 UI contract plus one targeted browser run (not a full smoke: the general
  browser smoke has expired October contest fixtures).
- Not covered: whether the 0.6 target produces good practice outcomes; that needs real usage over weeks.

## Open questions for the learner

- Default count 6 and 20 minutes per problem: keep?
- Should mock-contest results ever be separated from practice calibration (would need a schema change)?
- Reroll for the same date: wanted in a later version?

## v2 change (2026-10-28)

- [x] T11 Remove time-slot placement: no `placeStart`, no contest schedule input, `duration_minutes` default 120 for
  any count, policy `mock-contest-v2`. Calendar shows date and solving time only, also for the stored v1 record.

