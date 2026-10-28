# Weekly LeetCode mock contest: requirements (2026-10-28, draft)

Scope: an explicitly generated, contest-shaped set of 5-7 free LeetCode problems, matched to the learner's
LeetCode level estimate and shown on the calendar at a weekend slot. The system selects and records only.
Solving and judging happen on leetcode.com. No judge, statement collection, company data or title embedding is part of this scope.

## R1 Generation

- R1.1 WHEN the learner runs the explicit generate command for a date, THE SYSTEM SHALL produce one mock contest
  for that Asia/Seoul civil date. Viewing the calendar, journal or plan MUST NOT generate one.
- R1.2 The problem count SHALL be an integer in [5, 7], default 6. The solving time SHALL default to 120 minutes for
  any count and MAY be overridden by an integer in [30, 300] minutes. A mock contest has no start or end time.
- R1.3 Generation SHALL be deterministic: the same date, learner evidence, preferences, catalog version and policy
  version MUST produce byte-identical output.
- R1.4 A generated mock contest SHALL be immutable. A second generate for the same learner and date MUST return the
  stored record unchanged and report that nothing changed.

## R2 Candidate pool

- R2.1 Candidates SHALL be active-catalog LeetCode problems that are eligible, have access `free` or
  `free_at_collection`, have a native `lc_level` difficulty in {1, 2, 3}, and have at least one native tag.
  (catalog-v4 has 61 eligible untagged LeetCode rows: JavaScript, Pandas and SQL exercises. Found 2026-10-28.)
- R2.2 A candidate MUST NOT have any recorded attempt by the learner (any outcome, any source), after alias
  canonicalization.
- R2.3 A candidate MUST NOT appear in a daily plan snapshot or another mock contest within the configured
  `cooldown_days` before the date.
- R2.4 Title text, title embeddings and similarity-map neighbors MUST NOT influence candidate selection or ordering.

## R3 Difficulty fit

- R3.1 Composition (counts of Easy, Medium, Hard) SHALL be chosen from the existing LeetCode level calibration
  (`fitAll`/`predict`, conservative probability = mean minus one SD).
- R3.2 The system SHALL report expected solved count, the target it aimed for, and each problem's conservative
  and mean probability.
- R3.3 WHEN the LeetCode calibration status is `insufficient_evidence`, THE SYSTEM SHALL still use the calibration
  priors, and MUST label the contest as based on insufficient evidence. It MUST NOT substitute Codeforces or
  AtCoder evidence.
- R3.4 The target ratio and composition constraints are an unvalidated practice heuristic and SHALL be recorded
  with the policy version, not presented as a performance prediction.

## R4 Selection within a level

- R4.1 Within a level, candidates SHALL be ranked by coverage of the learner's current weak concepts, using only
  trusted concepts (`ai_reviewed` or `approved`). Unclassified problems contribute zero and are reported as unknown.
- R4.2 Two selected problems MUST NOT have the same non-empty native tag set.
- R4.3 Remaining ties SHALL be broken by a deterministic seeded order of (learner_id, date, problem_key), not by
  alphabetical order alone.
- R4.4 Every selected problem SHALL carry its score components (level, probabilities, weak concepts covered, tags)
  and a plain reason.

## R5 Calendar placement

- R5.1 The default date SHALL be the next date whose `weekly_rhythm` day is `virtual`; an explicit date MAY be given.
- R5.2 A mock contest SHALL be placed by date only, with its solving time. It MUST NOT read, depend on or be adjusted
  for official contest schedules. (2026-10-28 user decision: mock and real contests are separate; v1 time-slot
  placement is removed in policy `mock-contest-v2`.)
- R5.3 The calendar SHALL show the mock contest as its own layer, visually distinct from official contests and
  daily recommendations, with its solving time and every problem's LeetCode link, level and reason. It MUST NOT show
  a start or end time, including for stored v1 records that carry `start_at`/`placement_note`.

## R6 Evidence and results

- R6.1 A mock contest MUST NOT create, modify or imply attempts. Results are recorded through the existing daily log,
  Git ingestion or manual attempt paths.
- R6.2 The calendar MAY show a per-problem result by matching recorded attempts for the same canonical problem key on
  or after the mock contest date. This match is presentation only and MUST NOT be stored as evidence.
- R6.3 Attempts made during a mock contest SHALL be ordinary practice evidence. No new attempt field or activity
  type is added.
- R6.4 Daily plans generated after a mock contest exists MUST NOT recommend its problems as new practice within
  `cooldown_days` (R2.3 applied in the other direction).

## R7 Storage and safety

- R7.1 Mock contests SHALL be stored as `recommendation_snapshots` rows with kind `mock_contest`. No schema migration
  is required; existing immutability triggers and backup archives apply.
- R7.4 A backup restore SHALL keep mock contests as inert `archive-...` rows (backup-v2 rule). Restored mock contests
  SHALL be shown on the calendar as archived history and MUST NOT act as exposures; export SHALL unwrap them unchanged.
- R7.2 Tests MUST use `:memory:` or temporary databases with synthetic learners. The real learner DB MUST NOT be
  initialized, migrated or written by tests or smoke runs.
- R7.3 The new selection module SHALL be pure and Worker-safe (no Node-only imports), like `contests.mjs`.
