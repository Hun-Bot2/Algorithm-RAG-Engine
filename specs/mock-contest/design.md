# Weekly LeetCode mock contest: design (2026-10-28, draft)

Requirements: [requirements.md](requirements.md). Tasks: [tasks.md](tasks.md).

## 1. Context (checked in code, 2026-10-28)

- The calendar ([workspace-data.mjs](../../hosted/public/workspace-data.mjs)) projects journal entries into layers
  `actual`, `plan` and `contest`. Contests come from the provider cache; plans from immutable snapshots.
- `recommendation_snapshots` (migration 008) is keyed by (learner_id, kind, local_date) with no-update/no-delete
  triggers. `kind` is free text; backup-v2 archives every kind. A new kind needs no migration.
- `planDay` already handles `virtual` rhythm days by emitting one `contest` slot with `contestProposals`
  (Codeforces/AtCoder live contests and one `virtualProposal`). LeetCode has no contest grouping in catalog-v4.
- catalog-v4 LeetCode: 3,053 problems (2,949 eligible), all `free_at_collection`; difficulty is only
  `lc_level` 1/2/3 (792 / 1,538 / 723 rows). Premium problems were never collected.
- LeetCode calibration (`fitLeetCode`) is a per-level Beta model with priors E 3.2/0.8, M 2/2, H 0.8/3.2 and
  `MIN_EVIDENCE = 6`. `predict` returns `p_mean` and `p_conservative = max(0, mean - sd)` per level.
- Weak concepts come from `buildWeeklyReport(...).weaknesses`, the same input `virtualProposal` receives.
  `trustedConcepts(problem)` returns only `ai_reviewed`/`approved` concepts.
- Attempts with `activity_type` other than `review` already count as calibration observations.

## 2. Architecture

```text
cli.mjs  mock-contest create [--date D] [--count N] [--minutes M] [--json]
   |  reads: ledger (cutoff now), active catalog, preferences, plan + mock_contest snapshots,
   |         weekly report weaknesses
   v
mastery/mock-contest.mjs   (pure, Worker-safe; no Node imports)
   composeLevels() -> selectProblems() -> record
   |
   v
recommendation_snapshots (kind = 'mock_contest', local_date = D)   immutable
   |
   v
journal.mjs exposes mock_contests  ->  workspace-data.mjs layer 'mock'  ->  workspace.mjs UI
```

Planner change (R6.4): `cli.mjs plan` passes mock contest problems to `planDay` as additional prior exposures,
so the existing `exposures()` cooldown excludes them. `planDay` ranking logic is otherwise unchanged.

## 3. Algorithm (policy `mock-contest-v2`; selection is unchanged from v1)

### 3.1 Pool (R2)

`platform = leetcode`, `eligible`, `access in {free, free_at_collection}`, `difficulty_kind = lc_level`, non-empty native tags,
canonical key not attempted, not in plan or mock snapshots dated in `[date - cooldown_days, date)`.

### 3.2 Composition (R3)

For count N, enumerate (e, m, h) with e + m + h = N, e >= 1, m >= 1, and each level count not exceeding its
pool size. Expected solved E = e*pE + m*pM + h*pH using `p_conservative`. Target T = 0.6 * N.
Choose the minimum |E - T|; ties prefer larger h, then larger m (harder sets win ties).

Worked check with priors only (no evidence; values from `predict`, verified 2026-10-28): pE 0.621, pM 0.276,
pH 0.021, N = 6, T = 3.6 gives (5, 1, 0), E = 3.381. With pE 0.95, pM 0.60, pH 0.20 it gives (1, 4, 1), E = 3.55. The 0.6 ratio and the e >= 1, m >= 1
shape constraints are an unvalidated heuristic recorded under the policy version (R3.4).

### 3.3 Selection inside a level (R4)

Score = |trustedConcepts(p) intersect weak| (0 when unclassified; reported as `weakness_coverage: unknown`).
Order: score desc, then FNV-1a 32-bit hash of `learner_id|date|problem_key` asc, then key asc.
Greedy pick: skip a candidate whose sorted non-empty tag set equals an already selected one's.
If diversity leaves a level short, relax the tag rule for that level and record a note (never change levels).

Final order: Easy, Medium, Hard; inside a level, by the selection order. Slots are numbered 1..N.

### 3.4 Placement (R5)

Policy `mock-contest-v2` (2026-10-28): date plus solving time (default 120 minutes) only. No start time, and official
contest schedules are not read. The v1 text below is historical: the stored 2026-10-31 v1 record keeps its
`start_at: null` and `placement_note`, and the UI ignores both.

v1 (removed):

Date: explicit `--date`, else the next date (today included) whose rhythm day is `virtual`.
Window: `weekend_window.sat` or `.sun` for that weekday (Asia/Seoul); other weekdays have no window.
Busy intervals: cached contests (the same schedule the calendar shows) intersecting that date.
Start: the earliest of window start and each busy-interval end such that [start, start + duration] lies inside
the window and overlaps no busy interval. None found: `start_at = null` with a reason.
Intervals are half-open, so a set may end exactly when a contest starts or the window closes. A cached contest
without a positive duration blocks from its start to the window close and adds a note (legacy cache safety).
Placement does not look at the current time, so the record stays deterministic (R1.3).

### 3.5 Record

```json
{
  "policy": "mock-contest-v1", "catalog_version": "catalog-v4", "local_date": "2026-10-31",
  "duration_minutes": 120,
  "ability": {"status": "estimated", "n": 14, "levels": {"1": {}, "2": {}, "3": {}}},
  "composition": {"easy": 1, "medium": 4, "hard": 1}, "target_expected": 3.6, "expected_solved": 3.55,
  "evidence_label": null, "learner_id": "...",
  "weak_concepts": ["..."],
  "problems": [{"slot": 1, "problem_key": "leetcode:...", "title": "...", "url": "https://leetcode.com/problems/...",
    "level": 1, "tags": ["..."], "concepts": ["..."], "p_mean": 0.95, "p_conservative": 0.95,
    "components": {"weak_concepts_covered": ["..."], "weakness_coverage": 1}, "reason": "..."}],
  "notes": [], "inputs": {"count": 6, "attempt_ids": ["..."], "exposure_dates": ["..."], "pool_sizes": {"1": 0, "2": 0, "3": 0},
    "excluded": {"attempted": 0, "exposed": 0}}
}
```

`title` and `url` come from the catalog for display only; they never affect selection (R2.4).

## 4. Data flow for results (R6)

The learner solves on leetcode.com and records outcomes as usual (daily log O/X, Git trailers, manual attempt).
`workspace-data.mjs` joins a mock slot with the first non-superseded attempt for its canonical key whose
`occurred_at` local date is >= the mock date, and shows that outcome. Nothing is written back.

## 5. Tradeoffs and rejected alternatives

- Title embeddings / similarity neighbors: rejected. Inaccurate for algorithm matching (user decision, and
  AGENTS.md notes title neighbors mismatch technique).
- Self-hosted judge: rejected. Out of product scope; LeetCode statements and tests are not available or permitted.
- New ledger table or migration: rejected. Snapshot kind already gives immutability and backup coverage.
- Separate `mock_contest` activity type for attempts: rejected for v1. Would require event/side-ledger schema
  changes; contest-condition solves are still valid practice evidence. Revisit if timed results should be analyzed apart.
- Regeneration (reroll) for the same date: not in v1 because the snapshot key is (learner, kind, date). A later
  version could use `mock_contest:<n>` kinds with explicit supersession.
- Finer difficulty (community LeetCode contest ratings, e.g. zerotrac): out of scope until its license and coverage
  are reviewed and approved as a new data source.
- Company-weighted problem mix: possible later extension via Topics profiles; out of scope here.

## 6. Risks

- Coarse levels: Medium spans a wide range, so expected solved can be far off. Mitigated by disclosure only.
- Sparse evidence: priors drive composition until 6 LeetCode observations exist; flagged per R3.3.
- Pool exhaustion: unlikely (2,949 eligible). Composition never exceeds a level's pool size (3.2); when no valid
  (e, m, h) exists, generation fails with a clear error instead of storing a partial contest.
