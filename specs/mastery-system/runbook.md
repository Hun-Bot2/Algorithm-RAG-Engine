# Mastery system runbook

Local-first (ADR 003). Everything below runs offline unless marked NETWORK. The learner database
is `private-learning/hosted.sqlite3` (shared with `npm run dev`); catalogs live in
`private-learning/mastery/<version>/`. Nothing here commits or pushes.

## First use

```bash
node hosted/cli.mjs catalog use --version catalog-v3
node hosted/cli.mjs repo add --id solutions --path ~/code/algorithm-solutions
node hosted/cli.mjs prefs --mastery '{"target_concepts":["dynamic-programming","graph-traversal"]}'
```

Optional contest priors: `node hosted/cli.mjs rating set --platform atcoder --rating 1100 --participations 12`
or NETWORK (one Codeforces API call) `node hosted/cli.mjs rating sync-cf --handle <handle>`.

## Daily (Monday-Friday)

```bash
node hosted/cli.mjs repo ingest
node hosted/cli.mjs plan
node hosted/cli.mjs attempt --problem codeforces:1352C --outcome independent --hints none
node hosted/cli.mjs feedback --plan 2026-10-21 --source leetcode:coin-change --candidate codeforces:283C --verdict related --channel title_v2_cross
```

Git trailers are optional (`Problem:`, `Outcome:`, `Hints:`, `Minutes:`, `Activity:`, `Transfer-From:`).
Without them a commit counts only as reported completion, and only the first commit that touches a
problem counts. `repo ingest` stops with exit code 3 on rewritten history; re-anchor explicitly with
`--anchor <sha>`.

## Weekend and weekly

```bash
node hosted/cli.mjs plan                 # Saturday: live-contest annotations plus one virtual proposal
node hosted/cli.mjs weekly --out ~/code/algorithm-notes/reports
```

A weekly report is frozen only after the week ends; the Markdown file is never overwritten.

LeetCode mock contest (specs/mock-contest/): 5-7 unattempted free problems matched to the LeetCode level estimate,
on a date with a 2-hour solving time. It has no start time and is independent of official contest schedules.
One immutable set per date; solve and judge on leetcode.com.

```bash
node hosted/cli.mjs mock-contest create                  # next virtual day, 6 problems, 120 minutes
node hosted/cli.mjs mock-contest create --date 2026-10-31 --count 7 --minutes 150
node hosted/cli.mjs mock-contest show --date 2026-10-31
```

## Weekly catalog refresh (NETWORK, approved)

```bash
.venv/bin/python -m research.mastery_catalog.refresh --previous catalog-v3 --version catalog-v4 --live \
  --public-receipt research/mastery_catalog/receipts/catalog-v4-refresh.json
node hosted/cli.mjs catalog use --version catalog-v4
```

The refresh makes six rate-gated GETs (Codeforces API and AtCoder Problems resources), stops on
403/429, keeps the previous version, encodes only new titles, and recomputes exact neighbors.
Manual LeetCode or other additions: `--manual-additions file.json` with `{"rows": [...]}`
(provenance `user_entered`). Archive: `--archive archive.json` (see
`assets/mastery/archive.example.v1.json`; keep real archives private).

## Filtering

```bash
node hosted/cli.mjs find --platform codeforces --concept dynamic-programming --difficulty 1200..1500 --status unseen
node hosted/cli.mjs find --near leetcode:coin-change --channel title_v2_cross --max-rank 10
node hosted/cli.mjs find --date contest:2025-01-01..2025-12-31 --platform atcoder
```

Date filters must name their basis (`attempt:`, `contest:`, `first_seen:`).

## Offline evaluation (optional)

`python3 -m research.mastery_catalog.relatedness export|merge|agreement` (see its docstring).
AI judgments are recorded as `ai_reviewed`, never gold; learner verdicts are the primary signal.


## Daily Markdown study logs

Commit `leetcode/YY-MM-DD.md`, `atcoder/YY-MM-DD.md`, `codeforces/YY-MM-DD.md`,
or `daily/YY-MM-DD.md`. The first heading is `# YYYY-MM-DD` and the summary is:

```md
| 문제 | 결과 | 메모 |
| --- | --- | --- |
| Two Sum | O | optional note |
| https://codeforces.com/problemset/problem/4/A | X | |
```

Problem names, pasted URLs, and Markdown links to original problem URLs are accepted.
The pasted `[|Title](URL)|O|memo|` form is also accepted without editing the source.
Titles match exactly after case/whitespace normalization within the directory's platform;
`daily/` searches all platforms. No fuzzy matching or title-to-slug guessing is used.
URL identities must also exist in the active, hash-verified catalog. Ambiguous/unresolved
rows produce flags with file/line and are skipped. Activate a catalog before ingestion;
without it ingestion throws before advancing the watermark and can be retried.

O declares independent success (hints false); X declares failed/non-independent
(hints unknown). `-` remains unrecorded: it produces a flag and no attempt, rather
than inventing completion evidence. Memo text and detailed sections do not infer outcomes.
Daily rows use their own outcome, never a commit-wide trailer. Duplicate rows collapse;
conflicting outcomes for the same problem in a table are skipped with a flag.

Identity is repository + canonical problem + date, independent of row order, memo and file
location. Note edits and repeated polling are idempotent; the same problem on a new date
is a new attempt. An outcome change for an imported date is flagged, never overwritten.
Daily logs record only a date: `occurred_at` uses the Seoul date boundary (clamped to ingest
now), not a claimed solve time. Provenance retains source SHA, path, row, date and catalog
version (oversized source paths/rows retain a hash, with originals in the source commit). Legacy schema 1/2 files and trailer-based imports stay supported.

Run `repo ingest` followed by `journal` for an evidence-refreshed recommendation.
The legacy `plan` command freezes a date's first plan and does not refresh an existing one.
No backfill, learner-DB migration, or repair of previously misidentified date-file rows is
automatic. To import already-watermarked daily files, use a reviewed explicit anchor;
inspect any existing wrong IDs separately instead of silently deleting history.

## Dual-model Korean explanations

Use `node hosted/cli.mjs explain` with Gemini and NVIDIA credentials/model IDs to
concurrently generate both explanations from the same frozen evidence. This is the
default generation path; web/journal reads never dispatch. Durable attempt records,
independent caches, retry/usage/cost reporting and setup:
[explanation runbook](../recommendation-explanations/runbook.md).

Optional timing: use `문제 | 결과 | 시간 | 메모` with a four-column separator.
Write minutes as `10m` or seconds as `40s` (nonnegative integers, maximum 1440m / 86400s); empty or `-` means unrecorded, and `0m`/`0s` remain recorded zero. Seconds are stored exactly in optional `self_seconds`; minute records retain integer `self_minutes`.
Three-column logs remain valid. Changing elapsed time after ingestion is a semantic conflict (equivalent units such as 60s and 1m deduplicate)
for the same daily token, not a new attempt or a silent overwrite of the old record.


### Local contest discovery (2026-10-20)

Restart `node hosted/local.mjs` after pulling backend changes. In **프로그래밍 대회**, choose
**공개 일정 동기화**. The same-origin local endpoint `POST /api/contests/sync {}` synchronizes
CF official API, AtCoder official calendar and an attributed community CLIST LeetCode mirror.
No additional key, package or infrastructure is required. Refresh reads SQLite only; six-hour cache,
five-minute failed-attempt suppression and stale fallback prevent repeated requests. Source publication
age is separate from fetch success; inspect each modal's source/time and official URL. Weekdays retain
their actual KST date. Hosted/Worker discovery remains unavailable unless explicitly implemented later.

Fit labels are conditional opening-problem practice goals based on existing platform calibration and
historical native difficulty, with at least six distinct independent practice solves; insufficient data
stays **판단 근거 부족**. This does not estimate an official rating or assert registration eligibility.
See `specs/contest-discovery/` and `RECOMMENDATION_ENGINE/` for exact thresholds and evidence.

A `markdown_attempt_conflict` after adding a duration to an already-imported daily row preserves the
original attempt. It is not evidence that Markdown parsing lost the solve. Do not invent a new attempt
ID or reset the watermark to suppress it. An explicit imported-Git correction UI is still future work.

### Explicit review targets

```sh
node hosted/cli.mjs review add --problem leetcode:buddy-strings --date 2026-10-23
node hosted/cli.mjs review remove --problem leetcode:buddy-strings --date 2026-10-23
```

These commands select a dated review; they never create attempts, solved results or timer evidence. Dates default to today in Asia/Seoul. Only an exact active-catalog key is accepted. Repeated add/remove is idempotent. Selections live in the canonical learner preferences (`review_targets`) and existing full backups. Calendar readers use catalog-enriched `review_selections`; the computed review queue still does not populate the calendar. Markdown summary-table memo cells now accept the standalone `#복습` label. Commit the file and run repository synchronization to import the dated selection. Blank outcomes and `-` create no attempts; uncommitted files and detailed prose are not imported. The filename date and first heading MUST agree. Import is additive: removing a label does not remove an already imported selection; use `review remove` for that.

Example (in `leetcode/26-10-23.md`):

```md
# 2026-10-23

| 문제 | 결과 | 시간 | 메모 |
| --- | --- | --- | --- |
| https://leetcode.com/problems/buddy-strings/ | | | #복습 |
```

The date is the log date, not an inferred future review date. `O`/`X` preserve their existing outcome meaning independently of the label. No mastery or completion is inferred from `#복습`.

### Dedicated review column

Preferred format: `문제 | 결과 | 시간 | 복습 | 메모`. Set `복습` to `1` for a target, `0` for a non-target. Blank and legacy `O`/`-` remain readable. The column takes precedence over legacy memo hashtags. Old three/four-column logs and memo `#복습` remain supported when no review column exists. The time column is optional.

```md
# 2026-10-23

| 문제 | 결과 | 시간 | 복습 | 메모 |
| --- | --- | --- | --- | --- |
| https://leetcode.com/problems/buddy-strings/ | | | 1 | 풀이 중 |
```

### Binary daily outcomes

Current daily format uses result `1` (solved), `0` (not solved), blank (in progress/unrecorded); review `1` (target), `0` (not a target). A numeric success imports as `reported_completion` (stored as written); since policy v3 (ADR 005) every completion counts as solved and help is not tracked. Legacy `O` retains its explicit independent meaning. An already-ingested legacy O attempt survives its presentation-only migration to 1 without duplicate attempts or false conflict; changed duration or failure still raises a conflict. Legacy per-problem schema metadata is unchanged. Review imports remain additive: setting a previously imported target to 0 does not cancel it; use `review remove`.

Daily durations are shown as `m`/`s` in workspace history, details and the editable record modal. Confirmed timers remain separate. Weekly `self_reported_minutes` retains minute-unit subtotals for compatibility; `self_reported_seconds_total` sums both self-reported units exactly. Optional seconds survive full learning backups. Restart an already running local server to load the extended JSON validation contracts.
