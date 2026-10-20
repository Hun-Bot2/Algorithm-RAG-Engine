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
