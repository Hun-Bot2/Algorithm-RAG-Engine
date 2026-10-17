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
