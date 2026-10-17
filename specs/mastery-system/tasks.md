# Mastery system tasks v1

Rules:

- One task per Ralph iteration. Each task names its validation.
- E/M/N marks the target per `critique.md` section 1.
- Human review is never a gate (D-E). AI-assisted steps record `ai_reviewed`.

Commands:

- JS: `node --test hosted/test/*.test.mjs`
- Python: `python3 -m unittest discover -s tests -v`
- Research: `.venv/bin/python -m research.unified_map.verify private-learning/unified-map/map-v2`

## Phase 0: decisions

- [x] MS-001 D-B accepted (ADR 003) and D-E accepted, 2026-10-17.
- [ ] MS-002 User decides D-A, D-C and D-D. Gates: D-C gates Phase 2, D-D gates
  MS-703, D-A gates the contest-track defaults in MS-603.

## Phase 1: ledger foundation

- [ ] MS-101 (M) Document `hosted/local.mjs` as the canonical runtime and mark
  `src/learning/store.py` frozen. Validation: docs only; suites green.
- [ ] MS-102 (N) `hosted/core/ledger.mjs` reading `events` and
  `contest_participation`, with cutoff and high-water replay. Validation: merge
  order and reproduction tests on existing tables.
- [ ] MS-103 (N) Shared ledger contract test helper (idempotency, conflict,
  supersession, triggers), applied to the existing tables first.
- [ ] MS-104 (M) Snapshot payloads gain `high_water`, `policy_bundle` and slot
  metadata; existing snapshot readers still pass.

## Phase 2: evidence ingestion (gated by D-C)

- [ ] MS-201 (N) Migration `007_git_evidence.sql`: `git_repositories`,
  `git_watermarks`, `git_observations`. Validation: contract helper; v1 backup
  restores.
- [ ] MS-202 (N) `git-convention.mjs` parser (paths, trailers). Validation:
  table-driven tests.
- [ ] MS-203 (N) Ingester with a transactional watermark and rewritten-history
  stop. Validation: temporary-repo tests (linear, merge, force-push, missing
  trailer).
- [ ] MS-204 (M) Evidence derivation reads the merged ledger; only explicit
  independent outcomes advance state. Validation: policy tests.
- [ ] MS-205 (N) Baekjoon front-matter rule. Validation: fixtures, including
  malformed notes.
- [ ] MS-206 (M) Backup v2. Validation: round trip; v1 import.

## Phase 3: catalog database (read-only)

- [ ] MS-301 (N) `research/mastery_catalog/build.py` from catalog-v1, full-v2
  neighbors, aliases, reliability, contests and AtCoder problem models.
  Validation: unittest; no statement fields; receipt hashes.
- [ ] MS-302 (M) Read-only ATTACH in `hosted/local.mjs`; snapshots record the
  catalog version. Validation: a write attempt fails; version is recorded.
- [ ] MS-303 (N) Views and `hosted/cli.mjs find`. Validation: one query test per
  R13.1 dimension.

## Phase 4: calibration and planner

- [ ] MS-401 (N) Verify the AtCoder Problems IRT formula and rating display
  transform against its source; record citations in `design.md`. Validation: the
  reproduced P matches the published model on 5 sampled tasks, or Elo fallback is
  chosen.
- [ ] MS-402 (N) `calibration.mjs` (CF logistic, AtCoder IRT, LeetCode
  Beta-binomial with PAV). Validation: hand-computed fixtures.
- [ ] MS-403 (M) `planner.mjs` inside mission building: review, exploit, explore
  rotation, caps, remediation override, cold start. Validation: rotation table
  tests; 25% cap; deterministic replays.
- [ ] MS-404 (M) Rhythm preset and weekend window in preferences. Validation:
  rhythm tests.

## Phase 5: outcomes and weekly report

- [ ] MS-501 (N) `metrics.mjs` (`outcomes-v1`). Validation: a synthetic 6-week
  ledger with known answers for every metric and threshold.
- [ ] MS-502 (N) Migration for `weekly_reports` and `recommendation_snapshots`
  (immutable).
- [ ] MS-503 (N) `weekly.mjs` plus a Markdown writer that refuses to overwrite.
  Validation: golden Markdown.

## Phase 6: AI-assisted concepts and related practice

- [ ] MS-601 (N) Judgment record schema and validator (model, prompt hash, input
  hash, role, non-empty rationale). Validation: rejection tests.
- [ ] MS-602 (N) Offline concept-map batch for LeetCode/CF tags: draft plus blind
  review, then `ai_reviewed` or `abstained`. Validation: validator passes;
  mechanical spot-check report.
- [ ] MS-603 (N) AtCoder per-problem concept batch (titles only; cited
  recognition). Validation: same as MS-602.
- [ ] MS-604 (N) `related.mjs` plus the `practice_feedback` migration and CLI
  verdict entry. Validation: golden candidates; zero-overlap exclusion; contract
  helper.
- [ ] MS-605 (N) `contests.mjs`: fit annotations and virtual selection with alias
  exclusion; opt-in CF `user.rating` sync. Validation: calendar fixtures across
  KST midnight; solved-alias exclusion.

## Phase 7: incremental catalog refresh (gated by D-D)

- [ ] MS-701 (N) Delta classifier (key plus row hash) with receipt. Validation:
  fixture covering add, change, remove and unchanged.
- [ ] MS-702 (N) Embedding cache. Validation: bit-identical to full-v2 vectors.
- [ ] MS-703 (N) Fetcher with replay mode; live mode behind a flag. Validation:
  replay tests; 403/429 stop.
- [ ] MS-704 (N) Version N+1 build with exact neighbors and a `catalog_switches`
  entry. Validation: neighbor equality; refusal to overwrite.

## Phase 8: provenance-aware archive

- [ ] MS-801 (N) `archive-v1` schema and validator. Validation: the rejection
  rules in `design.md` section 10.
- [ ] MS-802 (N) Compile into the catalog database plus `v_archive` filters.
  Validation: query tests; supersession history readable.

## Phase 9: evaluation and optional adapters

- [ ] MS-901 (N) Offline relatedness judgments on shown candidates plus a 10%
  unshown sample; AI-learner agreement report.
- [ ] MS-902 (M) Optional Slack weekly message (dry-run payload test only).
- [ ] MS-903 (M) Dashboard filters and verdict buttons reusing MS-303.
- [ ] MS-904 (N) FSRS evaluation, only if retention at stages 30/60 is below 0.8
  over 8 or more reviews.
