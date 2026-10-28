# Adaptive progression: tasks (progression-v2)

Executable in order. Each task is independently testable. After every task run:

```bash
node --test hosted/test/*.test.mjs
```

Global rules: no change to `core/policy.mjs`, `calibration.mjs`, `progression.mjs`, `planner.mjs`, daily-log parsing,
migrations, ledgers or backups. Tests use in-memory or temporary SQLite and synthetic learners; never the real learner DB.
New modules must not import Node-only modules (`journal`, `catalog`, `git-ingest`, `cli`). Spec: `design.md` sections
in brackets. Mark a task done only with the commands' results recorded in the commit message.

- [x] **T0 Decisions recorded.** ADR 006, requirements, design (this revision).

- [x] **T1 Config and observations** [§2, §3]. Depends on: —.
  Files: `hosted/mastery/progression-v2.config.json`, `hosted/mastery/observations.mjs`, `hosted/test/mastery-observations.test.mjs`.
  Implement a frozen config loader with validation (numbers in range, weights ≥ 0, tau in (0,1), review_days increasing) and
  `deriveObservations(attempts, catalog)`.
  Accept: (a) one problem solved on days 1, 2, 5, 12 → 1 ability + 3 retention observations with gaps 1, 3, 7; (b) two attempts
  on one date → one, the latest; (c) shuffled input → identical output; (d) a 3-concept problem → weight 1/3 each; (e) attempts
  whose events carry a review flag or memo produce identical observations to ones without; (f) alias keys canonicalise;
  (g) unknown problem → reported, not dropped silently; (h) invalid config rejected.

- [x] **T2 Beta math** [§4.1]. Depends on: —.
  Files: `hosted/mastery/beta-math.mjs`; `research/recommendation_audit/beta_reference.py` (scipy, `.venv`) writing
  `hosted/test/fixtures/beta-reference.v1.json`; `hosted/test/mastery-beta-math.test.mjs`.
  Implement regularized incomplete beta (continued fraction), survival `Pr(theta ≥ x)`, quantile (bisection, fixed
  iteration count), variance, information value.
  Accept: ≥40 reference points (alpha, beta in [0.5, 60], x in [0.01, 0.99]) within 1e-9 for CDF and 1e-7 for quantiles;
  information value equals the closed form; quantile(cdf(x)) round-trips; deterministic across runs.

- [x] **T3 LeetCode cells** [§4.1]. Depends on: T1, T2.
  Files: `hosted/mastery/progression-v2.mjs` (cells only), extend `research/recommendation_audit/adaptive_sim.py` into
  `adaptive_reference.py` (leave-concept-out, 1/k weights, PAV), fixture `hosted/test/fixtures/adaptive-cells.v1.json`,
  `hosted/test/mastery-progression-v2-cells.test.mjs`.
  Accept: matches the Python reference within 1e-9 on ≥8 scenarios, including the `adaptive_sim.out` cases; an attempt on
  another concept changes `m_l(-c)` but not the cell's own counts; a PAV-pooled block reports an interval containing its
  mean; repeated solves of one problem change nothing (retention only); time and review flags change nothing.

- [x] **T4 Platform abilities and predictions** [§4.2]. Depends on: T1.
  Files: `progression-v2.mjs` (platform part), tests in `mastery-progression-v2-cells.test.mjs`.
  Accept: CF same problem ×20 equals ×1; predictions use `p_mean` with the ±1.2816 SD interval and lo ≤ p ≤ hi always
  (property test over random inputs); `calibration.mjs` tests unchanged and passing.

- [x] **T5 Readiness, follow-up, v2 reviews** [§7]. Depends on: T3, T4.
  Files: `progression-v2.mjs` (readiness, follow-up), `hosted/mastery/reviews-v2.mjs`, `hosted/test/mastery-progression-v2-state.test.mjs`.
  Accept: readiness needs both the posterior condition and ≥1 own observation (carry-over alone is never ready);
  prerequisite readiness excludes problems carrying the dependent concept; empty or exhausted cells are waived with the
  reason; target concepts bypass with a note; follow-up opens on failure, closes on a later solve at the same or lower level,
  expires at 14 days, and only one is open per concept; the reviews-v2 cases: solves on days 1, 2, 5, 12 advance stages
  where the gap reached the interval and keep the stage otherwise; a retention failure resets to priority; 복습 targets are
  unchanged.

- [x] **T6 Actions, scoring, focus, explanation** [§5, §6, §8]. Depends on: T5.
  Files: `progression-v2.mjs` (`planProgressionV2`), `hosted/test/mastery-progression-v2-plan.test.mjs`.
  Accept: the eight `adaptive_focus_sim.out` scenarios (S1-S8) reproduce the same winners as fixtures built with `catalogFromData`;
  candidates in one cell ordered by coverage gain, then purity, then FNV-1a (renaming keys changes the winner only through the hash);
  an empty action yields `insufficient_catalog_coverage`, and no title-neighbour candidate appears anywhere; transfer
  offered only after Medium ready/waived and only until one solved transfer; every slot carries the full §8 explanation, and the
  reason text uses only those fields; no-deadlock property: random ledgers over a catalog with candidates always produce a
  recommendation; output identical across repeated runs.

- [x] **T7 Journal integration and transitions** [§8, §10]. Depends on: T6.
  Files: `hosted/mastery/journal.mjs` (option `progressionVersion`, default `'v1'`), `hosted/test/mastery-journal.test.mjs`.
  Accept: with `'v1'` the journal view is byte-identical to the current output for the existing fixtures (snapshot test);
  with `'v2'` the slot comes from `planProgressionV2`, reviews come from reviews-v2, and `mastery_change` entries follow the
  §8 signature with cause and before/after estimates; the snapshot key includes the progression version and the config hash;
  explanations (`explanations.mjs`) accept v2 slots unchanged; UI contract tests pass.

- [ ] **T8 Real-catalog replay and sensitivity receipt** [§6, §10]. Depends on: T7.
  Files: `research/recommendation_audit/adaptive_replay.mjs`, receipt `research/recommendation_audit/receipts/adaptive_replay.out`.
  On catalog-v4 (read-only) with synthetic learners (all-solve, alternating, struggling, CF-only), report per step: concept,
  action, level, P, B. Also the sensitivity grid: kappa {2,4,8} × lambda {0.25,0.5,1}, weights variants (D only; default),
  attribution 1/k vs 1, authored vs flat priors.
  Accept: every concept either receives a recommendation within 60 simulated steps or reports a waiver/coverage status;
  no alphabetical dependence (shuffling key order changes no winner except via the hash); the receipt states it measures behaviour, not relevance or learning.
  Reviewer sign-off before T9.

- [ ] **T9 Default switch and docs** [§10]. Depends on: T8 + reviewer sign-off.
  Files: `journal.mjs` default `'v2'`, CLI `--progression v1|v2` (optional), `specs/mastery-journal/runbook.md`,
  `RECOMMENDATION_ENGINE/` pointer, `AGENTS.md`, and rename the weekly "independent solve rate" label to "completion rate
  (policy v3)".
  Accept: the full Node suite passes; v1 is still selectable and byte-identical; the real learner DB is opened by no test;
  the runbook explains the new recommendation fields in Korean for the user.

## Execution receipts

- T1 (2026-10-29): observations/config focused 5 pass; full `node --test hosted/test/*.test.mjs` 259/259 pass with loopback access. New modules only; no real learner DB opened. Config loader is a separate Worker-safe module for independent validation.
- T2 (2026-10-29): SciPy 1.14.1 reference 49 points; focused 2/2, full Node 261/261 pass. CDF tolerance 1e-9, quantile 1e-7.
- T3 (2026-10-29): 10 independent SciPy cell scenarios matched within 1e-9; full Node 266/266 pass (includes independently added review fixtures). v1 files unchanged.
- T4 (2026-10-29): platform repeat/interval property fixtures pass; full Node 267/267 pass.
- T5 (2026-10-29): readiness/follow-up/review focused fixtures pass; independent Terra review corrected sparse config validation and failure due-day; full Node 269/269 pass.
- T6 (2026-10-29): user clarified fresh concepts retain authored priors; independent 11-cell fixture updated. S1-S8, deterministic insertion order/cooldown/no-deadlock checks pass; full Node 276/276 pass. Sparse-array and custom failure-day regressions added.
- T7 (2026-10-29): default/explicit v1 serialized output matches pre-integration fixture; v2 snapshot/config isolation, evidence-prefix transitions and independent review lane tests pass. Future ratings excluded at each prefix; full Node 281/281 pass. catalog-v4 synthetic 10-plan run: 618.259292 ms; 10-prefix transition replay: 619.67925 ms (Node26.8.1 macOS arm64, one run, not a general benchmark).
