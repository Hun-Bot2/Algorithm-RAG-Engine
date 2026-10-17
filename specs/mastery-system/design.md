# Mastery system design v1

Status: revised 2026-10-17, before implementation. Implements `requirements.md`.
`critique.md` holds the feature inventory and the critical review.

## 1. Architecture

```text
  Learner Git repos           CF API / AtCoder Problems      Learner (CLI, dashboard)
  (LeetCode, Baekjoon notes)  (opt-in, rate-gated)           attempts, timers, verdicts
          |                            |                               |
          v                            v                               v
  +-------------------------------------------------------------------------+
  | JS core (hosted/core) on node:sqlite via hosted/local.mjs   [existing]   |
  |  ingest: git-ingest [new] | provider-service [existing] | routes [exist] |
  |  ledger.mjs [new]: merge ledger tables at a cutoff -> evidence stream     |
  |  policy bundle mastery-policy-v1:                                         |
  |    calibration [new] -> planner (review/exploit/explore) [modify mission] |
  |    related [new] | contests [modify] | metrics [new] | weekly [new]       |
  +-------------------------------------------------------------------------+
        | read/write (append-only)                     | read-only ATTACH
        v                                              v
  learner.sqlite3  [existing schema + additive]   catalog-vN.sqlite3 [new, immutable]
                                                       ^
              Python offline batches: catalog delta -> embed cache -> kNN ->
              AI-assisted evaluation -> versioned artifacts + receipts

  Optional adapters: Markdown report -> learner Git repo | Slack | Worker + D1
```

### 1.1 ADR 001 and ADR 003

| Topic | ADR 001 (accepted 2026-10-16) | ADR 003 (accepted 2026-10-17) | Effective rule |
|---|---|---|---|
| Deterministic no-AI core | Decided | Unchanged | Kept |
| Offline/optional AI | Decided | Extended to AI-assisted evaluation batches | Offline only; outputs are versioned data |
| Canonical learner store | D1 in the eventual hosted service; local SQLite for development and recovery | Local SQLite is canonical | ADR 003 supersedes |
| Hosting (Worker, D1, Access) | The release target (M8) | An optional deployment of the same core | ADR 003 supersedes; hosted tests keep passing |
| Unattended 06:00 phone delivery | Release requirement | Optional adapter; weekly Markdown is the default channel | ADR 003 supersedes |
| Python role | Offline profiling, ingestion, evaluation | Same, plus catalog versions | Kept |

Single-writer rule: exactly one deployment is canonical at a time. Moving between
local and hosted uses the existing backup export and restore-into-empty-owner
path. No two-way sync exists or is planned.

## 2. Data model

### 2.1 Ledger tables (learner database)

Existing tables, unchanged: `events`, `contest_participation`,
`mission_snapshots`, `manual_contests`, `provider_cache` and `delivery_outbox`.
The `events` CHECK constraints (`kind`, `source`) and the exact-field validator in
`hosted/core/events.mjs` stay as they are.

New migrations are additive. Every ledger table follows the contract in R1.2,
modeled on `004_contest_history.sql`.

| Table | Rank | Source values | Payload columns |
|---|---|---|---|
| `events` (existing) | 1 | manual, platform_import | as today |
| `contest_participation` (existing) | 2 | manual | as today |
| `git_observations` | 3 | git, manual_correction | repo_id, sha, path, problem_key, outcome, hints, self_minutes, raw_trailers |
| `practice_feedback` | 4 | manual | snapshot_ref, source_key, candidate_key, verdict (related/partly/unrelated/skip) |
| `catalog_switches` | 5 | manual | from_version, to_version |
| `git_watermarks` | (not evidence) | git | repo_id, branch, last_processed_sha, status (ok/history_rewritten/error) |

New non-ledger tables: `git_repositories` (registration), plus immutable snapshot
tables `recommendation_snapshots` and `weekly_reports`. Both have no-update and
no-delete triggers, like `006_immutable_missions.sql`.

### 2.2 Consistency model

- Idempotency: `INSERT` compares canonical JSON for an existing `entry_id`.
  An equal value is a no-op; a different one fails (existing rule, reused).
- Derived IDs:
  - `git_observations.entry_id = sha256("git" | repo_id | sha | problem_key)`
  - provider imports keep their existing external-ID rule
- Corrections: a `manual_correction` row supersedes one earlier row in the same
  table for the same learner. Supersession across tables is invalid. At most one
  successor per row (UNIQUE `supersedes`).
- Transactions: one Git batch writes its observations and appends one watermark
  row in a single transaction. This mirrors the provider rule "events written
  before watermark advances".
- Cutoffs: snapshots store `event_cutoff` (semantic time) and `high_water`
  (`{table: max sequence}`). Normal replay filters `occurred_at <= cutoff` and
  `recorded_at <= cutoff`. Reproduction replay filters `sequence <= high_water`,
  which is monotonic and immune to clock skew.
- Merge: `ledger.mjs` reads each table, applies supersession inside the table,
  then merges by (`occurred_at`, table rank, `entry_id`). This is the only reader
  that policy modules use.
- Derived state (ability, concept status, metrics) is recomputed by replay and
  never written back as truth.
- Backup v2 adds the new tables. v1 backups still restore; missing tables are
  treated as empty.

Rollback: restore the pre-migration backup. Migrations never drop or alter
existing tables.

### 2.3 Catalog database (immutable, one file per version)

| Table | Content |
|---|---|
| `catalog_meta` | version, parent version, input hashes, model revision, delta receipt hash |
| `problems` | key, platform, title, url, contest_id, contest_start, difficulty_kind (lc_level/cf_rating/atc_irt/none), difficulty_value, atc_discrimination, atc_experimental, access (free/premium/unknown), eligible, exclusion_reason, status (active/removed_upstream), first_seen_version, row_hash, provenance |
| `problem_tags` | native tags |
| `aliases` | from `assets/problem_identity_aliases.v1.json` |
| `concepts`, `concept_map` | (platform, native_tag or problem_key) to concept_id, evidence_status (approved/ai_reviewed/abstained/proposed), judgment_ref |
| `neighbors` | src, dst, channel (title_v2, statement_cc), rank, score, model_rev |
| `contests`, `contest_problems` | CF contest.list and problemset; AtCoder contests and contest-problem |
| `archive_records`, `archive_links` | compiled from learner archive files (section 10) |
| `reliability` | full-v2 tiers |
| `ai_judgments` | compiled AI-assisted evaluation outputs (section 8) |

LeetCode statement fields are never loaded. The builder reuses the
`object_pairs_hook` drop in `research/full_catalog/catalog.py`.

## 3. Git ingestion

The convention is pending decision D-C. Proposed path rules:

```text
leetcode/<slug>/...        -> leetcode:<slug>
baekjoon/<id>/...          -> baekjoon:<id>
codeforces/<contest><idx>/ -> codeforces:<contest><idx>
atcoder/<task_id>/...      -> atcoder:<task_id>
```

Proposed trailers:

```text
Problem:  <key>                                  (when the path is ambiguous)
Outcome:  independent | assisted | failed | accepted
Hints:    none | used
Minutes:  <int>                                  (self-reported)
```

`accepted`, or a missing outcome, maps to `reported_completion`.

Mechanics:

- Ingestion runs `git rev-list --reverse <watermark>..<branch>` and reads trailers
  per commit. It makes no network calls.
- If the watermark is no longer an ancestor of the branch, the repository status
  becomes `history_rewritten` and ingestion stops.
- The existing Baekjoon notes repo (front-matter `id`) maps to
  `baekjoon:<id>` as `reported_completion`.

## 4. Platform-specific calibration (`calibration-v1`)

Outcome coding for practice attempts: `y = 1` for independent with no hints;
`y = 0` for failed or assisted. `reported_completion` is excluded everywhere.

### Codeforces (logistic on the native rating scale)

```text
P(solve) = 1 / (1 + 10^((r_problem - theta_cf) / 400))
prior    theta_cf ~ N(mu0, 350^2); mu0 = CF contest rating if known, else 1000
update   theta_cf += K * (y - P), K = max(32, 200 / sqrt(1 + n_cf))
```

The CF problem rating is a contest-condition quantity. Untimed practice differs,
so `theta_cf` is a practice estimate, separate from contest rating (R3.3).

### AtCoder (AtCoder Problems IRT parameters)

```text
P(solve) = 1 / (1 + exp(-a * (theta_atc - d)))
           a = discrimination, d = difficulty (raw, unclipped)
prior    theta_atc ~ N(mu0, 400^2); mu0 = learner-entered rating mapped to the
         internal scale (the display transform below 400 is inverted), else 0
update   Laplace-approximate posterior update per observation
         (one Newton step on the log-posterior)
```

- Experimental models count with half weight. The 2026-10-16 snapshot has
  5,175 model entries: 4,889 with `difficulty` and `discrimination`, and 565
  flagged `is_experimental`.
- Tasks without a model get `P = unknown`.
- Verify the exact formula and the display transform against the AtCoder Problems
  source before implementation (task MS-401).

### LeetCode (Beta-binomial per native level)

```text
P(level) = (alpha_l + s_l) / (alpha_l + beta_l + n_l)
priors (alpha, beta): Easy (3.2, 0.8), Medium (2, 2), Hard (0.8, 3.2)
```

Pool-adjacent-violators keeps P(Easy) >= P(Medium) >= P(Hard).

### Shared rules

- Concept offsets: a per-concept shift `delta_k` (shrinkage prior N(0, 150^2) on
  the CF scale, analogous on AtCoder) activates only after 8 known outcomes in
  that concept. Before that, `delta_k = 0`.
- Predictions are written into the recommendation snapshot at decision time.
  Metrics use the stored value (R3.4).
- Calibration report (once at least 30 predictions have outcomes): Brier score,
  Brier score of a constant-rate baseline, and reliability in 3 bins. If the model
  does not beat the baseline, the weekly report says so and the target zone widens
  to 0.50-0.85.

## 5. Planner: review, exploit and explore (`planner-v1`)

Planning for a practice day (Mon-Fri by default):

1. Review: due items first (existing intervals). A failed or assisted review puts
   its concept into remediation.
2. Exploit: for each target concept k, using known outcomes in the last 56 days,

   ```text
   posterior   Beta(1 + s_k, 1 + f_k)
   need_k      = 1 - mean_k
   priority_k  = need_k + 0.5 * target_gap_k    # existing target-gap component
   ```

   Pick the top concept, then the problem whose predicted P is closest to 0.70
   within [0.60, 0.80]. Ties go to curriculum items, then the problem key.
   Fill the budget with further exploit slots.
3. Explore: at most one slot per practice day, capped at 25% of weekly practice
   minutes. The sub-kind comes from a fixed rotation:

   ```text
   index = (ISO week * 5 + weekday) mod 4
   0 uncertain concept : target concept with the largest posterior SD among n_k < 4
   1 stretch probe     : demonstrated concept, predicted P in [0.35, 0.55]
   2 related channel   : related-practice candidate from channel
                         rotation[ISO week mod C] (concept, title_v2, statement_cc*)
   3 new problem       : eligible problem first seen in the last 4 catalog versions
   ```

   `statement_cc` applies only to CF/AtCoder sources with CodeContests
   statements. LeetCode sources never use it. If the chosen sub-kind has no
   candidate, the next sub-kind in rotation order is used and the fallback is
   recorded.
4. Remediation override: when a concept is in remediation, the explore slot
   becomes a remediation practice item for that concept.
5. Cold start: with fewer than 6 known-outcome practice attempts on a platform,
   the slots on that platform use a placement pattern at P of about 0.85, 0.65
   and 0.45.
6. Maintenance: a concept with posterior mean >= 0.75 and n_k >= 6 leaves exploit
   rotation. It gets reviews and occasional transfer items only.

Why a schedule and not randomization: the rotation does not depend on outcomes,
so comparing sub-kinds and channels is not confounded by learner state, and no
propensity bookkeeping is needed. A single learner generates 3-5 explore slots
per week, too few for off-policy estimators.

## 6. Related practice and contests

### Related practice (`related-v1`)

```text
eligible(c): platform in {codeforces, atcoder}; eligible; access = free;
             not seen within cooldown (aliases); concept_overlap > 0 or
             reviewed edge; P(c) known

score(c) = 4 * reviewed_edge
         + 3 * jaccard(concepts)        # approved or ai_reviewed concept rows
         + 2 * rrf(channel rank)        # 1 / (60 + rank), rank <= 50
         + 2 * (1 - |P(c) - 0.70| / 0.70)
         - 1 * degraded_title
```

Labels follow R5.3. AtCoder concepts come only from `concept_map` rows keyed by
problem (AI-assisted, section 8) or from reviewed edges.

### Weekend contests (`contests-v1`)

- Live: list CF and AtCoder contests from the cached calendar that fall in the
  weekend window, labeled with fit.
  - Fit means the expected solved count (sum of P over the contest's problems,
    when known) is between 1.5 and 4. Otherwise the type rule applies: ABC or
    Div. 3/4 by default, Div. 2 at `theta_cf` >= 1400, ARC at `theta_atc` >= 1200.
  - The list is an annotated calendar, not a decision. Acceptance uses the
    existing `suitable` flow.
- Virtual: past contests at least 14 days old with no attempted or imported
  problem (aliases resolved), ranked by:

  ```text
  2 * weakness coverage (unknown -> 0, reported)
  + 1 * expected-solved fit
  + 0.5 * recency
  ```

  The run itself happens on AtCoder Problems virtual contests or Codeforces
  virtual participation. The system does not run contests.
- Sunday upsolve targets the first unsolved weekend contest problem.

## 7. Learning outcome metrics (`outcomes-v1`)

All metrics come from the merged ledger at the report cutoff. Each shows
`n`, the window, a Wilson 95% interval, and `insufficient_evidence` below its
minimum sample.

| Metric | Definition | Minimum n |
|---|---|---|
| Independent solve rate (ISR) | First attempts on new practice problems with known outcome: count(y=1) / n, by platform and concept | 10 per cell |
| Excess solve rate | mean(y - P_stored) over the same attempts; positive means outperforming calibration | 10 |
| Ability trajectory | weekly theta_cf, theta_atc, and LeetCode level posteriors, each with SD | n/a (always shown with SD) |
| Transfer success rate (TSR) | First attempts on problem B where B shares an approved or ai_reviewed concept with a problem A solved independently in the prior 30 days, and B is on another platform or came from a related-practice slot: count(y=1) / n | 8 |
| Transfer lift | TSR - mean(P_stored) over those attempts; split by channel using the explore rotation | 8 per channel |
| Review retention | Per stage s in {1,3,7,14,30,60}: independent no-hint reviews / reviews done within +/-1 day of due; plus lapse rate and on-time rate | 8 per stage |
| Contest performance | CF rating delta (`user.rating`, sync opt-in); AtCoder rating/performance as learner-entered; solved count vs expected solved (sum of P_stored); upsolve completion within 7 days | 1 contest (descriptive) |
| Process | Planned vs done slots; confirmed minutes and self-reported minutes, shown separately | n/a |

Trends are shown only with at least 4 weeks of data. The report states that the
metrics are observational and single-learner (R8.3).

## 8. AI-assisted and solving-based evaluation (`evaluation-v1`)

| Object | Primary signal (from solving) | AI-assisted signal | Status rule |
|---|---|---|---|
| Concept map: LeetCode/CF tags (about 110 tags) | Transfer outcomes per concept | Draft by model A; blind review by model B | Agree: ai_reviewed. Disagree: abstained |
| AtCoder per-problem concepts (titles; AI may recognize problems) | Outcomes of AtCoder slots | Draft plus blind review; a "recognized problem" basis must be backed by a title or URL citation | Same rule; empty rationale rejected |
| Related-practice candidates | Learner verdicts; transfer lift by channel | Offline relatedness judgments on all shown candidates and a sample of unshown ones | Used only for offline channel precision@k |
| Archive practice_for links | Outcomes on linked problems | Draft plus review | same_problem never from AI |

Protocol, reusing the existing labeling lessons:

- Two different model roles. The reviewer does not see the drafter's verdict.
- Rationales are required, and claims are mechanically spot-checked against the
  input files.
- `deceptive`-style fields are not used.
- Each record stores the model, date, prompt hash, input hash and role.

Runs are offline research batches (for example Claude Code subagents or
`research/` scripts), never the runtime core. Outputs compile into
`ai_judgments` and `concept_map` with versioned receipts.

Anti-circularity:

- A channel is never evaluated with labels produced from the same features. Do
  not evaluate the concept channel with AI concept labels; use learner verdicts
  and transfer lift.
- AI-learner agreement is reported where both exist, to show how much AI labels
  can be trusted.

Human review through `research/verification_workbench` is optional and can
upgrade `ai_reviewed` to `approved`.

## 9. Incremental catalog updates (`refresh-v1`, weekly, opt-in per D-D)

```text
1 fetch      CF problemset.problems, contest.list; kenkoooo problems.json,
             contests.json, contest-problem.json, problem-models.json
             serialized, >= 2.1 s apart, stop on 403/429; raw kept private
2 normalize  catalog.py rules, aliases applied, row_hash = sha256(canonical row)
3 delta      vs version N by (key, row_hash) -> added / changed / removed / unchanged
             receipt: counts, keys, input hashes
4 carry      unchanged rows copied from N; removed rows -> status removed_upstream
5 translate  new or changed Japanese titles only (full-v2 translator, offline)
6 embed      cache lookup sha256(text) + model_rev; encode misses only
7 neighbors  exact top-50 for all rows (see note)
8 judgments  new problems join the next evaluation batch; until then their
             concept status is unknown (eligible only for the explore new-problem slot)
9 build      catalog-v(N+1).sqlite3 + receipt; refuse existing output path
10 switch    learner appends catalog_switches; old snapshots keep their version
```

Note on step 7: full exact recomputation over about 24,300 x 384 float32 vectors
is seconds of BLAS work. An exact incremental merge is possible: score added rows
against everything, merge into existing top-k lists, and fully recompute rows
whose list contained a removed or changed row. It is deferred until a refresh
exceeds 60 seconds, and must pass an equality test against full recomputation
(R11.4).

AtCoder model parameters can change between versions. Stored predictions keep
past metrics stable, and new recommendations use the active version.

## 10. Provenance-aware archive (`archive-v1`)

Learner-authored `archive.vN.json` lives in a private repository and is compiled
into the catalog database.

```json
{"record_id": "kaist-grad-2024-s1-q3", "version": 2, "supersedes": "kaist-grad-2024-s1-q3@1",
 "category": "grad_entrance",
 "origin": {"organization": "...", "exam": "...", "year": 2024, "session": "1", "label": "Q3"},
 "source": {"kind": "official_publication", "citation": "...", "url": "...", "accessed": "2026-10-17"},
 "acquisition": "link_only | manual_transcription | recollection | purchased_book",
 "rights_status": "link_only | personal_notes | own_work | licensed",
 "private_copy_sha256": null, "confidence": "high | medium | low",
 "recorded_by": "learner | ai_draft", "statement_text": null, "notes": "learner's own notes",
 "links": [{"problem_key": "codeforces:1234B", "relation": "practice_for | concept_only | same_problem",
            "evidence": [], "status": "approved | ai_reviewed"}]}
```

The validator rejects:

- `statement_text` unless `rights_status` is `own_work` or `licensed`
- `same_problem` with fewer than 2 evidence citations, or recorded by AI
- `recollection` with `confidence: high`
- duplicate or dangling `supersedes`

Categories: `company_interview`, `grad_entrance`, `certification`,
`contest_past`, `other`.

## 11. Weekly report (`weekly-v1`)

Sections:

1. Activity
2. Review
3. Outcome metrics (current week plus 4-week rolling)
4. Calibration
5. Weaknesses and next-week focus
6. Contests
7. Evaluation tallies (verdicts, AI-learner agreement, explore-kind results)
8. Data quality (unclassified commits, rewritten histories, provider gaps,
   catalog age, unknown-coverage counts)

The report is stored as an immutable snapshot with cutoff and high-water marks,
plus optional Markdown at `<repo>/reports/<week_start>.md`.

## 12. Filtering

Read-only views:

- `v_problem_status`: unseen, exposed, attempted, solved_reported,
  solved_independent, review_due
- `v_problem_facets`: including predicted P from the current calibration
- `v_neighbors`
- `v_archive`

A new `hosted/cli.mjs find` composes them. Each date filter names its basis.
Dashboard filters come later and reuse the query builder.

## 13. Validation plan

- node --test:
  - ledger contract on each new table: idempotency, conflict, supersession,
    triggers
  - ledger merge order and high-water reproduction
  - Git ingester on temporary repos (linear, merge, force-push, missing trailer)
  - calibration updates against hand-computed fixtures
  - planner rotation and caps, remediation override, cold start
  - related scoring golden results
  - contest exclusion through aliases
  - metric definitions on a synthetic 6-week ledger with known answers
  - weekly golden Markdown
  - compliance: no `leetcode.com/(api|graphql|submissions)` URL anywhere
- Python unittest:
  - catalog builder (no statement fields; receipt hashes)
  - delta classification
  - embedding cache bit-identity with full-v2
  - neighbor equality with full recomputation
  - archive validator
  - AI judgment record validator (rejects empty rationale or missing hashes)
- Regression: existing hosted, `tests/` and research verify suites stay green;
  pinned research files unchanged.
