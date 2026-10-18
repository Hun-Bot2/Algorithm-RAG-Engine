# Mastery system: inventory and critical review

Status: revised 2026-10-17, before implementation. Read this before
`requirements.md`, `design.md` and `tasks.md`.

## 1. Feature inventory: exists, modify, new

Legend: **E** = exists and is reused unchanged; **M** = exists and needs
modification; **N** = genuinely new.

| Capability | Class | Current location | Change |
|---|---|---|---|
| Event envelope, validator, append-only events, supersession | E | `hosted/core/events.mjs`, `hosted/migrations/001_core.sql` | none (R1.9) |
| Timers and confirmed intervals | E | `hosted/core/events.mjs` | none |
| Spaced review 1/3/7/14/30/60 and remediation | E | `hosted/core/policy.mjs:7,31-40` | none |
| Evidence states unknown/learning/demonstrated/transfer | M | `hosted/core/policy.mjs:13-54` | read the merged ledger (Git evidence) |
| Immutable mission snapshots | M | `006_immutable_missions.sql` | add high-water marks, predictions, slot kinds to the payload |
| Mission candidate scoring | M | `hosted/core/policy.mjs:88-148` | add calibrated target zone, explore slots, catalog pool beyond 90 curriculum items |
| Beginner-only cold start | M | `hosted/core/policy.mjs:104` | replaced by the calibrated placement pattern |
| Weekly rhythm | M | `hosted/core/preferences.mjs:2` | add a Mon-Fri / Sat contest / Sun upsolve preset and a weekend window |
| CF/AtCoder submission sync with watermarks | E | `hosted/provider-service.mjs`, `hosted/providers.mjs` | none |
| CF rating history (`user.rating`) | N | none | new opt-in provider call for contest metrics |
| Contest calendar cache | E | `hosted/providers.mjs:2-5` | none |
| Contest suitability | M | `providers.mjs:63` (always false), manual contests | fit annotations; acceptance flow unchanged |
| Contest participation ledger | E | `004_contest_history.sql` | none |
| Virtual contest selection | N | none | `contests-v1` |
| Backup and restore | M | `hosted/backup.mjs` | backup v2 with new tables; v1 still restores |
| Local runtime on node:sqlite | M | `hosted/local.mjs` | read-only ATTACH of a catalog version |
| Catalog collection (CF, AtCoder) | M | `src/learning/catalog_sources.py`, `research/full_catalog/catalog.py` | reused by refresh; add problem models, contests, row hashes |
| 24,297-node catalog, full-v2 embeddings and neighbors | E | `private-learning/unified-map/full-v2` | read by the catalog builder |
| Identity aliases | E | `assets/problem_identity_aliases.v1.json` | none |
| Pinned MiniLM backend and translator | E | `src/learning/embedding_backend.py`, `research/language_bias/` | none |
| AI labeling protocol and lessons | M | `research/full_catalog/` labels, AGENTS.md | generalized to `evaluation-v1` records |
| Optional human verification | E | `research/verification_workbench/` | optional status upgrade only |
| Slack and hosted delivery | E | `hosted/dispatcher.mjs`, `hosted/notifications.mjs` | stays optional |
| Ledger merge and consistency contract | N | none | `ledger.mjs` |
| Git ingestion | N | legacy HEAD-vs-parent only (`src/main/generate_index_map.py:32-56`) | new ingester and tables |
| Platform calibration | N | none | `calibration-v1` |
| Explore/exploit planner | N | none | `planner-v1` |
| Related cross-platform practice | N | none | `related-v1` |
| Outcome metrics | N | none | `outcomes-v1` |
| Weekly report | N | none | `weekly-v1` |
| Catalog DB, delta refresh, embedding cache | N | one-off snapshots only | `refresh-v1` |
| Concept map (crosswalk) | N | none | AI-assisted |
| Archive | N | none | `archive-v1` |
| Filters and CLI | N | none | views plus `cli.mjs find` |

Count: 11 reused unchanged, 10 modified, 13 new. All new work sits behind
additive migrations or in separate modules.

## 2. Weaknesses of the original request (retained from v0)

- W1 "Similar problem of comparable difficulty" across platforms is unsupported.
  - The LeetCode side has only title and tags (statements are quarantined).
  - Other-platform title neighbors reach 33.9% tag agreement against a 20.2%
    random baseline (`research/unified_map/FULL-MAP-REPORT.md:59-73`).
  - 44 of 2,886 AI-reviewed judgments are >= 2.
  - There is no cross-platform difficulty mapping.
  - Resolution: per-platform calibration (design section 4), evidence-status
    labels, and transfer outcomes as the real test.
- W2 LeetCode has no compliant history or new-problem feed. Resolution: Git
  evidence and learner-entered rows. Using AtCoder/CF as the measured contest
  track (D-A) remains open.
- W3 Incremental vector indexing is unnecessary at this scale. Resolution: the
  data-level delta is incremental; neighbors stay exact (R11.4).
- W4 Duplicate engines. Resolution: ADR 003, a single JS core.
- W5 Archive rights risk. Resolution: provenance-aware, link/notes-first archive.

## 3. Critical review of the revised design

### 3.1 Complexity removed during the revision

| Was | Now | Why it is better |
|---|---|---|
| Staircase band rule plus a separate difficulty ladder | One calibration model per platform | The staircase was a crude Elo; calibration also yields predicted P for zones, metrics and contest fit |
| Six independently versioned policies | One bundle `mastery-policy-vN` with component versions | One identifier per snapshot; fewer compatibility cases |
| Randomized exploration with logged propensities | Fixed calendar rotation | Unconfounded comparisons without IPS; 3-5 explore slots per week cannot support off-policy estimators |
| Per-concept ability from day one | Concept offsets only after 8 outcomes | Avoids fitting noise |
| Live contest "selection" | Annotated calendar plus learner acceptance | The decision stays with the learner; less policy surface |
| Incremental kNN index | Exact recomputation per version | Seconds of compute; no consistency bugs |

### 3.2 Remaining complexity, and whether it is justified

1. Five ledger tables merged in code (`events` plus four side tables).
   - Alternative: an events schema v2 with new kinds in one table.
   - That needs rebuilding an append-only table with CHECK constraints (SQLite
     cannot alter CHECK), a v2 validator, and D1 and backup changes.
   - Recommendation: keep side tables, behind one `ledger.mjs` reader with a
     shared contract test. Revisit schema v2 if ledger tables exceed six or
     cross-table supersession is ever needed.
2. Separate catalog database with ATTACH.
   - Alternative: catalog rows inside the learner database.
   - That would mix immutable public-ish metadata with private evidence and bloat
     backups.
   - Recommendation: keep it. In hosted mode, features fall back to the curriculum
     module.
3. AtCoder IRT posterior update.
   - A one-step Laplace update is a little more code than Elo, but it uses the
     published discrimination.
   - Simpler alternative: Elo with K scaled by discrimination. That is acceptable
     if the formula check (MS-401) shows the parameters are hard to reproduce.
4. AI-assisted evaluation.
   - It is justified for the about 110 tags and for AtCoder per-problem concepts,
     where no tags exist.
   - It is of doubtful value for large-scale relatedness labeling. The 2026-10-16
     pool showed sparse positives and unreliable self-reported bases.
   - Recommendation: run relatedness judgments only on candidates actually shown,
     plus a 10% unshown sample.
5. Archive.
   - Still the lowest value per cost.
   - A simpler alternative: a `source` facet plus a notes file. This does not meet
     R12's provenance requirements, so the full archive stays, scheduled last.
6. Metrics volume.
   - At 15-25 attempts per week, most per-concept cells stay below their minimum
     sample for weeks.
   - Recommendation: lead the report with platform-level ISR, excess solve rate
     and ability trajectories; collapse per-concept rows that lack evidence.

### 3.3 Superior alternatives considered

| Area | Alternative | Verdict |
|---|---|---|
| Review scheduling | FSRS (per-item stability and difficulty) instead of fixed intervals | Adopt later if measured retention at stages 30/60 falls below 0.8 over 8 or more reviews. Fixed intervals stay until then, which keeps an unchanged baseline to compare against |
| Contest running | Build an internal virtual-contest runner | Rejected. AtCoder Problems and Codeforces already run virtual contests; the system only selects and records |
| Cross-platform aggregation | Third-party aggregators (for example clist.by) for ratings and contests | Not adopted: they add an API key, terms review and a dependency; native APIs already cover CF/AtCoder |
| Difficulty | A learned cross-platform mapping | Rejected: no ordering labels. Learner-level comparison of `theta_cf` and `theta_atc` may be reported descriptively later |
| Mastery model | BKT/IRT per concept | Rejected for now: single-learner data volume. Calibration plus Beta posteriors cover the decision needs |
| LeetCode history | A third-party extension pushing accepted code to Git | Allowed as the learner's own choice; the system only reads the commits |
| Delivery | Daily Slack push | Optional adapter; Markdown in Git is the default |

### 3.4 Residual risks

- Missing trailers make most Git evidence `reported_completion`. Mitigation: the
  data-quality section and a superseding `manual_correction`.
- AI concept errors leak into exploit targeting. Mitigation: abstain on
  disagreement, label everything `ai_reviewed`, and track per-concept transfer
  lift. A concept with negative lift after 8 transfers is flagged for re-judgment.
- Calibration drift when catalog models change. Mitigation: stored predictions
  and versioned catalogs.
- Self-assessed outcomes (independent vs assisted) are honor-based. Mitigation:
  report it as a limitation; there is no technical fix without a judge.

## 4. Decisions

| ID | Decision | Status |
|---|---|---|
| D-A | Dual track: LeetCode weekday practice plus AtCoder/CF measured contest track | Open |
| D-B | ADR 003, local SQLite canonical and JS core only | Accepted 2026-10-17 on the user's direction to preserve the local-first JS/SQLite architecture |
| D-C | Git trailer and path convention (`design.md` section 3) | Open |
| D-D | Weekly CF/AtCoder metadata refresh (about 6 rate-gated bulk GETs per week) | Open |
| D-E | No human-review prerequisite; AI-assisted plus solving-based evaluation | Accepted 2026-10-17 by user instruction |
