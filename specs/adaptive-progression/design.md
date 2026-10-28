# Adaptive progression: design (progression-v2)

Status: accepted 2026-10-29 (ADR 006); implementation pending. Implements `requirements.md`. References: the audit
`research/recommendation_audit/REPORT.md` and its receipts (*exec*), or source reading (*static*). (P) marks a provisional
assumption, versioned and shown, with no claim of empirical optimisation.

## 1. Why v1 is replaced, not tuned

| v1 behaviour | Consequence | Evidence |
|---|---|---|
| Easy/Medium pick = first eligible slug alphabetically | Order unrelated to the learner | exec (`replay.out`) |
| Advance = 3 Easy + 2 Medium distinct variations | Count-based, ignores uncertainty | static |
| Transfer/retention need `activity_type` transfer/review | Unreachable from Markdown (rows ingest as `practice`) | static (`daily-log.mjs`) |
| Spacing grows only on `activity_type=review` | Markdown re-solves stay at a 1-day interval | exec |
| Prerequisite met only after transfer | Concepts stall behind union-find, heaps, ... | exec (`replay.out`) |
| Repeats are independent observations | Same CF problem ×20: ability 1000 → 1341 | exec (`remed.out`) |

## 2. Module layout (all new, pure, Worker-safe unless noted)

```text
hosted/mastery/progression-v2.config.json   all constants (R25)
hosted/mastery/beta-math.mjs                 regularized incomplete beta, quantile, survival, variance, info value
hosted/mastery/observations.mjs              deriveObservations (R5-R9)
hosted/mastery/progression-v2.mjs            cells, readiness, follow-up, actions, scoring, focus, transitions
hosted/mastery/reviews-v2.mjs                v2 spacing replay (R20); core/policy.mjs untouched
hosted/mastery/journal.mjs                   option progressionVersion: 'v1' | 'v2' (Node-only, existing file)
```

`calibration.mjs`, `progression.mjs`, `planner.mjs` and `core/policy.mjs` keep their behaviour, so v1 stays byte-identical.

Config (`progression-v2.config.json`, version `progression-v2-config-1`):

```json
{"tau":0.70,"credible_mass":0.80,"readiness_probability":0.5,
 "level_priors":{"1":[3.2,0.8],"2":[2,2],"3":[0.8,3.2]},"kappa":4,"lambda":0.5,
 "weights":{"challenge_fit":0.5,"information":0.3,"coverage":0.2,"follow_up":0.3,"continuity":0.2},
 "follow_up_days":14,"review_days":[1,3,7,14,30,60],"transfer_platforms":["codeforces","atcoder"],
 "transfer_native_difficulty_floor":null,"exclude_experimental_atcoder":true}
```

Exposure cooldown reuses the existing preference `mastery.cooldown_days` (default 30): a problem recommended within that
window and not attempted is not offered again. Mock-contest exposures count, as today.

## 3. Evidence derivation (`observations.mjs`)

Input: `ledger.attempts` (v3 projection; `y = 1` for solved, `0` for failed; `reported_completion` already projected). Steps:

1. Canonicalise keys through catalog aliases; drop attempts outside the catalog (reported as `unknown_problem`).
2. Group by (problem, Seoul date). Keep the latest by `occurred_at`, then id (R6).
3. Per problem in date order: first → `ability`; later → `retention` with `gap_days` = days since previous kept date.
4. Each observation carries `{id, problem_key, platform, level|rating, concepts, weight=1/k, y, date, self_minutes|self_seconds}`.

Review flags and memos are never read here (R2, R3).

## 4. Estimation

### 4.1 LeetCode cells (Beta-binomial, partial pooling) — D2, D3, D4 (P)

```text
level mean outside c   m_l(-c) = posterior mean of Beta(prior_l) updated with ability observations at level l
                                 whose problems do NOT carry concept c (unit weight per problem)
prior centre           m'_E = m_E(-c);  m'_M = sigmoid(logit m_M(-c) + lambda*delta_E);  m'_H = sigmoid(logit m_H(-c) + lambda*delta_M)
                       delta_l = logit E[theta_{l,c}] - logit m'_l  (0 when the cell has no own observation)
cell                   theta_{l,c} ~ Beta(kappa*m'_l + sum w*y, kappa*(1-m'_l) + sum w*(1-y))
monotonicity           weighted PAV over E, M, H means within the concept; a pooled block reports the Beta of its summed
                       pseudo-counts, so mean and interval stay consistent
interval               central 80% interval by exact Beta quantiles (beta-math.mjs)
```

The order E → M → H matters (each carry-over uses the lower level's posterior), so cells are computed in that order before
PAV.

### 4.2 Codeforces / AtCoder

`fitAbility` from `calibration.mjs`, called with ability observations only (deduplicated). The prediction for a candidate
is the existing marginal `p_mean` with an 80% interval from ability mean ± 1.2816 SD pushed through the logistic. This
replaces `p_conservative` in v2 (fixes REPORT F8). Information value: `I = v*a^2*p*(1-p) / (1 + v*a^2*p*(1-p))` (Laplace
step). AtCoder experimental models keep weight 0.5.

### 4.3 Not modelled (D5, D11)

Time does not enter any estimate. LeetCode has no within-level difficulty; "a harder Easy" exists only on CF/AtCoder.

## 5. Actions, candidates, scoring

LeetCode levels relative to the concept's highest level with own observations (Easy when none):

| Relative level | Action |
|---|---|
| below | step_down |
| same | consolidate |
| +1 | step_up |
| +2 | stretch (only with ≥1 ability observation in the concept) |

Plus `transfer` (R15): eligible CF/AtC problems in the concept, AtCoder experimental excluded, choosing the rating whose
`p_mean` is closest to tau. Candidates for every action: eligible, `recommendable`, trusted concept present, not attempted,
not exposed within the cooldown.

Benefit proxy for action a of concept c (D6, P):

```text
D = 1 - |P - tau| / max(tau, 1 - tau)            challenge fit (assumption: desirable difficulty)
I = relative posterior-variance reduction          information value (statistical; about the estimate, not about learning)
C = 1 if the chosen candidate adds an unseen tag-variation signature for (c, level), else 0
F = 1 while a follow-up for c is open, else 0       (R19)
K = 1 if c had the latest ability observation and its Medium cell is not yet ready, else 0   (continuity)
B = 0.5*D + 0.3*I + 0.2*C + 0.3*F + 0.2*K
```

F and K are concept-level terms. They decide which concept gets focus and cannot change the order of actions within a
concept. The action and its candidate are chosen by D, I and C. P is the posterior mean (section 6 explains why a lower
bound is not used).

Candidate within an action: the LeetCode candidates in one cell are evidence-equivalent, so R16 applies directly. For
CF/AtC the order is closest `p_mean` to tau, then R16.

## 6. Conservatism and behaviour (exec, proposal reference)

`adaptive_sim.py` (`receipts/adaptive_sim.out`): scoring step-ups at the 10% quantile deadlocks (Medium is never chosen
after Easy 5/5). v2 therefore scores the posterior mean, and the conservatism comes from shrinkage (kappa), partial
carry-over (lambda < 1), leave-concept-out level means, and immediate step-down after failures.

`adaptive_focus_sim.py` (`receipts/adaptive_focus_sim.out`, single concept A vs a fresh concept, cold CF prior):

| Scenario (A history) | Winner | Key numbers |
|---|---|---|
| none | A consolidate Easy | P 0.80, B 0.689 |
| Easy 2 solved | A step_up Medium | P 0.56 vs Easy 0.87 |
| Easy 2, Medium 2 solved | A transfer CF 800 | P 0.67; Medium cell ready |
| Easy 2, Medium 0 of 1 (follow-up open) | A step_down Easy | Medium P falls to 0.45 |
| same, follow-up expired | A step_down Easy | follow-up affects focus, not action |
| Easy 1 of 3 | A consolidate Easy | P 0.60 |
| Easy 2, Medium 4, transfer done | fresh concept | A ready → K = 0; A best B 0.676 < 0.689 |

These reproduce the intended workflow: another Easy, an accessible Medium, back to Easy after a failure, then move on.
Sensitivity (`adaptive_sim.out`): consecutive Easy solves before a step-up is 1–4 for tau = 0.7 across kappa 2–8 and
lambda 0.25–1. Hard is rarely chosen under these weights (P ≈ 0.2–0.37 gives low D). This is a known property, revisited
with data. The T8 receipt must repeat these checks on the real catalog.

## 7. Readiness, prerequisites, follow-up, reviews

- Readiness (R17): `Pr(theta ≥ 0.70) = 1 - I_{0.70}(alpha, beta) ≥ 0.5` (posterior median ≥ tau) and ≥1 own ability
  observation. A prerequisite r of c is ready when r's **Easy** cell is ready, recomputed excluding problems that also
  carry c. The transfer gate (R15) and the continuity release (K) use c's own **Medium** cell. A concept with no
  LeetCode problems at the required level (e.g. constructive) has that requirement waived, so its CF/AtC actions remain
  available.
- Waiver: a required cell with no unattempted eligible candidate is waived (`waived: no_catalog_coverage` or
  `exhausted`). Combined with an acyclic authored graph, every concept is reachable, either through practice that can move
  its prerequisites or through a waiver. A target concept bypasses readiness.
- Deadlock check (T6/T8): a test asserts that for any ledger, if any concept has a candidate for any action, the plan
  contains a recommendation.
- Follow-up (R19): opened by any failed observation; closed by a later solved observation in c at level ≤ the failed level
  (including re-solving the same problem), or expired after 14 days. One per concept (latest failure).
- Reviews (`reviews-v2.mjs`, R20): stage interval `review_days[stage]`. First ability success → stage 0, due +1 day.
  Retention success with gap ≥ current interval → stage+1; earlier success → same stage, due restarted; any failure →
  stage 0, due +1, priority. The review lane and the 복습 targets are listed as today. Reviews never compete with the
  progression item.

## 8. Explanations and transitions

Slot (journal `next`, kind `practice`/`transfer`):

```text
{policy:'progression-v2', config_version, config_sha256, catalog_version, concept, action, problem_key, platform, level|rating,
 estimate:{p, interval:[lo,hi], n_own, n_level_outside, prior:'authored'|'learner'},
 components:{D,I,C,F,K}, weights, benefit_proxy:B,
 readiness:{medium:{state, probability}, prerequisites:[{concept,state,probability|waiver}]},
 follow_up:{open, failed_observation_id, expires}, coverage:{action: 'ok'|'insufficient_catalog_coverage'},
 tie_break:{coverage_gain, purity, hash}, alternatives:[top 3 other (concept,action,B)],
 reason:'<one sentence built only from the fields above>', limitations:['estimates are not calibrated', 'benefit is a heuristic']}
```

Transitions (journal `mastery_change`, v2): the concept signature is `{best_action, best_level, medium_ready,
prerequisites_state, follow_up_open}`. It is evaluated at each evidence prefix. A change emits an entry with the observation
id and the affected cell estimates before and after.

## 9. Methods: justified now vs later

| Justified now | Needs more evidence |
|---|---|
| Beta-binomial cells, partial pooling, exact intervals | Calibration claims (≥100 first-attempt pairs per platform, chronological) |
| Deduplicated Laplace logistic for CF/AtC | Concept offsets on CF/AtC; LeetCode within-level difficulty |
| Exact information value | Learning-gain models (BKT, AFM), bandits, off-policy evaluation |
| Documented deterministic tie-breaks; coverage and waivers | Time as a performance signal; fitted forgetting curves |

## 10. Rollout and validation

Order: tasks T1–T7 behind `progressionVersion:'v1'` (default), T8 replay and sensitivity receipt, then T9 switches the
default to v2. Validation: Node unit tests against a scipy-generated reference fixture, scenario fixtures from section 6,
property tests (determinism, alias/rename invariance apart from the seeded hash, repeat invariance, review-flag and time
invariance, no deadlock), byte-identity of v1 journal output, the full Node suite, and the real-catalog replay. None of
these measure learning benefit.
