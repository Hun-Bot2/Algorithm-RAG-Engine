# Adaptive progression: requirements (progression-v2)

Status: accepted 2026-10-29 (ADR 006), implemented T1-T9 (`tasks.md`, `validation.md`). Governs `progression-v2` only; v1 remains
for history. Decisions: ADR 006 D1–D11. Provisional assumptions are marked (P) and SHALL NOT be described as optimised.

## Input and preservation

- R1 THE SYSTEM SHALL accept the daily Markdown log unchanged: heading `# YYYY-MM-DD`; table `| 문제 | 결과 | 시간 | 복습 | 메모 |`
  (시간, 복습 optional); 결과 `1` solved, `0` failed, blank or `-` unrecorded. No new column, trailer or syntax SHALL be required.
- R2 The 복습 flag SHALL only request a future review (existing `review_targets`). It SHALL NOT be treated as a failure, a
  success, or ability evidence, and SHALL NOT change any estimate.
- R3 메모 and free-text sections SHALL NOT be read as evidence.
- R4 Ledgers, Markdown files, review targets, snapshots, backups and progression-v1 outputs SHALL remain byte-identical. v2
  SHALL be a pure derived view with its own version; no schema migration. v1 SHALL stay selectable.

## Evidence

- R5 Per problem, the first dated attempt SHALL be an ability observation; each later dated attempt SHALL be a retention
  observation carrying its actual outcome and the gap in days since the previous dated attempt (D7).
- R6 Several attempts on one problem on one date SHALL collapse to the latest one by `occurred_at`, then id.
- R7 Declared activity types (`review`, `transfer`) SHALL NOT be required; v2 classifies from dates.
- R8 Self-reported time SHALL be carried and shown; it SHALL NOT change any estimate or decision (D5). Missing time is unknown.
- R9 Ability estimates SHALL use ability observations only. Retention observations SHALL drive spacing and follow-up only.
- R10 A problem with k trusted concepts SHALL contribute weight 1/k to each concept's LeetCode cell (D3, P).
- R11 Platform abilities SHALL never be converted across platforms.

## Estimates and decisions

- R12 Every estimate SHALL report its posterior mean, central 80% credible interval, own observation count and prior source.
- R13 For the focus concept THE SYSTEM SHALL compare actions step_down, consolidate, step_up, stretch and transfer by a
  benefit proxy built from challenge fit, information value, coverage gain, follow-up and continuity (D6, P). The
  estimated solve probability and the benefit proxy SHALL be shown separately; the proxy SHALL NOT be called a probability
  or a learning gain.
- R14 Level changes SHALL result from the estimates, never from attempt counts. `stretch` SHALL require at least one
  ability observation in the concept.
- R15 `transfer` SHALL be offered only when the concept's LeetCode Medium cell is ready (R17) or waived, and only until
  the concept has one solved transfer ability observation. Its rating SHALL target tau under that platform's own model.
- R16 Ties between candidates SHALL be broken by coverage gain, then concept purity (fewer trusted concepts), then
  FNV-1a of `learner|date|concept|problem_key`. Alphabetical order SHALL NOT decide.

## Readiness, follow-up, reviews

- R17 A cell is **ready** WHEN Pr(theta ≥ tau | posterior) ≥ 0.5 AND it has ≥1 own ability observation; for prerequisite
  readiness, observations on problems that also carry the dependent concept SHALL be excluded. A requirement is **waived**
  WHEN its cell has no unattempted eligible candidate; the waiver SHALL be visible. A user target concept SHALL bypass
  prerequisite readiness with a note (D10).
- R18 A concept SHALL be eligible for focus WHEN all its authored prerequisites are ready or waived. The authored graph
  SHALL be validated as acyclic; no state SHALL exist in which every concept is ineligible while candidates remain.
- R19 WHEN an ability or retention observation fails, THE SYSTEM SHALL (a) keep the next-day priority review of that problem
  and (b) open one follow-up for the concept at that level. The follow-up SHALL close on a later solved observation in the
  concept at the same or a lower level, or expire 14 days after the failure (P). No permanent override SHALL exist (D8).
- R20 v2 review spacing: a retention success whose gap ≥ the current interval SHALL advance the stage; an earlier success
  SHALL keep the stage and restart the interval; a failure SHALL reset to stage 0, due next day, priority (D7).
- R21 WHEN an action has no eligible candidate, the output SHALL report `insufficient_catalog_coverage` for it. No
  title-embedding or other non-concept fallback SHALL be used (D9).

## Explainability and determinism

- R22 Every v2 recommendation SHALL record: policy and config hash, catalog version, concept, action, problem, P with
  interval and counts, every benefit component and weight, readiness and waivers, follow-up, coverage statuses, the
  tie-break values, and the next-best alternatives.
- R23 Every progression transition SHALL be derivable at evidence prefixes and SHALL name the observation that caused it,
  with the affected estimates before and after.
- R24 Output SHALL be deterministic for identical ledger, catalog, preferences, date, config and code.
- R25 All tau, prior, kappa, lambda, weight, interval and follow-up constants SHALL live in one versioned config file.

## Clarification accepted 2026-10-29: fresh concept priors

User decision during T6: a concept with no own LeetCode ability observation at any level SHALL retain authored level priors even when other concepts have learner evidence. Once it has at least one own ability observation, leave-concept-out level updates apply. Outside counts remain visible; prior source stays authored for a fresh concept. This supersedes the unqualified D2 / design section 4.1 update rule.
