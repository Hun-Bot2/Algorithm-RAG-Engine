# ADR 006: Adaptive progression (progression-v2)

Status: accepted (user decisions), 2026-10-29. Implemented T1-T9 (`specs/adaptive-progression/tasks.md`, validation.md).
Spec: `specs/adaptive-progression/`. Evidence: `research/recommendation_audit/REPORT.md`.

## Context

progression-v1 advances through fixed stages by counting distinct problems (3 Easy, 2 Medium, then transfer and retention).
It picks Easy/Medium problems in alphabetical slug order. Its transfer and retention stages need declared activity types
that the daily Markdown log never produces, so they cannot complete from the learner's real input. Several concepts
stall because of catalog coverage. The learner wants progression to stay the core experience and to adapt to their
performance. The Markdown format stays unchanged.

## Decision

Add `progression-v2`, a pure derived policy that chooses per-concept actions (consolidate, step up, stretch, transfer)
from first-attempt evidence, uncertainty, coverage and failure follow-up. v1 stays available and its history is unchanged.

| ID | Decision | Status |
|---|---|---|
| D1 | Target success rate tau = 0.70 (config) | Approved |
| D2 | Level priors: authored Beta(3.2,0.8)/(2,2)/(0.8,3.2), retained for fresh concepts; after own LeetCode evidence, updated per level by leave-concept-out learner means | Provisional assumption |
| D3 | Multi-concept attribution weight 1/k | Provisional assumption |
| D4 | Shrinkage kappa = 4, Easy→Medium→Hard carry-over lambda = 0.5 | Provisional assumption |
| D5 | Self-reported time is stored and displayed; it does not affect decisions | Approved |
| D6 | Benefit proxy weights w_D 0.5, w_I 0.3, w_C 0.2 (+ follow-up 0.3, continuity 0.2), components always shown | Provisional assumption |
| D7 | A later-date re-attempt is a retention observation with its actual outcome and gap. Spacing advances only on a success whose gap reached the current interval; a failure resets to priority review. The Markdown 복습 flag is only a user request for future review and is never evidence | Approved (refined) |
| D8 | No permanent remediation override. A failure creates (a) the existing next-day priority review of the problem and (b) a bounded concept follow-up that raises the concept's focus score until resolved by a later solve in that concept at the same or lower level, or until it expires after 14 days | Approved (refined); 14 days provisional |
| D9 | No title-embedding fallback in v2; report missing concept-backed candidates | Approved |
| D10 | Readiness of a cell = posterior Pr(theta ≥ tau) ≥ 0.5 **and** at least one own first-attempt observation, counted from problems that do not also carry the dependent concept. A requirement whose cell has no unattempted eligible candidate is waived with a visible status; a user target concept bypasses readiness with a note | Approved (refined) |
| D11 | No within-level LeetCode difficulty in v2 | Approved |

Provisional assumptions (D2, D3, D4, D6, follow-up window) carry no claim of empirical optimisation. They are versioned
in one config file, shown in every explanation, and covered by the sensitivity receipt (task T8).

## Rejected alternatives

- Tuning v1 thresholds: keeps count-based advancement and the unreachable stages.
- Lower-bound (pessimistic) scoring of step-ups: computed to deadlock (Medium never tried after Easy 5/5;
  `research/recommendation_audit/receipts/adaptive_sim.out`).
- Hard prerequisite gates without waivers: reproduce v1 dead ends (union-find, heaps, shortest-paths, ...).
- Bandits / learned benefit models / time-aware IRT: too little data (9 real attempts); deferred.
- Adding Markdown columns or trailers for activity type: rejected by the learner; activity is derived from dates.

## Consequences

- Recommendations change when v2 becomes the default (journal plans rotate once; earlier snapshots remain history).
- Review dates for re-solved problems change under v2 spacing (D7). The v1 review replay in `core/policy.mjs` is not
  modified; v2 uses its own replay.
- Calibration, learning benefit and transfer effects remain unmeasured; explanations must say so.

## Clarification accepted 2026-10-29: fresh concept priors

User decision during T6: a concept with no own LeetCode ability observation at any level SHALL retain authored level priors even when other concepts have learner evidence. Once it has at least one own ability observation, leave-concept-out level updates apply. Outside counts remain visible; prior source stays authored for a fresh concept. This supersedes the unqualified D2 / design section 4.1 update rule.
