# ADR 005: Solved / unsolved outcomes without help tracking

Status: accepted (user decision), 2026-10-28. Spec: specs/mastery-system/solved-only-outcomes.md.

Context: Policy v2 counted only `independent` attempts with explicit `hints: false` as success. Daily logs write `1`
for a solved problem, which imported as `reported_completion` and therefore never advanced stages or calibration.
The learner decided that whether help was used is a matter of personal honesty, cannot be verified or controlled,
and should not be part of the system.

Decision: Policy `mastery-policy-v3` treats every completion (`independent`, `assisted`, `reported_completion`) as
solved and `failed` as unsolved. The projection happens once, in `attemptStream` (hosted/mastery/ledger.mjs), which
every learner-state consumer already reads. Consumers see `outcome: 'independent'`, `hints: false` for solved
attempts. The stored evidence is not rewritten: each projected attempt keeps `recorded_outcome` and
`recorded_hints`. The UI and recording forms offer only solved / unsolved and no longer show help.

Consequences: Earlier `1`, `assisted` and trailer-less Git completions now count toward stages, calibration and
reviews. Solve-rate metrics now mean "recorded as solved", not "solved without help". Stored rows, validators and
backups are unchanged, so the decision can be reversed by a new policy version without data migration. The Worker
mission path (hosted/core/policy.mjs) reads raw events and is out of scope.

Rejected: rewriting stored outcomes (violates append-only evidence); changing ingest so `1` writes `independent`
(would assert a fact the learner chose not to record, and leave older rows inconsistent).
