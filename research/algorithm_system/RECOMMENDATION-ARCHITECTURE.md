# Deterministic recommendation architecture proposal

This is a research proposal grounded in the repository's existing local
learning and hosted-policy code. It does not change production behavior, add
model inference, or claim that any hosted scheduler or Slack delivery is
currently enabled. The proposal keeps candidate evidence, learner policy, and
delivery as separately versioned stages.

## Existing implementation inspected

The hosted learner currently builds a daily mission with
[`hosted/core/policy.mjs`](../../hosted/core/policy.mjs). It derives problem and
concept states from active attempt events, gates candidates on approved/free
curriculum entries, checks approved prerequisite edges, adds deterministic
score components, and breaks ties by problem key. It freezes a daily mission
using the `deterministic-v2` policy version. `REVIEW_DAYS` is `[1, 3, 7, 14,
30, 60]`. A successful practice schedules the first review after one day;
successful reviews advance through the later intervals. A non-independent
review restarts the one-day interval.

[`hosted/core/relationships.mjs`](../../hosted/core/relationships.mjs) builds
three bounded candidate sources: reviewed curriculum graph edges, concept-set
Jaccard overlap, and lexical overlap from independently authored annotations
and concepts. Each provider returns at most 20 candidates per source problem.
`fuseCandidates` uses reciprocal-rank fusion with `k=60`, retaining provider
evidence. Its module comment correctly says candidate signals do not establish
mastery. The candidate providers use the curriculum version; they are currently
computed from curriculum data rather than stored as a separate immutable
candidate-bundle artifact.

The hosted policy's score has explicit components for target path, learner
state, suitability, variety, review priority, and relationship rank. It uses
the weekly rhythm from
[`hosted/core/preferences.mjs`](../../hosted/core/preferences.mjs); the default
weekday budget is 60 minutes, and the policy also sets Saturday and Sunday to
60 minutes. Daily output is an ordered exposure list plus the activities that
fit the budget. The current policy has practice, review, and transfer activity
types. It does not yet model progressive challenge as a distinct type or
prerequisite repair as a separate learner-facing mode.

[`src/learning/curriculum.py`](../../src/learning/curriculum.py) validates
versioned curriculum documents, allowed platform URLs, access states, problem
and relationship review records, and acyclic prerequisites. Approval requires
a reviewer and evidence; an approved relationship requires approved endpoint
problems. Release validation requires at least 90 approved, explicitly free
problems and approved relationships. Native difficulty is presently a required
nonempty string in this v1 contract, so a missing value may only be represented
by a sentinel such as `unknown`.

That sentinel is unsafe for the current hosted difficulty helper:
`difficultyBand` in `hosted/core/policy.mjs` first consults a manual annotation,
then maps Easy/Medium and a narrow ABC-A/ABC-B pattern, and classifies all other
strings as `advanced`. The proposal below must preserve missingness explicitly
when a versioned difficulty contract is introduced; it must not treat an
unknown value as advanced.

The offline research profile in
[`src/learning/research_profiles.py`](../../src/learning/research_profiles.py)
reads only platform key, native difficulty, concepts, and allowlisted authored
title/rationale text. Its `MetadataProvider` uses fixed weights
`0.60 * concept Jaccard + 0.25 * same platform + 0.15 * same native difficulty`.
That research scorer compares raw native-difficulty strings without checking
that both values share a platform scale. It is a baseline implementation, not
a valid cross-platform difficulty bridge. The hosted policy's relationship
providers are a separate implementation and do not use this fixed-weight
score.

The evidence model in [`src/learning/events.py`](../../src/learning/events.py)
and [`hosted/core/events.mjs`](../../hosted/core/events.mjs) distinguishes
manual attempts from platform imports. Outcomes include independent,
assisted, failed, and reported completion; imported submissions are
reported-completion evidence only. An independent attempt requires explicit
`hints: false`. Corrections supersede earlier events. Timer start, stop, and
confirmation are separate manual events. Only a stopped interval with an
explicit confirmed start/end contributes confirmed elapsed time; an open or
unconfirmed timer has unavailable duration. Submission timestamps do not
substitute for study time.

[`hosted/scheduler.mjs`](../../hosted/scheduler.mjs) freezes a mission at the
configured daily cutoff. [`hosted/dispatcher.mjs`](../../hosted/dispatcher.mjs)
and [`hosted/notifications.mjs`](../../hosted/notifications.mjs) format and
deliver that frozen mission through Slack behind delivery configuration and
receipt tracking. This path does not call an LLM. Provider synchronization in
[`hosted/provider-service.mjs`](../../hosted/provider-service.mjs) retrieves
contest or submission metadata from supported platforms; that is external data
sync, not recommendation-model inference. Code, enabled flags, scheduler
invocation, accepted Slack sends, and an observed learner notification remain
separate evidence states.

## Proposed stage boundaries

### Versioned bounded candidate evidence

Candidate generation should publish an immutable, bounded set per source
problem. A bundle should include a schema version, bundle ID, generator and
configuration versions, source-profile/catalog hashes, taxonomy version,
difficulty snapshot ID, and candidate rows. Each row should carry its stable
problem key, provider rank, provider score in its own named score space,
evidence references, source platform, review/access state, and explicit
missingness. A provider may contribute at most a configured limit (20 matches
is the current hosted limit). Bundle construction must not contain learner
events, preferences, delivery state, or learner identifiers.

Candidate signals only identify possibilities. Similarity, lexical overlap,
native score proximity, and a precomputed embedding rank do not establish
mastery or prerequisites. Candidate bundles should be reproducible from their
versioned inputs and replaceable without rewriting learner history.

### Deterministic learner policy

The policy should consume a bundle version, a curriculum version, normalized
learner preferences, and the active event stream through an explicit cutoff. It
should emit an explainable ordered set of recommendation modes with score
components and event or reviewed-edge evidence. Its policy version and input
cutoff belong on every mission snapshot. Candidate generation can change
independently from policy logic; policy can be replayed independently from
delivery.

The policy should keep the current repository principles: stable tie-breaking,
bounded candidate work, explicit reason fields, evidence-derived progress, and
no generated content in a recommendation. Unreviewed problem or relationship
records are not eligible. Similarity and authored metadata may order otherwise
eligible candidates, but must never silently create prerequisite edges.

## Four recommendation modes

1. **Equivalent practice.** Offer another problem with a reviewed similarity
   relation or sufficiently supported primary-concept overlap. Preserve the
   candidate provider and evidence that caused the match. Similarity alone does
   not change mastery.
2. **Progressive challenge.** Offer a harder next problem only after explicit
   learner evidence demonstrates the current level and the candidate has
   comparable concept evidence. Difficulty ordering must be source-local. A
   source-native value or provisional percentile may support an ordering only
   within its declared source scale. If uncertainty intervals overlap, the
   estimate is missing, or the candidate comes from an uncalibrated source,
   suppress the claim that it is harder; it can remain an equivalent-practice
   candidate. Never interpret equal percentiles on two platforms as equal
   ability.
3. **Prerequisite repair.** Select only a prerequisite edge with approved
   review, approved endpoints, and eligible access. If the prerequisite lacks
   demonstrated or transfer-evidenced state, offer that prerequisite as the
   next task with the exact edge and evidence attached. Similarity, tag
   ancestry, title similarity, or model output cannot invent this edge.
4. **Spaced review.** Offer a demonstrated problem when its evidence-derived
   due date is reached. Keep the existing interval sequence of 1, 3, 7, 14,
   30, and 60 days unless a separately reviewed policy version changes it.
   Due-state and source event IDs must be visible in the explanation.

If multiple modes compete, the policy must record the selected mode and the
components that determined it. A reasonable initial deterministic precedence
is due review, prerequisite repair, equivalent practice, then progressive
challenge, with stable key ordering inside a tie. This precedence is a proposal
for offline policy tests, not current hosted behavior.

## Difficulty, concepts, and missing data

Difficulty evidence should be represented with the raw value and its source
scale, rather than as a bare number. A future record should include a
`difficulty_scale_id`, native raw value, source snapshot/version, estimate
method, cohort size, percentile tie interval, and any bootstrap interval. For
LeetCode, Easy/Medium/Hard is an ordinal category distribution. Codeforces uses
its rating points. AtCoder's `problem-models.json` values are estimates from an
unofficial community dataset and are not an AtCoder-native rating. Its negative
values remain raw evidence. The available empirical ranks and row-bootstrap
uncertainty are documented in [DIFFICULTY.md](DIFFICULTY.md) and
[calibration.v1.json](calibration.v1.json).

Percentiles are provisional within-source summaries, not cross-source
calibration. The bootstrap intervals only describe variation from resampling
the observed catalog rows; they do not cover rating-process uncertainty or
catalog selection. Any cross-source bridge would require separate independently
reviewed difficulty-order judgments, source/family-disjoint calibration and
final partitions, and held-out validation. The current 3,647 AI-reviewed
similarity pairs are not difficulty judgments and remain evaluation-only.

Concept features must retain raw source tags, taxonomy mapping relation and
confidence, taxonomy version, and ancestors as a separate field. Only reviewed
equivalent/narrower concepts with sufficient confidence may form primary
concept features. Related, unmapped, unknown, empty, and unavailable states
remain distinguishable. AtCoder tags are unavailable in the current catalog;
do not fill them from statements or titles. The existing statement corpus has
source-specific availability and identity constraints; absent statements
remain absent, and a statement-derived feature is available only for an
identified, rights-reviewed row with a retained source version/hash.

Unknown difficulty, tags, or statements should not be converted to zero,
advanced, unrelated, or a negative label. The policy may continue with other
available candidate evidence while exposing what was unavailable. If the mode
requires a missing field—for example progressive challenge without comparable
difficulty evidence—the policy should abstain from that mode and explain why.

## Events, mastery, timing, and delivery

Mastery remains a deterministic summary of explicit active events, not a model
prediction. An imported completion can establish that a platform reported a
submission, but not that it was independent, hint-free, understood, or timed.
For independent demonstration, require the existing explicit no-hints outcome.
Keep event ID, occurrence time, recording time, source, and supersession
provenance available to explanations and replay.

Elapsed study time is usable only after a learner explicitly starts, stops, and
confirms an interval. Overlapping confirmed intervals should use the existing
union rule to avoid double counting. Open, stopped-unconfirmed, or unavailable
intervals contribute no inferred duration.

The current default weekday budget is 60 minutes. The existing hosted policy
also fixes weekends at 60 minutes. A policy snapshot must record the effective
budget and weekly focus. Daily mission creation, frozen snapshot, outbox state,
Slack acceptance, and observed delivery are separate events. Delivery formats
the deterministic mission and invokes no AI service.

No reinforcement learning, reward optimization, or online weight updates are
proposed. A later policy revision should be an explicit versioned change with
offline evaluation and rollback evidence.

## Proposed contract example

The following is a synthetic interface example only. `fixture_*` keys and event
IDs are authored placeholders; this is not a real learner history.

```json
{
  "contract_version": "recommendation-set.v1",
  "candidate_bundle": {
    "bundle_id": "fixture-bundle-01",
    "generator_version": "fixture-generator-v1",
    "catalog_sha256": "fixture-catalog-hash",
    "taxonomy_version": "algorithm-taxonomy-fixture-v1",
    "difficulty_snapshot": "native-difficulty-fixture-v1"
  },
  "policy": {
    "policy_version": "deterministic-proposal-v1",
    "curriculum_version": "fixture-curriculum-v1",
    "as_of": "2030-01-08T00:00:00Z",
    "timezone": "Asia/Seoul",
    "effective_budget_minutes": 60
  },
  "learner_evidence": [
    {
      "event_id": "example-only-event-01",
      "kind": "attempt",
      "outcome": "independent",
      "hints": false,
      "occurred_at": "2030-01-07T10:00:00Z"
    }
  ],
  "recommendations": [
    {
      "rank": 1,
      "mode": "equivalent_practice",
      "problem_key": "atcoder:fixture_next",
      "candidate_evidence": ["fixture-provider:rank-2"],
      "learner_evidence": ["example-only-event-01"],
      "difficulty": {
        "scale_id": "atcoder-problems-community-fixture-v1",
        "raw_native_value": null,
        "within_source_percentile_interval": null,
        "uncertainty": null,
        "status": "unavailable"
      },
      "missingness": {
        "tags": "unavailable",
        "statement": "not_in_metadata_catalog"
      },
      "explanation": {
        "reason_code": "challenge_suppressed_missing_comparable_difficulty",
        "score_components": {"concept_overlap": 0.5, "relationship_rank": 2}
      }
    }
  ],
  "delivery": {
    "status": "not_attempted",
    "channel": null,
    "receipt": null
  }
}
```

An implementation should represent a selected mode once; the example shows the
reason why a challenge claim was suppressed and an equivalent-practice fallback
was chosen. It must not report the absent difficulty estimate as a score.

## Validation plan before implementation

- Replay the same versioned candidate bundle, curriculum, preferences, and event
  cutoff twice; require identical candidate ordering, mode, reasons, components,
  and output hashes.
- Change one candidate-bundle version while keeping learner events fixed; verify
  only candidate evidence changes and no learner history is rewritten.
- Verify approved/free eligibility and approved relationship endpoints. Reject
  unreviewed prerequisite edges, duplicate edges, cycles, and stale bundle or
  taxonomy versions.
- Check that equivalent-practice evidence never marks a problem or concept as
  mastered. Check that only reviewed prerequisite edges produce prerequisite
  repair.
- Exercise source-local native category/numeric order, tied empirical CDF
  intervals, bootstrap uncertainty, missing source values, and overlapping
  uncertainty intervals. Verify no cross-source percentile equality is used as
  an ability comparison.
- Preserve unavailable versus empty tags, unknown tags, missing statements,
  missing difficulty, and reported-completion-only imports through ranking and
  explanation output.
- Replay event cutoff and supersession cases; verify independent mastery still
  requires explicit `hints: false`. Verify timer duration stays unavailable
  until confirmation and overlapping confirmed time is counted once.
- Verify review due dates across 1/3/7/14/30/60-day stages, reset behavior,
  Asia/Seoul date boundaries, weekly focus, the 60-minute weekday budget, and
  stable order when scores tie.
- Exercise mission freezing and delivery separately. Verify the deterministic
  mission formatter makes no AI call and that disabled delivery produces no
  external Slack request. Treat Slack receipts and observed user receipt as
  distinct validation evidence.
- Compare recommendation modes against a frozen, independently reviewed,
  query/family/source-disjoint evaluation set. Report abstentions, coverage,
  uncertainty, and source-specific metrics. The existing metadata-pooled
  semantic labels must remain evaluation-only for difficulty.

This plan describes future validation gates. It does not claim those tests,
human reviews, source-disjoint sets, browser checks, or delivery observations
have been completed by this architecture document.
