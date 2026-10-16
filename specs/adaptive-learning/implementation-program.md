# Remaining adaptive application implementation program

## Research summary

The original batch path is preserved. New local event storage and curriculum
validation are implemented; there is no existing hosted application to extend.
Node 26 and standard-library SQLite are available for local integration tests.
Cloudflare Worker assets must run through the Worker first to authenticate static
files as well as APIs. Access requires signature, issuer and audience validation.
D1 batch statements provide transactional mutation; tests must exercise SQL.
Slack's scheduleMessage metadata failure remains documented as of this inspection.

Sources inspected 2026-10-16:
- https://developers.cloudflare.com/workers/static-assets/binding/
- https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/authorization-cookie/validating-json/
- https://developers.cloudflare.com/d1/worker-api/d1-database/
- https://docs.slack.dev/reference/methods/chat.scheduleMessage/
- https://codeforces.com/apiHelp
- https://github.com/kenkoooo/AtCoderProblems/blob/master/doc/api.md

## Requirements

Implement the remaining M4-M10 code under the existing no-AI specification.
User accepts Luna drafts followed by independent Sol review as the curriculum
review process. Record the reviewers and source boundaries accurately; this
amends the earlier human-only interpretation, not empirical learning evidence.
External deployment, actual Slack sends, scheduling activation, paid compute and
training still require their specific gates; prepare all reviewable code first.
The user requires stopping and waiting whenever a user decision is necessary.

## Design proposal

Use a single private Worker with static assets and D1; deterministic pure JS
policy is shared by Worker and local Node execution. Keep Python evidence tools
and optional research offline. A local Node D1-compatible SQLite adapter exercises
real migrations and APIs without introducing an unauthenticated production mode.
All release data is independently authored minimal curriculum annotations rather
than copied problem statements or historical restricted vectors.

## Task breakdown

1. Luna drafts 90 problems and 180 evaluation judgments; Sol independently reviews.
2. Sol implements event parity, D1 ownership/authentication and deterministic policy.
3. Luna implements responsive Today/Progress/Contests and explicit evidence entry.
4. Add cached provider adapters, manual fallback and provenance-bearing imports.
5. Add Slack formatter, durable outbox, ambiguous-send reconciliation and dispatcher.
6. Integrate local service and generated curriculum, verify full flows.
7. Add offline frozen benchmark and training/shadow gates; no automatic adoption.
8. Prepare private deployment, quota/backup instrumentation and release checklist;
   stop for account/identity/channel decisions before external activation.

## Validation plan

Python and Node offline suites, real SQLite migration/atomicity tests, golden
learner scenarios, ownership/JWT rejection, notification fault injection and asset
hash checks. Verify representative mobile flows in an actual local browser where
available. Distinguish Node/static checks from browser and deployed-runtime proof.
Benchmark labels remain evaluation-only; historical data remains unchanged.

## Risks

Curated content may have uncertain live access; exclude unknown access from
missions. AI review can be mistaken; retain rationales, disputes and versioning.
D1/Worker quotas and workstation-off delivery require deployed measurements.
30-day delivery and longitudinal learning benefits cannot be inferred from local
implementation. Model training cannot pass without independent training labels
and an owned-compute decision; implement gates rather than fabricate completion.

## First implementation step

Parallel isolated ownership: curriculum draft (Luna), hosted policy/auth/storage
(Sol), dashboard (Luna). Parent reviews and integrates providers, dispatch, research
and verification. Existing dirty Ralph state and unrelated tests stay untouched.
