# Adaptive learning design v1

## Architecture and migration boundary

Permitted catalog + independently authored annotations -> reviewed graph and
bounded relationship candidates -> deterministic policy + private events ->
mission snapshot -> dashboard and delivery outbox -> new learner events.
Optional offline models produce versioned candidate lists or unapproved drafts.
The hosted Worker never invokes models. Access provides authenticated identity;
D1 stores events/projections/outbox. Static assets contain only allowlisted data.
The current Python heavy/light batch remains a compatibility reference.

M1-M3 implement offline foundations in `src/learning`, using only Python standard
library SQLite. D1 porting, authentication, dashboard and network integrations
remain M6-M8 gates. Local storage APIs are trusted-process boundaries, not an API
server. Keep private databases and exports outside Git and hosted static assets.

## Replaceable contracts

- RelationshipProvider: query key, asset version -> candidates with key, evidence,
  score components, namespace. First providers: reviewed graph, metadata, lexical.
- EmbeddingBackend: pinned manifest + permitted text -> offline vectors in exactly
  one namespace. Historical OpenAI vectors have no free encoder for arbitrary text.
- AnnotationAssistant: provenance-bearing drafts; no approval or event-write access.
- LearningPolicy: catalog/graph, candidates, preferences, event cutoff -> versioned
  mission, target-gap/weakness/suitability/variety scores and exposure record.
- ExplanationRenderer: templates over supported evidence; optional rewriting cannot
  change selection or evidence.
- NotificationTransport: immutable mission + uniqueness key -> reconciled delivery
  states, independent of ranking. No inference on the deadline path.

## Storage contract and flow

`migrations/001_events.sql` creates schema-version history, learners and append-only
events. Migration SQL SHA-256 is stored and checked on open; unknown versions or
modified applied migrations fail closed. New migrations must be additive and
preceded by private backup/export. Rollback uses the prior application plus a
verified export; never destructively downgrade a database.

Events contain schema_version=1, event_id, learner_id, occurred_at, recorded_at,
kind, source, payload and optional supersedes. UTC timestamps use canonical
`YYYY-MM-DDTHH:MM:SS.ffffffZ`; lexical sort then event ID is deterministic.
`source` is manual or platform_import; imported attempts require platform provenance.
Kinds and payloads:

| Kind | Payload |
| --- | --- |
| attempt | problem_key, outcome (independent/assisted/failed/reported_completion), hints (bool or null) |
| timer_started | interval_id, problem_key |
| timer_stopped | interval_id |
| timer_confirmed | interval_id, started_at, ended_at |

Attempt payloads additionally allow explicit `activity_type` (practice/review/transfer);
transfer requires a different platform-qualified `transfer_from` key. Omission means
unspecified context, not an assumed review or transfer. This additive v1 payload
extension does not alter existing SQL or original events.

Problem keys use `leetcode:slug`, `atcoder:task_id`, `codeforces:contestIdIndex` (for example `codeforces:4A`),
or `baekjoon:id`. Stop and confirmation require prior active start evidence;
confirmation also requires stop evidence. Confirmation may correct an interrupted
interval after explicit stop; end cannot exceed confirmation occurrence time.
Open/stopped-only timers contribute zero confirmed seconds with explicit status,
not a claim of zero actual study. Confirmed elapsed time unions overlapping
intervals. Corrections replay after removing superseded records; duplicate timer
actions or dangling confirmations fail before append/import commits. Corrections
must preserve timer identity, cannot branch and cannot change event kind.

Export is a versioned JSON object with learner and all events (including superseded
records). Import validates the entire document and executes in one transaction;
existing owner mismatch, conflicting IDs or invalid replay rolls everything back.
No untrusted pickle or executable SQL backup is imported. A sequence field is
unnecessary: export uses append row order for correction dependencies and replay
uses occurrence/event ID order. Event IDs are globally unique across learners.

## Assets, rights and namespaces

`assets/source_registry.v1.json` stores byte hashes and explicit absent assets.
`assets/legacy_numpy.manifest.v1.json` stores ordered keys and historical unknowns.
`scripts/learning/assets.py` verifies originals without reading models or regenerating
vectors; bundle mode copies only explicit approved assets into a new directory,
never whole data/index directories. This initial registry permits no historical
asset into hosted bundles. Approval requires a reviewed registry revision.
Known indexed queries may later use cached neighbors; unknown keys fall back to
metadata/lexical evidence. Future local encoder namespaces cannot reuse this index.

## Tradeoffs and risks

Append-only evidence costs storage but supports correction audits and replay.
SQLite offers deterministic local tests without new dependencies; passing them is
not D1/Access/runtime proof. Historical unknown model/input provenance blocks
inference compatibility claims. Human curriculum review cannot be automated away.
Optional models add maintenance/annotation/compute cost and need independent gates.

## Validation

Run `python3 -m unittest discover -s tests -v` and
`python3 scripts/learning/assets.py verify`. Validate idempotency, conflicting IDs,
corrections, ownership, invalid payloads, timestamp order, timer interruption and
union, concurrent writers, migration drift and atomic restore. Inspect asset hashes
before/after. Future golden policy tests must remove keys/model servers and simulate
missing/incompatible assets; hosted release must separately prove workstation-off
missions, quota limits, restore and a 30-day authorized delivery trial.
