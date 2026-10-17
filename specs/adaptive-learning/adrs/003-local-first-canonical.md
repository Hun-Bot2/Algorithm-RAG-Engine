# ADR 003: local SQLite canonical store, hosting optional

Status: accepted 2026-10-17. The user directed that the mastery-system design
"preserve the existing local-first JS/SQLite architecture and reuse the
adaptive-learning core". This ADR supersedes only the hosting and
canonical-store parts of ADR 001.

## Relationship to ADR 001

| ADR 001 element | Status after ADR 003 |
|---|---|
| Deterministic no-AI core; models never called by the runtime | Unchanged |
| Optional offline AI | Unchanged; extended to AI-assisted evaluation batches whose outputs are versioned data |
| Python for offline profiling, permitted ingestion and evaluation | Unchanged; also builds immutable catalog versions |
| Worker + D1 + Access as "the eventual private service" | Superseded: an optional deployment of the same core |
| Local SQLite "for development/recovery" | Superseded: local SQLite is canonical |
| "Local-only cannot satisfy unattended phone delivery" | Accepted as a consequence: unattended daily push is optional; the weekly report in the learner's Git repository is the default channel |

## Context

- The JS core already runs locally on `node:sqlite` through `hosted/local.mjs`,
  with the same migrations.
- M8 hosting is unreleased, and its account, quota and 30-day trial gates remain
  open.
- The mastery system needs Git integration, a large read-only catalog (ATTACH)
  and low cost.

## Decision

- The local learner database is canonical.
- The JS core is the only learner-state engine.
- Python produces read-only, versioned catalog and evaluation artifacts.
- Hosted deployment and Slack delivery are optional adapters, and their existing
  tests must keep passing.
- Single writer: only one deployment is canonical at a time. Switching uses
  backup export and restore-into-empty-owner. No two-way sync.

## Consequences

- Missions are produced only while the learner's machine runs. Backups are the
  learner's responsibility.
- Catalog-wide features are local-only until a hosted catalog packaging decision
  exists.
- `src/learning/store.py` is frozen for recovery and legacy use.

## Rejected

- Porting the policy to Python.
- A third store.
- Two-way local/hosted sync.
- A mandatory hosted service for one learner.
