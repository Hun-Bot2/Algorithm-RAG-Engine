# Design

Extend replayProblemAttempts for universal scheduling and priority metadata. A shared
review queue provides aging-aware ordering to the existing planner and journal.
Versioned pattern configuration and pure progression replay operate on the merged
learner ledger and read-only active catalog. Journal planning composes existing
planDay reviews/calibration with progression actions, not a second learning engine.

The local runtime attaches the catalog selected by existing catalog_switches.
Node-only journal handlers are injected into createApp; Worker paths remain
compatible and do not import Node SQLite code. The existing recommendation_snapshots
table holds journal revisions, keyed by evidence/config/preferences/catalog hashes;
all revisions remain readable. Timeline mastery transitions are derived from the
ledger at each evidence cutoff; no mutable Markdown store or duplicate event log.

Optional self_minutes is an additive manual-attempt payload field validated by the
existing version1 envelope. Existing events and backups stay valid. No SQL migration
is needed: review/progression are deterministic projections of existing evidence;
snapshots use the existing immutable schema and backup mechanisms.

Unknown catalog sources, missing candidate variation/concepts, semantic fallback,
unconfirmed timing and insufficient retention remain explicit. Tests demonstrate
policy mechanics, never educational effectiveness. Longitudinal retention and
transfer outcomes must accumulate from actual learner use before effectiveness claims.

## Implementation decisions

- Core review policy bumped to deterministic-v3; planner-v2/mastery-policy-v2
  records changed ordering/remediation behavior. Earlier frozen snapshots remain.
- Journal snapshots fingerprint implementation and configs and key only relevant
  attempt/rating evidence, so timer changes do not rotate current practice.
- Local server defaults now resolve from repository root, matching the CLI.
  Existing alternative database files are preserved and never auto-merged.
- History replay skips candidate search/calibration for historical prefixes; these
  are derived states, not a mutable second mastery ledger. Current recommendations
  still use existing calibration and inspectable native difficulty uncertainty.
- Explicit unknown/unavailable states replace fabricated candidate coverage.
  Global corpus/AtCoder labels are reused as-is; no research assets are rewritten.

## Backup restore and historical recommendation evidence

Backup v2 recommendation snapshots are learner-visible historical evidence and
SHALL survive restore in the existing immutable `recommendation_snapshots`
table. Restore SHALL store each imported row under a reserved `archive-<digest>`
kind and wrap its payload as
`{archive_type:'recommendation_snapshot',original_kind,original_payload}` while
preserving its original `local_date`. Archived rows SHALL be available to the
chronological journal only; they SHALL NOT participate in active plan lookup,
exposure/cooldown calculations, or be returned as a current plan. A journal
read SHALL unwrap the original kind/payload for historical display.

Backup export SHALL flatten archived wrappers to their original v2
`{kind,local_date,payload}` representation. This preserves the existing backup
contract across repeated restore/export cycles without nested wrappers or
duplicate archived copies. V1 backups have no mastery archive and remain valid.
No migration is needed. Archived recommendation rows are restored atomically
with the rest of the backup. Weekly report archive behavior is unchanged by this
decision.
