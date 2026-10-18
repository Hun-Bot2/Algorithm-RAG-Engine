# Adaptive Mastery Journal requirements

- Reuse the existing Node runtime, learner SQLite, append-only evidence, catalog
  attachment, calibration, reviews and snapshots. No parallel database or scheduler.
- Every attempt SHALL receive a problem review schedule. Independent no-hint
  success uses normal spaced review; failed/assisted outcomes use priority review
  and concept remediation. Imported completion SHALL never become mastery evidence.
- Priority SHALL account for overdue aging and failure severity without starving
  old normal reviews. Recovery MUST preserve historical failures.
- Versioned pattern rules SHALL default to three distinct Easy variations then
  one or two Medium applications then unseen independent cross-platform transfer.
  Counts alone SHALL NOT imply mastery. Retention evidence is required; Easy,
  application, transfer and advanced evidence remain separate.
- Journal recommendations SHALL compose existing planner/calibration with pattern
  progression, source concepts, prerequisites, aliases and exposure history.
  Semantic-only transfer candidates MUST be labeled unverified and not credit mastery.
- The local journal SHALL expose date-grouped attempts, recommendation snapshots,
  mastery transitions, review queues, upcoming tasks and weekly diagnostics; provide
  date/platform/pattern/difficulty/status/priority filters and direct O/X recording.
- Historical recommendations SHALL be immutable, recorded in existing SQLite
  snapshots with catalog/policy/high-water provenance; same-day new evidence SHALL
  produce a new snapshot rather than overwrite the earlier one.
- CLI and local Git ingestion SHALL continue without network. Optional self-reported
  time MUST remain distinct from explicitly confirmed timer intervals.
- Test synthetic databases only. Preserve real learner data, catalog/research files
  and unrelated work. Core SHALL require no API/LLM/cloud; no deployment or push.

## Markdown-first attempt entry

The registered Git ingester SHALL accept explicit, versioned flat front matter from
committed Markdown/MDX. No duplicate TXT/commit outcome entry is required. The body
SHALL NOT be used to infer success. Schema 2 SHALL accept only O/X, omit hints,
and default omitted activity to practice. O declares independent success; X covers
failed or non-independent solving. Schema 1 retains its explicit hint contract;
unknown outcome remains unverified. Problem keys must agree with recognized paths.
An attempt token scopes a real attempt per problem/repository: note-only changes
MUST NOT add attempts, while a new token permits review/new attempts. Changing a
counted token's metadata SHALL be flagged instead of silently overwriting history.
Existing explicit commit outcome trailers remain supported and take precedence.
Original files, real learner history and remote Git configuration remain untouched.
