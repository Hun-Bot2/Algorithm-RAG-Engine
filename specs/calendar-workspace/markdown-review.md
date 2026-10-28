# Markdown review selection

## Requirements
- The importer SHALL accept standalone `#복습` in a daily summary-table memo.
- Empty and `-` outcomes MUST NOT create attempts.
- Catalog identity and matching filename/title dates MUST be validated.
- Existing history, recommendation decisions and manual selections MUST be preserved.

## Design
The pure daily parser returns review targets separately from attempt rows. Git ingestion merges dated selections into existing learner preferences before advancing its watermark. Retries deduplicate problem/date pairs. Only committed blobs are read; working files and detailed notes are excluded. No schema migration is required. Imports are additive; deletion uses the existing explicit CLI command.

## Tasks and validation
- [x] Run existing parser/selection tests before editing (9 passed).
- [x] Parse labels independently of blank outcomes.
- [x] Connect committed Git import to existing selections.
- [x] Verify invalid identity, conflicting rows, date mismatch and idempotency (11 focused tests passed).
- [x] Document syntax and additive removal semantics.

## Correction to previous receipt
The user clarified Buddy Strings was currently being solved. Its earlier manual review selection was removed; no actual solve was recorded. This implementation does not modify their Markdown or reselect Buddy Strings automatically.

Full regression: `node --test hosted/test/*.test.mjs` passed 198/198. The first sandboxed run could not bind loopback sockets; the permitted rerun passed. No browser verification was performed for this importer-only change.

## Dedicated column migration
The parser SHALL support optional `복습` after optional `시간`, before `메모`. Only `O` selects a review; blank/`-` remain unspecified. This explicit column overrides memo hashtags. Existing attempt semantics and old table formats remain supported.

All four active dated files in algorithm-solution, README examples, daily template and Markdown guide were migrated. Detailed problem sections were checked byte-for-byte during migration. No review selection was inferred. The unfinished October 23 heading was corrected to its filename date. The standalone convention checker was updated to match; historical schema templates were preserved.

Column migration validation: Engine full Node suite passed 199/199; algorithm-solution suite passed 12/12, including validation of all four current dated files and a dedicated review-column regression. No browser checks or real learner synchronization were performed for this parser/document migration. Changes remain uncommitted. Suggested commit: `feat: support explicit review columns in daily Markdown logs`.

## Binary format (supersedes column values above)

- Result SHALL accept `1` solved, `0` failed, blank unavailable/in progress. Legacy O/X/- remains readable.
- Review SHALL accept `1` target, `0` non-target. Legacy O/blank/- remains readable.
- Numeric success MUST NOT infer independent solving or no hints. It imports as reported_completion.
- Existing independent evidence MUST remain unchanged during an O-to-1 presentation migration; genuine result/time changes still conflict.
- Active daily Markdown and example tables were migrated; detailed sections were preserved byte-for-byte. The user's current Buddy Strings review=1 was preserved, with its outcome still blank.
- Changes: daily parser, Git ingestion compatibility, focused regressions, runbook; sibling repository daily files, README/guide/template, checker and tests.
- Review cancellation remains explicit via CLI, as with the prior additive importer.

Binary validation: focused Engine tests 14/14; full Engine suite 201/201; standalone Markdown checks 13/13. Before changes, standalone checks had one failure because the user had already entered review=1; it now passes. No browser or real learner synchronization was performed. Suggested commit: `feat: use binary result and review flags in daily logs`.
