# Daily duration units

## Requirements and design
Daily summary times SHALL accept nonnegative integer `s` (seconds) or `m` (minutes), up to the existing 24-hour maximum. Blank/`-` SHALL remain unavailable; zero SHALL remain recorded zero. Seconds MUST NOT be rounded into whole minutes. Existing minute records, outcomes, recommendation decisions and timer evidence MUST remain unchanged.

Use optional `self_seconds` on JSON attempt payloads and unified views; existing integer `self_minutes` remains unchanged. Exactly one non-null duration unit is permitted. This is a backward-compatible optional JSON extension, not a SQL schema migration; old readers may omit seconds, so restart the local application after upgrading. Preserve append-only records and signature compatibility; changing units to an equivalent duration MUST NOT create duplicate/conflicting attempts.

## Tasks and validation plan
- [x] Inspect parser, event/ledger contracts, journal and UI; run baseline tests (Engine focused 14, standalone 13).
- [x] Extend parser and evidence contracts without rounding.
- [x] Expose and display exact seconds; preserve them in record editing.
- [x] Update standalone checker, templates and documentation; preserve existing Markdown durations.
- [x] Test unit boundaries, missing/zero, invalid values, Git roundtrip and unit-change deduplication.
- [x] Run full regression suites. Browser validation is not assumed.

Risk: second-based records have no integer-minute measurement; never fill unavailable minutes with zero or combine them with confirmed timers as disjoint elapsed evidence.

## Validation report
- Engine full Node suite: 204/204 passed. Standalone solution repository: 14/14 passed.
- Parser cases: 10m, 40s, both unit zeroes, missing/-, 24-hour boundaries, invalid decimal/negative/unitless/combined-unit values.
- Git evidence preserves exact seconds; equivalent unit changes deduplicate; genuine time changes flag conflicts.
- Journal tests cover second-based manual validation, current/historical views, correction, zero, full backup/restore and exact weekly self-reported totals.
- Actual Chrome targeted synthetic checks passed: 17m save, 40s correction, preserved prior event, details text, reopening editor with 40s, desktop and 390px mobile; screenshots visually inspected. Artifacts: /tmp/duration-units-focused-browser/.
- The broader pre-existing browser scenario reached its contest section then failed on expired fixed October contest fixtures; no full-browser-suite pass is claimed. Focused duration checks ran separately against disposable data and stubbed providers, without learner changes or real model calls.
- Initial full Node run caught two report text compatibility expectations; old minute-subtotal wording was preserved alongside the exact seconds total, then all tests passed.
- Existing daily Markdown durations and detailed solutions were not rewritten. Standalone checker/template/README/guide updated; legacy per-problem schema and commit-trailer minute semantics unchanged.
- No database migration, deployment or commit. Suggested commit: `feat: preserve second and minute durations in daily logs`.

Build passed (`npm run build`). Restarted the session-owned loopback app on 8787 after guarding the exact original command; retained the same environment-file configuration. No live learner writes or provider generations were requested for verification.
