# Sequential tasks

- [x] Audit frontend, journal, append-only corrections, providers and explanation contracts.
- [x] Run baseline Node suite (169 passed with loopback access).
- [x] Research mandatory references and document design/requirements/risks.
- [x] Add pure date/event mapping and analytics helpers with regression cases.
- [x] Implement adaptive workspace, calendar month/week/agenda, history, dialogs and corrections.
- [x] Integrate cached contests and explanations without new acquisition/generation.
- [x] Implement source-backed analytical interactions.
- [x] Run Node suite, asset build and static/module checks.
- [x] Exercise browser responsive/modal/edit/explanation flows with disposable fixtures; inspect screenshots.
- [x] Document exact verification and remaining unavailable sources; suggest commit.

## Interaction refinement

- [x] Re-audit implementation, baseline tests and four mandatory references.
- [x] Add date/category/filter/snapshot and suitability regressions.
- [x] Refine day selection, conditional agenda, statistics and compact monthly visualizations.
- [x] Replace contest destination with a list and preserve actual-date calendar placement.
- [x] Preserve supplied contest discovery/assessment contract and expose source provenance.
- [x] Complete final browser, build/regression and screenshot inspection receipts (192 Node tests, 3 Chrome scenarios; 1440/834/390px).

## Plan-first day panel and carry-over (2026-10-28)

- [x] Project plan progress and 7-day carry-over in workspace-data (presentation only), with a deterministic unit test.
- [x] Replace the agenda rail with a full-width day panel: unfinished plans as blocks, carried plans as a secondary group, attempts as chips.
- [x] Calendar cells list unfinished plans and schedules first and summarize multiple attempts in one chip; remove "하루 자세히".
- [x] Node tests (241 pass) and visual check at 1440px and 375px against the running local app (read-only).
- [x] Repair hosted/workspace-browser-smoke.mjs for policy v3 (no hints field) and make its contest/midnight fixtures relative to today (3 scenarios pass).
- [x] Explain carried plans from their stored snapshots (`decorate` with `snapshotPlan`); generate only the displayed origin per problem.

## Research library view (2026-10-28)

- [x] Curated catalog `research/library.v1.json` (7 folders, 28 files) and local-only id-based file server with sandboxed HTML.
- [x] "연구 자료" view: folder list, thumbnail cards, image and sandboxed-page previews; tests in hosted/test/local-library.test.mjs.
- [x] Same-platform toggle for the comparison viewer maps.
