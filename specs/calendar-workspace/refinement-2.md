# Calendar interactions and contest-list refinement — 2026-10-20

## Result and changed files

- `hosted/public/workspace.mjs`: persistent date selection, conditional selected-day agenda/statistics/actions, four calendar filters, three-item overflow in both month/week views, contextual dialogs, compact monthly analytical panels, chronological contest list, platform/scope filtering and source-aware contest details. The list and calendar consume the same loaded events. Server-provided contest recommendations remain authoritative and their evidence is inspectable.
- `hosted/public/workspace-data.mjs`: explicit civil-date and timezone handling, disjoint event categories, active day counts, chronological contest filtering, conservative fallback suitability and grouping of recommendation snapshot revisions without discarding their evidence.
- `hosted/public/styles.css`: compact cells, separate Today/selection states, selected-day hierarchy, mobile category counts, monthly chart grid and dense responsive contest rows.
- `hosted/public/app.mjs`: supplied contest synchronization adapter sends same-origin JSON. Existing request/retry, append-only record writes, recommendation ordering and Gemini/NVIDIA generation/cache behavior are preserved.
- `hosted/test/workspace-data.test.mjs`, `hosted/workspace-browser-smoke.mjs`: focused regression coverage and disposable Chrome/CDP flows. Existing unrelated dirty work and the separately added contest-discovery modules were preserved.
- Governing requirements/design/tasks, README and AGENTS guidance updated.

## Design-reference rationale

The mandatory public pages were reopened and rendered in Chrome. Public reference screenshots and observed page summaries are in `/tmp/workspace-refinement-references/`.

| Reference | Observed pattern and adaptation | Deliberate boundary |
| --- | --- | --- |
| [Linear](https://linear.app/) | Public issue/detail demo: compact navigation, restrained row separation, contextual properties. Adapted to dense contest rows, quiet status badges, focused day statistics and a single replaceable modal. | Organizational complexity and dark marketing treatment are not copied. |
| [Notion Calendar](https://calendar.notion.so/) and its [official guide](https://www.notion.com/en-gb/help/guides/getting-started-with-notion-calendar) | Guide screenshots and documented Today/J/K navigation informed stable date selection and details on demand. Selecting a date updates the agenda; overflow opens the day dialog. | Calendar app redirects to authentication; no authenticated-interface inspection is claimed. Historical attempts cannot be dragged to a plan date. |
| [GitHub Contributions](https://github.com/torvalds) | Actual public month/day grid and sequential green bins informed the monthly heatmap, consistent count bins, numeric dates and drill-down. | Counts describe recorded attempts, not mastery, a guaranteed daily streak or failure on empty dates. |
| [Observable Plot](https://observablehq.com/plot/features/interactions) | Rendered interactive examples demonstrate exact values and source details on demand. Monthly trend/distribution marks open the corresponding attempts/problems. | No decorative scatterplot, hidden normalization or additional chart dependency. |

## Interaction and data semantics

- Initial selected date and Today derive from the same Asia/Seoul instant. A filled green circle identifies Today; an outlined day cell and pressed date button identify selection. Browser testing additionally changes the browser timezone to America/Los_Angeles.
- UTC instants convert once. Civil YYYY-MM-DD values remain civil dates. Null, invalid and timezone-less timestamps do not become 1970 dates or browser-local evidence. Regression cases cover the exact October 20 14:59:59.999Z / 15:00:00Z boundary.
- Solved records and other attempts are disjoint calendar categories. Solved includes recorded independent/assisted/reported completions, retaining their distinct symbols and detail labels; imported completion is not promoted to independent evidence. The "시도 · 미해결" filter covers attempts without recorded completion. Plans and contests never count as attempts.
- Day statistics count attempts, completed records, failed attempts and explicit unknown outcomes. Counts follow the active calendar filters. Corrections remain in problem history but are excluded from active counts.
- Both calendar modes show at most three concise entries. Full titles remain accessible in tooltips, labels and details; long titles use a concise platform/status or contest-time label rather than clipped text. +N opens all filtered day events.
- Repeated recommendation snapshots group by date/problem for display. Distinct revisions and original reasons remain in recommendation details; duplicate representations of the current snapshot are one revision. Current decorated explanations take precedence. Nothing is rewritten in storage or re-ranked.
- Selected-day sections appear only when populated, with actual records first. The panel limits its preview and provides full-day drill-down. Monthly charts appear when actual records exist and show counts, weekly solving/completion records, independence/assistance and topic frequencies. Multi-topic attempts may contribute to more than one topic; no mastery is inferred.
- Contest view is a date-grouped chronological list with upcoming/ongoing vs all-loaded and platform filters. "캘린더에서 날짜 보기" selects the actual KST date in the primary calendar. Weekend shading never changes timestamps.
- Existing native dialogs retain Escape, focus containment, responsive sizing, return focus and scroll restoration; form dialogs initially focus the first editable input. Dialogs replace rather than nest.

## Contest integration boundary

During this refinement, the checkout also acquired a separate `contest-discovery` adapter and optional server recommendation payload. The UI preserves these changes and prioritizes that supplied assessment; it does not recalculate the model or change problem recommendation rankings. See `specs/contest-discovery/` for that adapter's requirements and source decisions.

Current local capability exposes an explicit public-schedule sync action. Official platform schedules and LeetCode's community mirror retain their source URLs, source kind, retrieval time, source update time and stale states. The UI labels the community mirror rather than implying an official LeetCode schedule API. Missing or failed sources stay explicit. A disabled capability disables the sync action.

When the server supplies no assessment, the fallback uses corrected same-contest participation or an explicitly marked manual preference. Completion supports a review recommendation; partial/abandoned participation supports a retry/stretch label; direct manual selection is labeled as user choice. Platform activity volume and contest-title keywords alone never establish ability. All other cases remain insufficient evidence.

The server's new optional historical opening-problem comparison is a provisional practice-goal heuristic, not a validated full-contest difficulty, rating, mastery or learning-effect estimate. Missing calibration, historical difficulty or eligibility stays explicit. The frontend presents the supplied reasons and evidence rather than changing these decisions.

## Validation

- Before this refinement: **177 Node tests passed** with loopback socket access.
- Final full Node regression: **192 passed**, including the preserved discovery integration tests and focused UI data regressions. Log: `/tmp/workspace-refinement-final-tests.log`.
- Static asset build and pinned asset verification passed: **7 verified, 1 explicitly missing**, preserving that missingness.
- Worker dry-run passed with five browser assets; no deployment or hosted-runtime proof is claimed.
- Module syntax and `git diff --check`: passed. No repository lint script is configured; this is not a fabricated lint result.
- Actual Chrome/CDP suite: **3 scenarios passed**. Receipts/screenshots: `/tmp/workspace-refinement-verified-v2/` (`receipt.json`, `selected-day-1440.png`, `monthly-insights-1440.png`, `populated-contests-390.png`). The successful synthetic public-sync path made three injected source requests and demonstrated October 21 weekday / October 25 weekend placement. Synthetic provider/model transports and disposable learner databases only; production records are never written.

Browser coverage includes desktop/tablet/mobile (1440, 834, 390px), all four destinations, initial KST selection under a non-Korean browser timezone, Today vs selection, modal Tab/Escape/focus return, date selection without a modal, daily overflow, filters, monthly drill-down, native record add/correction, stable-ID offline retry, dual-provider success/failure/fallback/cache cases, source freshness, public-sync routing, real-date contest placement, weekday/weekend distinction and no horizontal overflow. Screenshot review found duplicate snapshot clutter and 12-hour time wrapping; grouping and 24-hour clocks corrected both.

## Remaining limits

No additional paid model invocation, production database mutation, deployment, human usability study, screen-reader audit or educational validation was performed by the UI refinement. This report's contest-sync browser proof uses injected source responses, not live-source freshness. Any separately recorded live discovery receipts belong to their own report. Source changes/outages can still make schedules unavailable. Only fields actually supplied by the current journal/API are editable or visualized; no missing notes, start/completion times, tokens or skill labels are fabricated. No 3D features or new runtime dependencies are introduced.

Suggested commit: `feat(ui): refine calendar selection and contest agenda`

Stage only intended hunks: this checkout also contains pre-existing explanation, ingestion and separately added discovery work.

## Follow-up: explicitly selected reviews (2026-10-23)

The learner will mark review targets with a separate label. No such label field is currently present in the inspected journal contract. Calendar projection now omits automatic review slots, historical review-plan overlays and the upcoming review queue. Actual review attempts, ordinary recommendations, persisted snapshots and backend review policy remain intact. Label editing/ingestion is not implemented until its format is defined. A focused regression exercises current, historical, upcoming and core-only paths.

### Explicit review target implementation and receipt

Manual review targets now persist in the canonical learner preferences as unique `{problem_key, date}` pairs. `review add|remove` validates the active catalog and is idempotent; `journal.review_selections` enriches the display from that catalog. Existing full backup/restore preserves the optional JSON field, with no new SQL table or migration. Automatic review plans remain hidden.

User explicitly confirmed Buddy Strings after inspection of `algorithm-solution/leetcode/26-10-23.md`. The file is untracked, its heading says October 20 and its result is blank; it was not changed, committed or imported as solve evidence. Registered `leetcode:buddy-strings` for October 23 for `local-owner`, after a private pre-change learner backup. The local app was reloaded with its existing env-file command to read the additive preference field. SQLite verification: **events 0 → 0, git observations 3 → 3**; one explicit review target saved.

Validation: **196 Node tests passed** (`/tmp/review-target-all.log`), including CLI idempotency, invalid dates, original preferences, catalog enrichment, zero attempts and backup restore. The initial full run exposed the earlier automatic-review expectation plus a contest HTTP test with expired fixed dates; the former now uses an explicit selection, and the latter keeps its own local source fixtures in the future without changing production behavior. Live Chrome read-only verification at 1440px and 390px confirms today's selected date, Buddy Strings in the calendar and panel as "복습 예정", and no horizontal overflow. Private local screenshot: `/tmp/buddy-strings-review-2026-10-23.png`. No production browser write or paid inference was used for that verification.
