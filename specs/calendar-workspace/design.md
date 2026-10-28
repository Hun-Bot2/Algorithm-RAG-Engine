# Audit and design

## Verified repository audit

Frontend: hosted/public/index.html, app.mjs and styles.css; no framework/router/chart library. Four existing tab handlers; narrow 780px content, stacked journal/plan/filter/review cards. Existing DOM constructors, forms and usage table are reusable. local.mjs serves an explicit static allowlist. Worker uses ASSETS. Existing dirty explanation/daily-ingestion work predates this task and must remain intact.

Data: /api/state exposes curriculum, manual/core events, preferences, mission, contest cache/source status, timers and participation. /api/journal combines catalog-backed ledger attempts, archived/current recommendation snapshots and evidence-derived concept changes. Entries include id, occurred_at, recorded_at, title, platform, difficulty, pattern, outcome, hints, self_minutes and superseded status. No separate start/completion timestamp or editable notes field exists in this contract. Reported completion is not independent evidence. Confirmed timers may overlap self-reported minutes and must not be summed.

Editing: /api/journal/attempt accepts occurred_at and supersedes; preserve original problem/activity/transfer fields. Only event-backed manual records can safely be corrected here; Git observations are edited in their source repository. Existing pending journal retry queue remains the write path.

Providers: CF/AtCoder schedules already use cached provider service with failure status; local runtime providers are disabled. LeetCode has no admitted automatic schedule adapter. No live synchronization is implied by cached fixtures or tests. Explanation generation remains in existing CLI/service, independent provider selectors; UI displays persisted outputs only.

## Reference research (2026-10-20)

- [Notion Calendar](https://calendar.notion.so/) is an app shell with no readable unauthenticated document. Its [official interface guide](https://www.notion.com/en-gb/help/guides/getting-started-with-notion-calendar) and calendar view examples inform compact grid, Today/month controls and J/K navigation. Adapt to month-first recorded learning; avoid timed blocks for attempts with no duration.
- [Linear](https://linear.app/) publicly renders sidebar/workspace, dense issue rows and contextual properties. Adopt persistent restrained navigation, compact typography and contextual details; avoid its organizational complexity and marketing dark styling.
- [GitHub](https://github.com/) public landing page is marketing; the actual contribution grid was inspected on the public [torvalds profile](https://github.com/torvalds), including month/day labels, year navigation and sequential green intensity bins. [Contribution reference](https://docs.github.com/en/account-and-profile/reference/profile-contributions-reference) documents contribution provenance/date semantics. Adapt consistent count bins and date drill-down; do not equate contribution/activity counts with mastery or infer unrecorded days are failures.
- [Observable Plot interaction examples](https://observablehq.com/plot/features/interactions) demonstrate details-on-demand and filter-driven updates. Adopt source-backed interactive marks and exact values; no new library is needed for accessible heatmap/buttons and bars. Avoid decorative scatterplots and unsupported mastery scores.
- [Muzli calendar gallery](https://muz.li/inspiration/calendar/) presents multiple compact calendar layouts. Adopt grouped toolbar/grid hierarchy; avoid oversized cards and decorative effects.

Research boundary: web document/interface examples were inspected; authenticated Notion/Linear applications were not accessed. Rendered reference inspection receipts, when available, are recorded in validation.md.

## Architecture and information flow

Existing app remains owner of requests, pending writes, learner state and legacy tools. New workspace-data.mjs contains pure mapping/calendar/filter/aggregation helpers; workspace.mjs renders navigation content and common dialog. app.mjs supplies state and write/form adapters. No policy imports or learner engine duplication.

Calendar is default; history, analytics and contests are primary destinations. Goals/timers/backups remain an auxiliary Tools view. Calendar uses full desktop width with a compact agenda rail. Month cells show solid outcome-labeled attempts, dashed plans and violet contests; weekends subtly shaded. Mobile exposes a compact date picker and selected-day agenda. Details replace a single native dialog rather than stacking.

Tokens retain warm paper, white surfaces, green activity and restrained violet contests, 4/8px spacing, subtle borders, visible focus. Motion is optional. Native dialog traps focus and makes background inert; origin focus and body scroll are restored on close.

## Risks and tradeoffs

No timezone preferences exist; UI discloses policy timezone rather than changing backend day boundaries. No LeetCode scraper will be added. Catalog-wide arbitrary problem search is absent in API; add form uses available catalog/problem keys and recommends existing paths for unknown keys. No separate attempt-start, completion timestamp or notes storage is invented. Recommendation conversion is a descriptive temporal match, not causal attribution. Scope of loaded history is disclosed.

## Refinement 2 — calendar selection and contest list (2026-10-20)

Research rechecked the four mandatory live public pages and Notion's official getting-started guide. Linear informs compact list rows, quiet status badges and one contextual dialog; Notion informs persistent date selection distinct from Today and overflow drill-down; GitHub informs a fixed-bin monthly count heatmap with visible date labels; Observable Plot informs exact counts and drill-down from each mark. Authenticated Notion remains unavailable. These are adapted interaction principles, not copied styling.

Requirements: date clicks SHALL select the day without opening a dialog; +N SHALL open its daily dialog. Both month/week cells SHALL show at most three concise rows. Solved records, other attempts, plans and contests SHALL be independently filterable without double counting. Today's filled date marker MUST remain distinct from the selected-day outline. Selected-day sections SHALL be conditional, with actual attempts and compact statistics first. A monthly heatmap and compact distributions SHALL use active observed records and open their source rows.

Contest view SHALL be a chronological list with platform/upcoming filters and exact KST timestamps; the main calendar SHALL retain the same fetched records. No new collector/API/schema or recommendation-policy change is needed. Suitability MUST NOT use title keywords or equate platform activity with contest readiness: direct same-contest completed participation supports a review recommendation; partial/abandoned participation supports a stretch/retry label; explicit suitable=true is a user preference only. Otherwise evidence is insufficient. Active participation corrections MUST be respected. Stale schedules MUST disclose verification needs. No unseen future contest demand is inferred.

Affected files: hosted/public/workspace{,-data}.mjs, styles.css, hosted/test/workspace-data.test.mjs, hosted/workspace-browser-smoke.mjs and governing documentation. Preserve app.mjs write/explanation adapters and existing unrelated dirty work.

Sequence: pure classification/date/suitability helpers and regression tests; calendar/day panel; contest list; compact monthly visualizations; responsive/modal browser checks; full regression and documentation. Validation includes KST October 20/21 boundaries, filter partitions, actual-first overflow, list/calendar consistency, selected/today state, Escape/focus restoration and unchanged explanation/append-only edit/offline retry flows. Risks: stale provider data, missing contest demands, unknown outcomes, duplicate or corrected history, and mobile density. All missingness is visible rather than fabricated.

## Explicit review target implementation (2026-10-23)

Use the canonical learner SQLite preferences JSON for manual dated review selections, independent of attempts, planner decisions and computed due dates. Validation allows only unique problem/date pairs and exact native keys; the CLI verifies catalog membership. `journal.review_selections` enriches keys from the active catalog; the frontend maps these to manual review plan events and retains their precedence over same-day automatic practice cards. Existing automatic review projections remain omitted. Add/remove updates the existing preferences revision and is idempotent; existing backup/export preserves selections through its current preference contract. No new table or side database is needed. The source Markdown remains untouched and uncommitted, so its blank result and inconsistent heading cannot create solve evidence.

Affected modules: core/preferences, cli, mastery/journal, public/workspace-data; tests for invalid/duplicate dates, round-trip state, CLI idempotency, manual/automatic distinction and zero added attempts. Verify with a disposable fixture before applying the explicitly authorized Buddy Strings selection to local-owner on 2026-10-23. Running local readers need the updated code to accept the additive JSON key; retain the existing env-file launch configuration when reloading the local application.

## Plan-first day panel and carry-over (2026-10-28)

`workspaceEvents(journal,state,{today})` marks each plan with `progress` (`pending`/`tried`/`done`) and, for pending system plans
within `CARRY_DAYS=7`, sets `carried_to` on the original and adds one synthetic `carry:<today>:<key>` event per problem with
`carried_from` (earliest origin). An existing plan for the same key today gets `carried_from` instead of a copy. `pendingPlans()` is
the single filter for unfinished work, used by the calendar cells, the day panel and the day dialog. The planner still excludes these
problems through its exposure cooldown, so today's fresh plan and the carried list do not overlap.

Tradeoffs: any attempt (including a failure) ends carry-over, because failures already get next-day priority from the review policy.
The 7-day bound keeps the list finite; older unsolved plans stay on their own date. Moving rather than duplicating keeps past cells
quiet; their day content still lists what moved ("오늘로 이월"). Tests pass `today` explicitly so fixtures stay deterministic.
The right-hand rail became a full-width panel above the calendar: blocks need horizontal room, and today's work should be visible first.
