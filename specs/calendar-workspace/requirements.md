# Calendar workspace requirements

- The landing workspace SHALL be a monthly calendar with a week option and mobile agenda.
- Actual attempts MUST use occurred_at in the supported policy timezone Asia/Seoul. No configurable timezone currently exists.
- Recommendations MUST use local_date/date, remain a separate layer, and never count as attempts. Superseded records MUST remain inspectable and be excluded from activity totals.
- History SHALL search/filter available problem, platform, outcome, topic and difficulty fields. Details SHALL show all previous attempts.
- Corrections MUST append via the existing supersedes contract; imported Git observations SHALL remain read-only in the UI.
- Analytics SHALL count recorded active attempts, disclose missingness, separate platform difficulty scales and provide source-row drill-down.
- Explanations SHALL retain both providers, cached outputs, errors, deterministic fallback and original evidence without changing ranking. Browsing MUST NOT trigger paid calls.
- Contest views SHALL preserve timestamps, source/fetched_at/staleness and show unavailable sources explicitly. Availability SHALL follow the current adapter/capability contract; community mirrors and manual schedules MUST be labeled with their actual provenance.
- Dialogs SHALL support native focus containment, Escape, focus restoration, single-dialog replacement and mobile scrolling.
- No 3D, new production dependency, schema migration, statement collection or external deployment is in scope.

## Interaction refinement

- Date clicks SHALL select the day and update the agenda; +N SHALL open the full day dialog.
- Both calendar modes MUST show at most three concise entries and preserve all remaining events in daily details.
- Today MUST remain visually distinct from selection, with initial selection based on Asia/Seoul.
- Solved, other attempts, planned and contest categories SHALL be independently filterable.
- Contests SHALL have a chronological list destination while retaining their actual dates on the primary calendar. Supplied server suitability evidence MUST be preserved; title-only ability inference is forbidden.
- Repeated same-day/problem recommendation snapshots SHALL group for presentation while retaining distinct revisions and original evidence in details.

## Explicit review selection (2026-10-23)

- The calendar MUST NOT turn all solved problems or automatic review-queue dates into scheduled review events.
- Until the learner's separate review-label format is defined and integrated, automatic review slots, historical review-plan overlays and upcoming queue overlays SHALL be omitted from the calendar projection.
- Actual attempts with activity_type=review MUST remain in history and active activity counts. Ordinary practice recommendations and backend review evidence MUST remain preserved.

## Manual review selection storage (2026-10-23)

- Explicitly selected reviews SHALL be separate from attempts and the computed review queue.
- An optional `preferences.review_targets` array SHALL store exact `{problem_key, date}` selections. Missing means no selection; dates are civil Asia/Seoul dates.
- CLI add/remove SHALL validate the active catalog identity, preserve unrelated preferences, and be idempotent. Selection MUST NOT create a completion or attempt.
- Calendar SHALL project only these manual review selections; automatic review schedules remain hidden. Catalog-backed journal responses SHALL provide title/URL/difficulty for the selected keys.
- This is an additive preferences-JSON extension, with no SQL schema change or migration. Existing stored preferences remain readable. Readers must run the updated normalizer before new fields are written; standard learner backups already include preferences.

## Plan-first day panel and carry-over (2026-10-28)

- The selected-day panel SHALL show unfinished plans first, as full-width blocks above the calendar. Attempts SHALL appear as compact chips. The separate "하루 자세히" action is removed; overflow ("+N개 더 보기") and the multi-attempt summary chip open the same day content in a dialog.
- A plan is settled WHEN an active attempt for the same problem exists on or after its date (`done` if any completed, else `tried`). Settled plans SHALL NOT appear as unfinished work; their attempts represent them.
- WHEN a system plan (not a manual review, review slot or archived snapshot) is unsettled and dated within the last 7 days, THE SYSTEM SHALL show it on today as carried over, with its earliest origin date, and hide it from its original date's cell.
- WHEN explanations are generated, THE SYSTEM SHALL also explain carried plans from their own stored snapshot, dispatching at most one
  generation per problem (the entry the calendar shows); other revisions only read the cache.
- Carry-over is presentation only: snapshots, attempts and planner inputs MUST NOT change, and carried copies MUST NOT count as extra plans in recommendation-follow-up statistics.
