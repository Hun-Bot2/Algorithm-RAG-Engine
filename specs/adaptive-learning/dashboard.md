# M6 private learning dashboard

## Scope

The static Worker application provides mobile-accessible Today, Progress and
Contests views over the authenticated learner's hosted state. It records only
explicit learner actions. This UI is a client of the Worker API and does not
calculate mission selection, mastery, timer duration, or contest suitability.

## API use

- `GET /api/state` loads `{learner_id, preferences, curriculum, events, mission,
  progress, timers, contests, delivery}`. The client renders absent optional
  collections as empty with an explicit unavailable/empty message, while a
  failed state request is an error with a retry action.
- `POST /api/events` receives exactly the v1 event envelope and typed payload
  described in `design.md`. Manual attempts use source `manual`, outcome
  `independent`, `assisted`, `failed`, or `reported_completion`, and `hints`
  boolean or null. Activity type is optional (`practice`, `review`, `transfer`);
  transfer records also carry a supplied `transfer_from` problem key. The UI
  offers transfer recording only when a matching mission activity supplies its
  source problem.
- Timer actions are explicit `timer_started`, `timer_stopped`, and
  `timer_confirmed` events. A start needs `interval_id` and `problem_key`; stop
  needs the interval ID; confirmation includes learner-entered UTC start/end
  times. Reloaded open and stopped intervals remain recoverable and count as no
  confirmed study time until explicitly confirmed.
- `PUT /api/preferences` saves the weekday minute budget and target keys.
- `GET /api/export` and `POST /api/import` retain the M3 event-only export
  contract. The dashboard's primary backup uses `GET /api/backup/export` and
  `POST /api/backup/import` for the complete learning-data restore document.
- `POST /api/providers/sync` refreshes a Codeforces or AtCoder contest calendar
  without a handle, or imports reported completion events for an explicitly
  entered public handle. Disabled providers show a manual-entry path. Sync
  results report imported counts and incomplete historical coverage.
- `POST /api/participation` records a manual contest participation envelope.
  The learner selects a listed contest or enters an ID, a participation mode,
  outcome and actual participation time. Confirmed minutes remain null unless
  the learner enters a value.
- `POST /api/contests/manual` submits user-entered contest information when a
  provider is unavailable. Provider-backed contest data may be absent or stale.

The UI treats API errors as errors and retains failed event writes for retry.
Pending writes stay in browser local storage until the API acknowledges them.
Each pending write keeps its event ID across retries and reloads. A successful
write removes it and reloads current state. No data write is inferred from page
view, submission timestamps, or navigation.

## Views and evidence language

- Today displays mission activities and their supplied reasons, plus explicit
  attempt and timer controls. Empty mission means no activity is currently
  available; it does not create sample activity.
- Progress presents only supplied evidence states and recorded totals. Concept
  evidence labels are `unknown`, `learning`, `demonstrated`, and
  `transfer-evidenced`; missing outcomes are described as no recorded outcome.
  Superseded attempts are excluded from active totals but remain visible in
  recent history as corrected records. Supplied problem `due_date` values are
  shown as review dates. Confirmed minutes are distinct from open,
  stopped-unconfirmed, or unavailable timer intervals.
- Contests displays supplied contest information and freshness/status, with a
  manual entry path. Provider freshness is read from `contest_sources` when the
  Worker exposes it; upstream `suitable: false` is shown as unreviewed
  suitability, while a manual choice is shown as the learner's judgment. The
  view includes provider sync controls, explicit participation recording, and
  append-only participation history.
- Preferences expose the weekday budget and target problem keys.
- Export and import require an explicit user action. Import requires a selected
  JSON file and a clear submit action. The full restore copy says restore is
  allowed only into empty history, does not delete or overwrite existing
  records, and pauses delivery until reconciliation. The delivery panel labels
  scheduled, posted, unknown and missed Slack states; it does not claim that a
  phone alert was received.

All learner-facing copy is Korean. Controls have visible labels, keyboard focus,
semantic headings, and responsive single-column mobile layout. External problem
links are allowed only for canonical HTTPS platform hosts and use a new tab with
`noopener noreferrer`.

## Validation

Use Node's built-in test runner for pure view/event/participation helpers and UI
contract checks. Check JavaScript syntax and static HTML references. Browser
interaction and Worker/Access behavior remain separate validation gates; local
helper tests do not prove hosted API authorization or visual rendering.


## Final browser and privacy checks

Pending writes are partitioned by verified learner ID in browser storage;
authentication failures clear the displayed pending state. Confirmed timer inputs
preserve seconds/milliseconds (step .001), preventing default values from being
rounded before their recorded start. Goals use problem names rather than raw
keys; seven weekday controls configure the rhythm. Contest/virtual/upsolve cards
open contest records instead of an invalid problem timer. Actual Chrome 154 at
390x844 verified explicit interval confirmation, offline reload/retry with one
stable ID, cross-owner queue separation, participation with unavailable minutes
and the full learning backup route. Screenshots and raw receipts remain private.
