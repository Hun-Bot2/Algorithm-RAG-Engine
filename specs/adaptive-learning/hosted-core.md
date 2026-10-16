# Hosted core implementation contract (M5/M8 local step)

## Research and boundary

The Python M3 event schema and replay rules are the compatibility baseline. Cloudflare Access requires a verified application JWT even behind Access; its assertion header alone is untrusted. D1 `batch` executes prepared statements transactionally. Hosted release, rights review, quotas, external providers, and delivery remain separate gates.

## Requirements

- Every Worker route, including static assets, SHALL verify RS256 Access signature, configured HTTPS issuer and audience, subject, allowed email, expiration and not-before. The verified subject SHALL derive the storage partition. API bodies MUST NOT select another partition.
- Hosted event validation SHALL preserve the exact M3 v1 envelope, source/payload constraints, correction chain and timer replay. Optional manual attempt activity fields are `activity_type` (`practice`, `review`, `transfer`) and `transfer_from` (qualified key, only for transfer). Historical v1 payloads remain accepted.
- A mutation SHALL use an optimistic learner revision guard in the same D1 batch as all inserts. Conflicts or invalid replay SHALL leave the entire batch unchanged. Duplicate identical event IDs SHALL be idempotent.
- Only approved, explicitly free curriculum entries SHALL be mission candidates. Reviewed prerequisite edges SHALL plan backward from target keys. Unknown/paid access and proposed relationships SHALL never supply eligibility.
- Mission ranking SHALL be deterministic for a fixed curriculum, preferences, event cutoff and date. Recorded and occurred times both bound the cutoff before corrections are applied. A frozen daily exposure SHALL be immutable and returned by GET without regeneration.
- Imported platform completions and unconfirmed timers SHALL not establish mastery. Reviewed independent attempts SHALL advance evidence; failed/assisted review SHALL schedule next-day remediation.

## Design and tasks

`hosted/core/events.mjs` ports validation/replay. `store.mjs` owns D1 SQL, revision-guard batches, preferences, immutable mission and manual contest records. `policy.mjs` computes concept evidence, due review, target backplanning, component scores and a frozen mission. `auth.mjs` validates Access JWT via WebCrypto and issuer JWKS, with a bounded key cache. `app.mjs` owns authenticated routes; `worker.mjs` binds the approved generated curriculum and static assets. `hosted/migrations/001_core.sql` is additive to a new hosted database, with append-only event triggers and version receipt. Node tests use SQLite-backed D1 API emulation against the actual SQL.

The initial policy chooses a bounded practice/review set within the weekday minute preference. Cold starts admit beginner difficulty only, using an annotation band or conservative native difficulty; candidate problem approval and explicit free access are separate mandatory gates. An explicitly selected target with no eligible reviewed path does not silently fall back to unrelated practice. Review intervals are 1, 3, 7, 14, 30 and 60 calendar days in Asia/Seoul. Failed or assisted review returns the problem to learning and schedules next-day remediation, preserving the earlier raw evidence. Scores expose target gap, weakness, suitability and variety. Explanations use only reviewed relation/concept and learner event evidence. No inferred skill or model signal appears. A suitable contest on the local date replaces the ordinary activity list; a near-future contest annotates practice preparation, and a recent contest can supply an upsolve fallback. Suitability is provider or manual evidence, never inferred by the policy.

`GET /api/state` returns owner-scoped preferences, curriculum and annotations, raw events, derived progress/timers, contests and their source status, delivery hook result, and the daily mission. Before the scheduler freezes a snapshot it returns a computed, unpersisted mission with `snapshot_status: "preview"`; after freezing it returns the immutable snapshot with `snapshot_status: "frozen"`. This keeps the dashboard usable during local development without claiming a scheduled delivery. `PUT /api/preferences` accepts `weekday_minutes` (1–480) and `target_keys`; `POST /api/events` accepts one full v1 envelope; `GET /api/export` and `POST /api/import` use the M3 export envelope; `POST /api/contests/manual` accepts ID, title, platform, HTTPS URL, UTC start, duration minutes and explicit suitability. Mutations require same-origin JSON requests and are body-size bounded. The Worker authenticates static assets as well as API routes; `assets.run_worker_first` must remain true.

## Validation and rollout

Run `node --test hosted/test/core*.test.mjs` and the existing Python offline suite. Test authentication rejection, ownership, append/import idempotency, conflict rollback, timer recovery, future corrections at cutoff, eligibility, deterministic ties and due remediation. The local core suite passed 12 tests. The combined core, notification, backup, provider and UI suites passed 44 of 44. The separate loopback HTTP test could not bind `127.0.0.1` (`EPERM`) in the restricted sandbox; the parent run has a browser smoke receipt. Runtime D1, Access, quotas, backups, hosting and live delivery still require separate verification. Do not deploy or turn on schedules from this local step.


## Final integration amendments

Policy `deterministic-v2` includes a configurable seven-entry Monday–Sunday
weekly_rhythm (practice/review/virtual/upsolve), defaulting to practice, review,
practice, review, practice, virtual, upsolve. A suitable contest today overrides
it. Virtual/upsolve durations are planning budgets, never observed time; absent a
chosen past contest the activity explicitly asks the learner to select one.
Approved transfer candidates require independently demonstrated source evidence;
recording success still requires an explicit user outcome. Old two-field
preferences/backups remain accepted and receive default rhythm values.

`relationships.mjs` exposes reviewed graph, metadata and lexical providers with
bounded candidates, evidence and curriculum version. Target-conditioned signals
use reciprocal rank fusion with k=60; raw scores from incompatible spaces are
never averaged. Final prerequisite/access checks remain policy-owned. Disabled
or failed providers remove that signal without blocking mission generation.
Mission SQL triggers prevent snapshot mutation. The Node-to-Python attempt
contract now also rejects self-transfer and imported activity-type assertions.

Core full backup/participation/provider/operations routes are integrated. Request
instrumentation records only coarse path classes/status/count/wall durations;
no event contents, IP, email, tokens or raw query strings. Actual CPU/billing
quotas remain a deployed measurement gate. Static files receive restrictive CSP,
no-referrer and nosniff headers. All routes remain JWT authenticated.
