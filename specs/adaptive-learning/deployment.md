# Private pilot deployment and verification

## Current boundary

The Worker, D1 SQL, static mobile app and dispatcher are implementation artifacts.
No account, Access application, live D1 database or Slack destination is selected
by these files. Default scheduler/delivery/providers are disabled, no cron is configured,
and absent Access settings reject every request. The local Node service is bound
to 127.0.0.1 and is never deployed; it is not an authentication substitute.

Run `python3 scripts/learning/check_release.py` for a read-only configuration and
code preflight. Missing account/identity/channel choices remain user decisions.
Do not paste bot or Cloudflare tokens into chat or commit them to the repository.

## Reproducible local verification

Requires Python 3.10+ and Node with built-in SQLite (Node 26.8.1 verified).

```bash
python3 scripts/learning/build_hosted.py
python3 -W error::ResourceWarning -m unittest discover -s tests -v
cd hosted
npm ci
npm test
npm run dev
```

Use the loopback development app at http://127.0.0.1:8787. Private state defaults
to `private-learning/hosted.sqlite3` relative to the command's working directory;
set LOCAL_DATABASE to an explicit private path when needed. The default local
owner is separate from the authenticated hosted owner. Never mix browser smoke
records with real learning history.

Build a Worker without publishing:

```bash
WRANGLER_LOG_PATH=.wrangler/logs/dry-run.log WRANGLER_SEND_METRICS=false XDG_CONFIG_HOME=.wrangler/config npm run dry-run
node runtime-smoke.mjs
```

The runtime smoke uses local workerd/D1 with generated RSA test keys and intercepts
all outbound calls. It is distinct from Node SQL emulation and from deployed
Cloudflare proof. Some sandboxes require permission to bind loopback sockets.
Wrangler/Miniflare versions are pinned through the committed npm lockfile.
The runtime test uses Miniflare's exported v4-option converter for its installed
v5 API; the installed README still contains older constructor examples.

For actual browser verification, launch headless Chrome with a private temporary
profile and remote debugging restricted to loopback. Start the local app using
LOCAL_LEARNER_ID=browser-smoke and a separate LOCAL_DATABASE, then run
`node hosted/browser-smoke.mjs` from the repository root. Private screenshots and
receipts are written under ignored `private-learning/browser-evidence/`.

## Deployment steps after user selection

1. Select the Cloudflare account and allowed owner email, then create D1 and the
   Access application with that identity. Fill the database ID in wrangler.jsonc.
   Choose the route/hostname; do not open public registration.
2. Set Access issuer, audience and comma-separated allowed email configuration.
   Use Wrangler secret management for SLACK_BOT_TOKEN; use a scoped bot with
   chat:write and the history scope appropriate to the destination, plus needed
   scheduled-message permissions. Invite it only to the chosen private destination.
   Confirm the current API scopes in official docs during setup.
3. Export/back up any existing D1 data before applying additive SQL migrations.
   Apply hosted/migrations to the selected D1 database with Wrangler. Deploy
   with DELIVERY_ENABLED=false, PROVIDERS_ENABLED=false and no cron triggers.
4. Verify unauthenticated/wrong-user rejection on APIs AND static files, permitted
   login, owner-derived learner ID, timer recovery, export/restore and absence of
   historical data/embeddings/secrets from assets. Set DELIVERY_LEARNER_ID to the
   actual verified ID; do not derive it from a caller-supplied body.
5. Enable INSTRUMENTATION_ENABLED for the private pilot. `/api/operations` exposes
   aggregate route/status counts, wall durations and own event JSON character
   counts, without query strings, email, IP or event contents. Wall time is not
   billable CPU; measure Worker CPU, D1 rows/storage and requests in Cloudflare's
   account instrumentation separately before asserting free-tier feasibility.
6. Authorize one explicit Slack smoke message. Confirm Slack posted receipt,
   phone notification settings and actual phone alert separately. Record errors
   and missingness; a successful API response does not prove a phone alert.
7. Only after authentication, quotas, restore and the smoke pass, explicitly
   set SCHEDULER_ENABLED=true, enable delivery and a five-minute UTC cron. 05:40 KST freezes the snapshot;
   06:00 target / 06:10 deadline are computed in Asia/Seoul. No AI inference is
   performed. A personal computer can then be off; prove it during the pilot.
8. Record each day for 30 days (scheduled/posted/unknown/missed and observed phone
   alert separately). Review free-tier consumption and learning outcomes before
   declaring the pilot passed. Unknown sends require reconciliation, never blind
   retry. Turn delivery off first if a regression is found.

Optional provider sync is explicit per-account action; public handles stay private
in D1. Stale calendars and unmeasured submission coverage remain visible. No new
LeetCode scraping is implemented. Use manual calendar and learning records when
providers are unavailable.

## Restore and rollback

The browser's full learning backup includes append-only events, preferences,
manual contests, participation and read-only mission/delivery archives. Restore
only into empty owner state. Restored accounts have a durable delivery pause;
archives never recreate sendable outbox rows. Reconcile Slack before clearing the
pause with an operator-reviewed D1 update. Event-only imports remain available for
idempotent additions/corrections to existing history.

For complete operational rollback, export the selected D1 database through
Wrangler, retain the prior Worker deployment and policy/curriculum hashes, and
verify restoration locally before switching the application binding. No destructive
SQL down migration is provided. Disable delivery before DB rollback to avoid
replaying an older outbox. Keep historical datasets and vectors unchanged.

## Deferred model operations

Metadata/lexical baselines and local artifact validation run without model servers.
Optional encoders require separately pinned revisions, licenses and local input
contracts. No weights were downloaded as part of core operation. The 180 reviewed
judgments are evaluation-only and cannot train a model. Training requires a new
problem-disjoint adjudicated dataset, frozen configuration/resource budget and
explicit authorization. Owned compute availability and annotation effort remain
measured costs rather than assumed free resources.

Mission freezing has its own SCHEDULER_ENABLED gate and works with Slack disabled.
Recovery records both the fixed evidence cutoff and the actual prepared_at time;
it does not pretend a late build physically ran at 05:40. Current preference and
catalog snapshots are used during recovery, while learner events are as-of cutoff.
