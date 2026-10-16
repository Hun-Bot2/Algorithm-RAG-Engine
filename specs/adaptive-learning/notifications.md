# M7 notification contract: local implementation

## Research summary

The legacy `DailyReviewBot` formats a header, count, bounded problem sections and a context footer from a recommendation artifact. That path remains a separate compatibility reference. The new path formats an immutable mission using canonical reviewed links and a stable marker. Slack's `chat.scheduleMessage` returns a scheduled ID and must omit `metadata`; `chat.scheduledMessages.list` and `conversations.history` paginate with cursors. Slack returns HTTP 429 with `Retry-After`. A scheduled acceptance is not proof that a message posted.

## Requirements

- The dispatcher SHALL run only when `DELIVERY_ENABLED=true`, a bot token, channel, and a single configured `DELIVERY_LEARNER_ID` are present. No code path SHALL activate Cron triggers or send during local tests.
- The daily mission SHALL freeze at the 05:40 Asia/Seoul cutoff, including recovery later that morning. A previously frozen mission with another cutoff or curriculum version SHALL block delivery. The 06:00 local target SHALL use `chat.scheduleMessage` before 06:00; recovery after 06:00 SHALL use `chat.postMessage` only through 06:10. Later unsent deliveries SHALL become `missed`.
- Outbox uniqueness SHALL be `(learner_id, local_date, channel_id)`. The message SHALL include a stable, non-PII uniqueness marker. Claims SHALL be atomic: one ready row can enter sending once, and a stranded sending lease SHALL become unknown rather than ready.
- HTTP 429 SHALL persist `retry_at` from `Retry-After` without sleeping. Network failure, 5xx, malformed response or acceptance timeout SHALL become `unknown` and SHALL never auto-resend.
- Reconciliation SHALL search bounded pages of scheduled messages and channel history for the marker. A match may establish `scheduled` or `posted`; absence, including exhaustive current listing, SHALL NOT establish non-acceptance because Slack visibility may lag. Incomplete pagination SHALL leave `unknown`.
- `scheduled` SHALL not be reported as `posted` until history confirms it. Formatter text and blocks SHALL stay within Slack bounds and use canonical links from the approved curriculum; no title-to-slug guess shall create a production link.

## Design and task breakdown

`002_delivery.sql` adds an append-only mission reference in an outbox row with state and claim fields. `notifications.mjs` owns formatting, message marker, row insertion/claim/transitions and owner-scoped reads. `dispatcher.mjs` owns time windows, Slack transport and reconciliation. Existing API state reads the outbox through `listDeliveries`. Validation uses SQLite-backed D1-compatible tests with injected fetch responses for 429, accepted timeout, pagination and concurrent claims.

## Validation and limits

Executed `node --test hosted/test/notifications.test.mjs`: 11 of 11 passed. The combined core, notification, backup, provider and UI suites passed 44 of 44. Asset verification found 7 matching assets and 1 explicitly missing manifest entry. `git diff --check` and Node syntax checks passed. The loopback HTTP test is separate because the sandbox rejects binding `127.0.0.1`; the parent run has an authorized browser smoke receipt.

These checks do not establish Slack scopes, real acceptance, delivery, D1 quota, scheduled Cron, or a 30-day reliability result. Release requires a private authorized smoke test, scope and channel review, hosted recovery check and explicit schedule activation. No live Slack request or active Cron trigger was made by this implementation.
