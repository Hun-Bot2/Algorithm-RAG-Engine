# Validation and release notes — 2026-10-18

## Verified implementation

Extended the existing local Node/SQLite application, not a replacement database
or recommendation stack. Learner migrations are unchanged. New projections use
existing events, Git ledgers, catalog selection and immutable recommendation
snapshots. Full backup v2 retains inactive archived recommendations across restore.
The optional self_minutes manual payload field preserves unknown versus zero time.

`node --test hosted/test/*.test.mjs`: **130/130 passed**, including real loopback
HTTP integration with socket permission. Tests use :memory: or temporary learner
stores. Relevant focused suites also passed during each phase. Worker bundle
compatibility was checked using pinned Wrangler `deploy --dry-run`; it exited
successfully without deployment. Backup hashing uses Web Crypto, so the shared
Worker-safe backup path does not gain a Node-only import.

Acceptance coverage:

| Requested journey | Evidence |
|---|---|
| 3 Easy variations → 1/2 Medium → transfer | progression tests: distinct IDs/aliases, variation signatures, configurable 1-Medium and custom 4-Easy/3-Medium paths |
| Transfer and normal spaced review | progression + review replay tests: 7+ day retention, existing 1/3/7/14/30/60 intervals |
| Failed transfer → review/repair → different transfer | priority queue + progression: repeated target never credits novel transfer; Medium repair then new target |
| Failed Easy/Medium without advancement | review and progression tests; Medium failure preserves completed Easy evidence |
| Repeated failures/regression/recovery | failure severity, urgency, stage regression and retention recovery; Hard evidence distinct and not inferred mastered |
| Sparse history/candidate gaps | conservative native calibration, missing variation/candidate reason, no fabricated filler |
| AtCoder missing concepts | semantic-only fallback labeled unverified and excluded from transfer credit |
| Duplicate identities/Git/snapshots | canonical aliases, Git idempotency/rewrite stops, retries across clocks, corrections and immutable snapshots |
| Real catalog + date-grouped UI | catalog-v4 read-only attachment and actual journal.view() on synthetic learner; Chrome mobile test |
| Existing backups/history | v1 import, v2 archived snapshot restore, flatten/deduplicate re-export, self-reported time roundtrip |

The journal also tests that timer-only events do not rotate the current problem,
date rollover creates a new snapshot/ages reviews, Git failures enter the same
queue, and unrelated concepts do not emit fake initial mastery-change records.

## Browser evidence

Actual headless Chrome 154, 390×844 mobile viewport, disposable in-memory
`browser-smoke` learner, read-only catalog-v4. O/X recording, durable history,
recommendation change, date filtering, accessible forms, timer recovery/confirmation,
offline stable-ID retry, mobile overflow and first-viewport date-group visibility
passed. Runtime errors: none. Synthetic service is stopped.

Receipt and screenshot:
`private-learning/mastery-journal-browser-final-v1/{receipt.json,journal.png}`.
The initial large timeline exposed an implementation bug (initial states for every
concept looked like changes); replay now compares against an empty-evidence baseline
and a regression test protects the fix. The final image shows seven actual entries,
not fabricated concept transitions. Browser proof is local synthetic UI evidence,
not a real learner outcome or deployed-runtime proof.

## Preservation

- All **191 protected research files** match their recorded hashes.
- catalog-v3 and catalog-v4 match their existing receipts.
- Real learner database remains at migrations 1–3, with 0 learners and 0 events;
  read-only inspection matches the audit. SHA-256 remains
  `a6b96ed8665b6a31cd337510d49fcb308eb0a8d47e19c838c7867e86c3534dfe`.
- Unrelated Ralph modifications and untracked production-test work are preserved.
- No push, deployment, live provider collection, Slack send or visibility change.

## Meaningful deviations and limits

The journal deliberately composes reviews/calibration with curriculum practice,
rather than showing the legacy exploit/explore plan as a second competing plan.
The legacy CLI plan remains frozen by date; new journal revisions use evidence,
settings, catalog and implementation fingerprints. Weekend curriculum/review is
available in the journal while contest planning remains accessible separately.

Pattern ordering/prerequisites are authored configuration. Catalog tag/concept
variation signatures are a provisional diversity proxy, not verified instructional
variation. AI-reviewed tags and self-reported independence remain labeled evidence;
semantic matches cannot establish algorithm equivalence. Three Easy plus Medium
counts never produce transfer/retention credit automatically. Advanced evidence is
kept separate from foundational progression.

No real learner data establishes retention, transfer success or educational gains.
Weekly reports will expose denominators and missingness as real evidence arrives.
Passing journeys proves policy mechanics only. Source statements, original catalogs,
embeddings and historical research quality conclusions were not altered.

See [runbook](runbook.md) for actual startup/CLI/Git usage and path compatibility.
Machine-readable preservation/validation receipt: [validation.v1.json](validation.v1.json).

## Automatic Git reflection follow-up

The local server now polls registered repositories every five seconds using the
existing ingester and exposes a read-only local revision/status endpoint. Idle,
visible journal views refresh on evidence changes without replacing an open result
form; filters are restored. No Git hooks, automatic commits or external fetches.

`node --test hosted/test/*.test.mjs`: **135/135 passed** after integration. A real
loopback test creates a temporary Git repository, commits a failed solution and
observes automatic ingestion/revision change without invoking ingest; a repeated
tick leaves exactly one observation. Additional cases cover rewriting-history halt,
per-repository error containment, close waiting and missing outcome/hint provenance.
The prior mobile browser receipt predates automatic polling; it is not presented
as proof of this new timer-driven refresh. Actual repositories/learner DBs were
not registered, changed or ingested during implementation.

## Markdown-first ingestion — 2026-10-18

Full Node regression suite: **143/143 passed**, with loopback socket permission.
The initial restricted run could not bind localhost (EPERM); the permitted rerun
passed. The new eight-test Markdown suite covers outcome/hints validation,
per-file multi-problem metadata, trailer precedence, note-edit deduplication,
new attempt tokens, conflicting reused tokens, invalid blocks, template exclusion,
folder README identity and legacy behavior. Sync tests additionally verify that
warnings survive idle polling and clear after clean subsequent ingestion.
The actual sibling solution template was parsed: unfilled placeholders reject,
and filled metadata yields independent success. Post-validation learner SQLite SHA-256 was recorded as
fd0745a387cce626d7431701d932cd76770a11c306c792849c6a24eaa9dc0141;
this is a current receipt, not a before/after integrity assertion.
No real solutions were imported; no schema, remote hooks, fetch/pull or deployment
was added. The existing sibling convention checker may still warn about absent
Outcome commit trailers; the new ingester accepts valid Markdown metadata.

### Simplified entry follow-up

Focused Markdown and Git regression tests: 16/16 passed. Schema 2 tests verify
O/X-only validation, rejection of hints, default practice, explicit review and
actual committed success/failure ingestion. Existing schema 1 cases still pass.
The sibling Markdown template and guide now use YY-MM-DD attempt tokens.

### Shared solutions-repository cleanup

Engine regression: 145/145 passed (including local HTTP integration). Standalone
solutions-repository checker: 5/5 passed, including schema 2 success/failure/review,
invalid assisted/hints and path mismatch. Parser snapshot equality was verified
after replacing only the engine keyPattern import. Working-tree originals were
backed up to /tmp/algorithm-solution-before-format-cleanup.tar.gz, and obsolete
scaffolding is retained under templates/legacy/. No push or real learner import.
