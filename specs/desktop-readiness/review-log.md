# Desktop readiness: independent review log

Append one section per implementation iteration. Severity: P0 blocker, P1 important, P2 improvement.
Evidence labels as in `status.md` (exec / static / reported / unverified).

## R1 — review of iteration-1 (2026-10-29)

Reviewed: HEAD `5fe47c8` plus uncommitted diff to `hosted/mastery/git-sync.mjs` and
`hosted/test/mastery-git-sync.test.mjs`; report `iteration-1.md`. Environment: macOS arm64, Node 26.8.1.

### Executed

| Check | Result |
|---|---|
| `node --test hosted/test/*.test.mjs` | exec: 245/245 pass, exit 0. The EPERM failure in iteration-1 was the implementer's sandbox, not a code defect |
| `python3 -m unittest discover -s tests` | exec: 74 pass |
| `python3 scripts/learning/assets.py verify` | exec: 7 verified, 1 explicitly missing (pre-existing) |
| CLI copied to a directory with a space, `status --db <tmp>` | exec: exit 0, no output, DB not created |
| `ATTACH 'file:<path>?mode=ro&immutable=1'` with `#`, `%41`, space | exec: space ok; `#` attaches an empty DB and creates a stray empty file at the truncated path; `%41` fails to open |
| Research suites, browser smoke, Windows | not run (no research/UI change; no Windows host) |

### Accepted implementation

- git-sync `close()` now clears `started`. Correct and harmless: `started` has no consumer outside tests
  (static: grep of hosted/), `/api/local/sync` only passes it through. The extended test asserts started/running/closed
  after an in-flight tick and rejection of a later `start()`; it would fail without the change. Low value, but acceptable
  as a commit of its own: `fix(sync): report polling stopped after close`.
- Inventory: no deletion proposed, research/historical files kept in place, worker/explanations deferred rather than
  removed. Agreed: nothing supports removal.

### Findings

**F1 — P0 (release blocker; no data loss today). CLI main guard fails on spaced paths and on Windows.**
`hosted/cli.mjs:282` compares `import.meta.url` with `` `file://${process.argv[1]}` ``. Observed: from `/…/repo copy/`
the CLI exits 0 silently (exec). On Windows the right side is `file://C:\…`, never equal to `file:///C:/…` (static),
so every CLI command would be a no-op. Iteration-1 named only the `ROOT` derivation, not this guard.
Expected: same guard as `hosted/local.mjs:83` (`pathToFileURL(resolve(process.argv[1])).href`).
Fix: one-line change. Verify: test that spawns the CLI from a temp directory whose name contains a space and a
non-ASCII character and asserts output; run on Windows.

**F2 — P1. `ROOT` keeps percent-encoding.** `hosted/cli.mjs:54`, `hosted/local.mjs:17` use `new URL('..').pathname`.
Observed: `…/dir%20with%20space` (exec). Default DB path would become a new `%20` directory, i.e. an empty learner DB
next to the real one, while the user's data looks lost. Fix: `fileURLToPath(new URL('..',import.meta.url))`.
Verify: same spawned-CLI test asserts the default DB path. (Iteration-1 P1, confirmed and upgraded with a scenario.)

**F3 — P1. Catalog attach builds a SQLite URI by string concatenation.** `hosted/mastery/catalog.mjs:26`. Observed:
`#` in the path drops `?mode=ro&immutable=1`, opens read-write at the truncated path and creates an empty file; `%XX`
is decoded (exec). Windows drive paths through `file:` URIs are unverified. Iteration-1 classified catalog.mjs KEEP
without noting this. Fix: escape the path as a proper SQLite URI (`%`→`%25`, `#`→`%23`, `?`→`%3f`, `\`→`/` with
`file:///C:/…` on Windows) or open a separate read-only `DatabaseSync` handle. Verify: attach test over the four path
shapes above; assert no stray file is created.

**F4 — P1 (desktop upgrade/rollback). Migrations run silently and have no newer-schema guard.**
`hosted/test/helpers/d1.mjs` applies every pending migration on open and ignores versions it does not know (static).
After an auto-update rollback, an older build would write into a newer schema without warning; no backup is taken
before migrating. Fix (when the app-data contract is written, DR-3): refuse writes when `max(schema_migrations.version)`
exceeds the newest local file; back up before applying. Verify: test with a DB that has a fake version 999.
Moving this adapter out of `test/` (iteration-1 P1) is fine but is a rename; do it in the same change as DR-3, not alone.

**F5 — P2. README described tracking of help after ADR 005 removed it.** Fixed in documentation (below).

### Challenged claims in iteration-1

- "Baseline exit 1, not a passing baseline": true for that sandbox only; the suite passes with loopback sockets. Future
  reports should say which tests were blocked rather than treat the run as a baseline.
- "User's desktop direction supersedes AGENTS.md's personal-only limitation": no recorded user decision or ADR says so.
  Kept as open decision D1 in `status.md`; do not build external-user features until it is recorded.
- Raw logs cited under `/tmp/` are not reproducible evidence for later reviewers; record counts in the report.

### Documentation updated

- New `status.md` (living baseline: product, architecture, readiness tracker, draft requirements, open decisions,
  inventory) and this log. ReadME.md top section corrected; RECOMMENDATION_ENGINE/README.md dated banner;
  AGENTS.md desktop-readiness gotchas.

### Next iteration (recommended order)

1. F1 + F2 together with one spawned-CLI path test (smallest change, unblocks everything on Windows).
2. F3 with the path-shape attach test.
3. Write DR-1/DR-3 as requirements (app-data directory, pre-migration backup, newer-schema guard) before moving
   `helpers/d1.mjs`; then F4.
4. Run the full suite on a Windows machine; record results here.
Deferred: framework choice (D2) until a packaged measurement exists; remote Git sync until D3 has requirements.

## Implementation follow-up — F1–F5 (2026-10-29)

Completed source fixes and documentation in separate local commits; see iteration-3.md for hashes and validation. F4 now includes SQLite-consistent snapshots and runtime adapter ownership. Full Node 254/254 and Python 74/74 pass in the implementation run. Windows execution remains unavailable. Earlier R1 statements about broken paths and missing safeguards describe the reviewed pre-fix state.
