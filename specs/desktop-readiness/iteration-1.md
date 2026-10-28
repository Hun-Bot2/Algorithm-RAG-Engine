# Desktop readiness — iteration 1 (2026-10-28)

## Baseline and scope

Branch main; HEAD 5fe47c864dabb5a94e2e97aa6075ee371064863f. Initial untracked work: `.claude/`, `specs/distribution-rights/`; preserved. Recent commits implement participation-first contests, calendar/research UI and policy v3. Environment: Darwin arm64, Node v26.8.1, Apple Git 2.54.0. No Windows or Intel Mac execution. No packaging or performance benchmarks conducted.

Governing sources inspected: AGENTS.md and mastery-system/design.md (canonical local SQLite, single writer, additive ledgers). User's desktop direction supersedes AGENTS.md's older personal-only product limitation. This is a bounded first-pass source audit, not a full security or release certification.

## Research summary and inventory

| Disposition | Files / responsibility | Evidence and rationale | Risk / affected checks |
|---|---|---|---|
| KEEP | hosted/mastery/{ledger,planner,progression,reviews}.mjs | Versioned evidence and policy; preserve policy v3 behavior | Policy changes require explicit spec; mastery and solved-outcomes tests |
| KEEP | hosted/mastery/{git-ingest,git-convention,daily-log,markdown-attempt}.mjs | Local committed-history ingestion; append-only watermarks and rewrite halt; supported platform paths | LeetHub layouts are not generally proven compatible; git/markdown/daily tests |
| REFACTOR | hosted/mastery/git-sync.mjs | Local polling only, no fetch/pull; synchronous Git work underneath | UI blocking and polling cost unmeasured; git-sync tests |
| REFACTOR | hosted/test/helpers/d1.mjs; hosted/{cli,local}.mjs | Runtime imports adapter under test/. Actual node:sqlite, 5s timeout, explicit migration and batch transactions | Move only with caller audit and migration/rollback regression; core/backup/local/CLI tests |
| KEEP | hosted/backup.mjs; hosted/migrations/001–008 | Backup v2 preserves registrations and inert recommendation archives; additive migrations | Cross-machine paths and recovery UX unresolved; backup tests |
| REFACTOR | hosted/{cli,local}.mjs | Root derived via URL.pathname; defaults private-learning under checkout | Encoded spaces and Windows drive paths need fileURLToPath; app data directory contract needed |
| KEEP with distribution gate | hosted/mastery/catalog.mjs; assets/; curriculum/ | Immutable catalog and hash checks | Private catalog licensing unresolved; catalog/asset verification before distribution |
| REFACTOR | hosted/public/{app,workspace,workspace-data}.mjs | Existing local UI, presentation-only calendar transformations | No final information architecture/accessibility evidence; workspace/UI/browser checks |
| CONSOLIDATE candidate | hosted/{cli,local}.mjs startup defaults | Both configure database/catalog roots | First extract shared configuration only after app data spec; CLI/local tests |
| DEFER | hosted/worker.mjs; wrangler dev dependency; provider/delivery adapters | Optional hosted/runtime paths still present; not evidence of obsolescence | Retain callers/tests; exclude from eventual desktop packaging allowlist |
| ARCHIVE disposition only | research/; specs/production-recommender/; historical README portions | Scientific and historical value, not desktop runtime | No move/delete: pinned artifacts and receipts may depend on paths |
| DEFER | hosted/mastery/explanations.mjs and explanation-store.mjs | Optional AI presentation and separate durable ledger | Terms, credentials, backup scope; not core offline dependency |
| REMOVE | none established | No safe deletion proven | No deletion performed |

## Priorities

P0 release gates (not demonstrated current data corruption): authorized distributable catalog/assets; explicit user-data location and non-destructive upgrade/recovery contract; actual cross-platform runtime checks. Never package private-learning or statements by default. Existing release cleanup is not repeated.

P1: runtime adapter in test/; URL.pathname portability; missing Git/credentials recovery; separate local polling from remote clone/fetch policy; synchronous Git/SQLite responsiveness; LeetHub fixtures including source-only import; interrupted migration and multi-process contention checks. Current git wrapper uses execFileSync and can buffer 256 MiB: real resource usage unmeasured. Register verifies an existing local branch; no remote registration implementation established. Rewrites halt rather than silently resetting evidence. No credential workflow or automatic remote sync is established by local polling.

P2: configurable timezone (currently KST), onboarding/manual/offline states, accessible keyboard flows, research/deployment boundaries, distribution notices/SBOM. Catalog metadata and source code must be distinguishable from README statements during format expansion.

## Requirement and design for selected fix

WHEN Git polling closes, the service SHALL report started=false, closed=true and, after outstanding work settles, running=false. Closing SHALL continue awaiting the current tick; later start SHALL remain rejected. No evidence, recommendation, migration or scheduling semantics change.

Smallest implementation: clear started with timer teardown in git-sync.close. Tests use temporary Git repos and in-memory SQLite; real learner DB is never opened. Risk: callers interpreting started as historical rather than active lifecycle flag; field pairs with close/start and now reflects shutdown consistently.

## Tasks and validation

- [x] Inspect branch, dirty state, recent commits, runtime and relevant specs.
- [x] Record module inventory and unresolved product gates.
- [x] Fix close lifecycle flag and extend existing shutdown regression.
- [x] Run focused Git regression: `node --test hosted/test/mastery-git-sync.test.mjs hosted/test/mastery-git.test.mjs` — 10/10 pass.
- [x] Attempt baseline `node --test hosted/test/*.test.mjs` — exit 1, loopback bind EPERM observed (including explanation restart child). This run is NOT a passing baseline. Raw log: /tmp/desktop-baseline-tests.log; focused log: /tmp/desktop-focused.log (local ephemeral files).
- [x] `git diff --check` — pass.
- [ ] Full suite in socket-enabled environment; browser/device/Windows/Intel Mac checks.

Python, asset verifier and research suites were not run: no Python, pinned asset, catalog or research changes. No performance numbers are inferred from test duration. Electron/Tauri selection deferred until equivalent workloads can actually be packaged and measured.

## Reviewer handoff / next iteration

Inspect hosted/mastery/git-sync.mjs close ordering and hosted/test/mastery-git-sync.test.mjs lifecycle assertions. Check active ticks still complete, close reports started=false, repeated start after close rejects, and ingestion semantics remain unchanged. Diff contains no database schema or policy changes.

Next: fix file-URL path conversion in CLI/local startup with a fixture checkout containing spaces/non-ASCII characters, then verify Windows path behavior on Windows. Afterwards extract the existing SQLite adapter from test/ without changing implementation and run migration/backup/CLI/local regressions. Remote Git sync requires its own requirements and credentials/conflict recovery design; do not add pull implicitly.

Suggested commit: `fix(sync): report polling stopped after close`.
