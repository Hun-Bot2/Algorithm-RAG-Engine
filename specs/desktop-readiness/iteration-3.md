# Iteration 3 — approved desktop fix plan (2026-10-29)

User confirmed scope: review-log.md F1–F5 only; recommendation_audit policy changes excluded. Baseline main 5fe47c864dabb5a94e2e97aa6075ee371064863f plus existing uncommitted changes. Runtime Node 26.8.1, Darwin arm64, Apple Git 2.54.0.

## Delivered, in dependency order

1. f1a488a — prior approved git-sync shutdown flag and regression.
2. 696f288 — F1/F2: CLI main guard and decoded root paths, copied-checkout regression with spaces/Korean/#/%41.
3. dbd70bf — F3: correctly encoded immutable SQLite URI, read-only/no-stray-file regressions.
4. cc09cf5 — F4: data-safety contract; newer-schema/receipt preflight, SQLite-consistent pre-migration backup (including WAL), abort on backup failure, resource cleanup; production adapter moved to runtime with test compatibility export. New/empty/in-memory/current DBs do not receive unnecessary backups. Existing defaults unchanged.
5. 42ef61e — F5: README policy-v3 wording and historical recommendation audit banner.

## Validation receipts

- `node --test hosted/test/*.test.mjs`: 254 tests, 254 pass, 0 fail, exit 0 with loopback access. Restricted first run encountered EPERM; successful run supersedes that environment failure.
- `python3 -m unittest discover -s tests`: 74 tests, OK, exit 0.
- `node --test hosted/test/database-migrations.test.mjs`: 3 pass after final formatting. Tests assert newer-schema preservation, WAL snapshot integrity/data, no repeated backups and backup-failure abort.
- `git diff --check`: pass.
- Synthetic/in-memory stores only. No production learner DB migration, automatic relocation, deletion, restore or remote push performed.

## Limits and handoff

No Windows machine or Intel Mac available: review-log's actual Windows full-suite step remains unexecuted. No packaging/browser benchmark implied. App-data locations are specified for future installers; existing checkout defaults intentionally preserved. Upgrade snapshots require adjacent free space/write permission; backups remain until explicit deletion. Canonical single-writer contract still applies: no new cross-process upgrade lock. No framework, remote Git synchronization or recommendation-policy changes.

Reviewer should inspect CLI/local URL conversion, catalog ATTACH encoding, runtime/database.mjs preflight/backup/migration ordering and connection cleanup, plus desktop-paths/catalog-paths/database-migrations tests. Confirm learner evidence and recommendation semantics remain unchanged.

Unrelated work preserved outside commits: .claude/, specs/distribution-rights/, research/recommendation_audit/, and statistical-audit additions in specs/mastery-journal/requirements.md and specs/mastery-system/design.md. Earlier iteration reports remain historical; status.md reflects current implementation. Future release gate: execute Windows suite and inspect Windows ACL/SQLite URI behavior.
