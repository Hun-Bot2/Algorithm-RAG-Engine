# Data location and upgrade contract (2026-10-29)

Scope: approved review-log F4. Preserve existing CLI/server defaults and explicit --db/database overrides. No automatic relocation of existing learner data is authorized by this change.

## Requirements

- DR-1: A future installed application SHALL store mutable state outside its installation directory: macOS Application Support/Algorithm RAG Engine and Windows LOCALAPPDATA/Algorithm RAG Engine. User-selected directories SHALL be supported. Existing checkout paths SHALL remain unchanged until an explicit, validated import is requested. Desktop onboarding implementation is deferred with framework selection.
- DR-3a: Before applying pending migrations to an existing nonempty file, the adapter SHALL produce a SQLite-consistent backup including committed WAL data. It SHALL NOT overwrite any backup. Backup failure SHALL abort migrations.
- DR-3b: A database with a migration version newer than this app SHALL be rejected before migrations; receipt drift SHALL also be rejected before changes.
- DR-3c: New empty databases and in-memory fixtures SHALL NOT create migration backups. Reopening an up-to-date DB SHALL NOT create one.
- DR-3d: Backup directories SHALL be private on POSIX; backups SHALL be retained until the user explicitly removes them. There SHALL be no automatic restore or data deletion.

## Design and recovery

The synchronous Node SQLite runtime adapter lives in hosted/runtime/database.mjs. A compatibility re-export preserves test imports. No new dependency. Existing per-migration transactions and 5-second timeout remain unchanged.

SQLite VACUUM INTO creates before.sqlite3 in a unique .migration-backup-* directory adjacent to the explicit DB file; production code never copies a live database file with fs copy. This requires adjacent free space and write permission; failure aborts. createDatabase returns migrationBackup (null when not created). No automatic restore: close all processes, preserve the failed/current DB and its sidecars, inspect the backup read-only, and select an explicit recovery destination. Explanation ledger and catalog are separate stores, not included in this backup.

Canonical deployment remains single-writer (mastery design). This snapshot is taken before migration, not an interprocess upgrade lock: stop other writers before upgrade. Concurrent process upgrade coordination and retention UI remain desktop release gates. Windows ACL behavior and fsync/power-loss durability are not proven by POSIX mode flags or unit tests.

## Tasks

- [x] Specify safe locations without silently moving existing databases.
- [x] Reject newer schemas and drift, snapshot existing pending-migration databases, fail closed on backup error.
- [x] Move adapter ownership to runtime with compatibility re-export.
- [ ] Execute Windows full regression on an actual Windows host (unavailable here).
