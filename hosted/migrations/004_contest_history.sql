ALTER TABLE learners ADD COLUMN delivery_reconciliation_required INTEGER NOT NULL DEFAULT 0
    CHECK (delivery_reconciliation_required IN (0,1));
CREATE TABLE contest_participation (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL UNIQUE,
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    contest_id TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('live','virtual','upsolve')),
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('completed','partial','abandoned','unknown')),
    confirmed_minutes REAL,
    source TEXT NOT NULL CHECK (source = 'manual'),
    supersedes TEXT UNIQUE REFERENCES contest_participation(entry_id),
    canonical TEXT NOT NULL,
    CHECK (confirmed_minutes IS NULL OR confirmed_minutes >= 0),
    CHECK (supersedes IS NULL OR supersedes != entry_id)
);
CREATE INDEX contest_participation_owner ON contest_participation(learner_id,sequence);
CREATE TRIGGER contest_participation_no_update BEFORE UPDATE ON contest_participation
BEGIN SELECT RAISE(ABORT, 'contest participation is append-only'); END;
CREATE TRIGGER contest_participation_no_delete BEFORE DELETE ON contest_participation
BEGIN SELECT RAISE(ABORT, 'contest participation is append-only'); END;
INSERT INTO schema_migrations(version,name) VALUES (4,'004_contest_history');
