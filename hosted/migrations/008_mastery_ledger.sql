-- Mastery system: feedback, ratings, catalog switches and immutable snapshots. Additive only.
CREATE TABLE practice_feedback (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL UNIQUE,
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('manual')),
    payload TEXT NOT NULL,
    supersedes TEXT UNIQUE REFERENCES practice_feedback(entry_id),
    canonical TEXT NOT NULL,
    CHECK (supersedes IS NULL OR supersedes != entry_id),
    CHECK (occurred_at <= recorded_at)
);
CREATE INDEX practice_feedback_owner ON practice_feedback(learner_id,sequence);
CREATE TRIGGER practice_feedback_no_update BEFORE UPDATE ON practice_feedback
BEGIN SELECT RAISE(ABORT, 'practice feedback is append-only'); END;
CREATE TRIGGER practice_feedback_no_delete BEFORE DELETE ON practice_feedback
BEGIN SELECT RAISE(ABORT, 'practice feedback is append-only'); END;
CREATE TABLE rating_observations (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL UNIQUE,
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('manual','codeforces_api')),
    payload TEXT NOT NULL,
    supersedes TEXT UNIQUE REFERENCES rating_observations(entry_id),
    canonical TEXT NOT NULL,
    CHECK (supersedes IS NULL OR supersedes != entry_id),
    CHECK (occurred_at <= recorded_at)
);
CREATE INDEX rating_observations_owner ON rating_observations(learner_id,sequence);
CREATE TRIGGER rating_observations_no_update BEFORE UPDATE ON rating_observations
BEGIN SELECT RAISE(ABORT, 'rating observations is append-only'); END;
CREATE TRIGGER rating_observations_no_delete BEFORE DELETE ON rating_observations
BEGIN SELECT RAISE(ABORT, 'rating observations is append-only'); END;
CREATE TABLE catalog_switches (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL UNIQUE,
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('manual')),
    payload TEXT NOT NULL,
    supersedes TEXT UNIQUE REFERENCES catalog_switches(entry_id),
    canonical TEXT NOT NULL,
    CHECK (supersedes IS NULL OR supersedes != entry_id),
    CHECK (occurred_at <= recorded_at)
);
CREATE INDEX catalog_switches_owner ON catalog_switches(learner_id,sequence);
CREATE TRIGGER catalog_switches_no_update BEFORE UPDATE ON catalog_switches
BEGIN SELECT RAISE(ABORT, 'catalog switches is append-only'); END;
CREATE TRIGGER catalog_switches_no_delete BEFORE DELETE ON catalog_switches
BEGIN SELECT RAISE(ABORT, 'catalog switches is append-only'); END;
CREATE TABLE recommendation_snapshots (
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    kind TEXT NOT NULL,
    local_date TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (learner_id, kind, local_date)
);
CREATE TRIGGER recommendation_snapshots_no_update BEFORE UPDATE ON recommendation_snapshots
BEGIN SELECT RAISE(ABORT, 'recommendation_snapshots are immutable'); END;
CREATE TRIGGER recommendation_snapshots_no_delete BEFORE DELETE ON recommendation_snapshots
BEGIN SELECT RAISE(ABORT, 'recommendation_snapshots are immutable'); END;
CREATE TABLE weekly_reports (
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    week_start TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (learner_id, week_start)
);
CREATE TRIGGER weekly_reports_no_update BEFORE UPDATE ON weekly_reports
BEGIN SELECT RAISE(ABORT, 'weekly_reports are immutable'); END;
CREATE TRIGGER weekly_reports_no_delete BEFORE DELETE ON weekly_reports
BEGIN SELECT RAISE(ABORT, 'weekly_reports are immutable'); END;
INSERT INTO schema_migrations(version,name) VALUES (8,'008_mastery_ledger');
