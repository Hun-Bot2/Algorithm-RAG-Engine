CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS learners (
    learner_id TEXT PRIMARY KEY NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    preferences TEXT NOT NULL DEFAULT '{"weekday_minutes":60,"target_keys":[]}'
);
CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('attempt','timer_started','timer_stopped','timer_confirmed')),
    source TEXT NOT NULL CHECK (source IN ('manual','platform_import')),
    payload TEXT NOT NULL,
    supersedes TEXT UNIQUE REFERENCES events(event_id),
    canonical TEXT NOT NULL,
    CHECK (supersedes IS NULL OR supersedes != event_id)
);
CREATE INDEX IF NOT EXISTS events_owner ON events(learner_id, sequence);
CREATE TRIGGER events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TABLE IF NOT EXISTS mutation_guard (
    ok INTEGER NOT NULL CHECK (ok = 1)
);
CREATE TABLE IF NOT EXISTS mission_snapshots (
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    local_date TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (learner_id, local_date)
);
CREATE TABLE IF NOT EXISTS manual_contests (
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    contest_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (learner_id, contest_id)
);
INSERT INTO schema_migrations(version, name) VALUES (1, '001_core');
