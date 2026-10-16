CREATE TABLE learners (
    learner_id TEXT PRIMARY KEY NOT NULL
);
CREATE TABLE events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('attempt', 'timer_started', 'timer_stopped', 'timer_confirmed')),
    source TEXT NOT NULL CHECK (source IN ('manual', 'platform_import')),
    payload TEXT NOT NULL,
    supersedes TEXT UNIQUE REFERENCES events(event_id),
    CHECK (supersedes IS NULL OR supersedes != event_id)
);
CREATE INDEX events_owner ON events(learner_id, sequence);
CREATE TRIGGER events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
