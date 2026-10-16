CREATE TABLE IF NOT EXISTS delivery_outbox (
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    local_date TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    uniqueness_key TEXT NOT NULL UNIQUE,
    mission_hash TEXT NOT NULL,
    mission_payload TEXT NOT NULL,
    message_payload TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('ready','sending','scheduled','posted','unknown','missed')),
    retry_at TEXT,
    lease_until TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    scheduled_message_id TEXT,
    slack_ts TEXT,
    last_error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (learner_id, local_date, channel_id)
);
CREATE INDEX IF NOT EXISTS delivery_state_retry ON delivery_outbox(state,retry_at);
CREATE TRIGGER delivery_immutable_snapshot BEFORE UPDATE OF
    learner_id,local_date,channel_id,uniqueness_key,mission_hash,mission_payload,message_payload
    ON delivery_outbox
BEGIN SELECT RAISE(ABORT, 'delivery snapshot is immutable'); END;
INSERT INTO schema_migrations(version,name) VALUES (2,'002_delivery');
