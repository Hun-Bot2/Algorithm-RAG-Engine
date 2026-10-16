CREATE TABLE provider_cache (
    learner_id TEXT NOT NULL,
    provider_key TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    lease_until TEXT,
    lease_token TEXT,
    PRIMARY KEY (learner_id, provider_key)
);
INSERT INTO schema_migrations(version,name) VALUES (3,'003_providers');
