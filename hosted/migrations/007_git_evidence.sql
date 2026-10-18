-- Mastery system: Git evidence (specs/mastery-system/design.md section 2-3). Additive only.
CREATE TABLE git_repositories (
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    repo_id TEXT NOT NULL,
    branch TEXT NOT NULL,
    path_hint TEXT NOT NULL,
    convention_version TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    PRIMARY KEY (learner_id, repo_id)
);
CREATE TRIGGER git_repositories_no_update BEFORE UPDATE ON git_repositories
BEGIN SELECT RAISE(ABORT, 'git repositories are append-only'); END;
CREATE TRIGGER git_repositories_no_delete BEFORE DELETE ON git_repositories
BEGIN SELECT RAISE(ABORT, 'git repositories are append-only'); END;
CREATE TABLE git_observations (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL UNIQUE,
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('git','manual_correction')),
    payload TEXT NOT NULL,
    supersedes TEXT UNIQUE REFERENCES git_observations(entry_id),
    canonical TEXT NOT NULL,
    CHECK (supersedes IS NULL OR supersedes != entry_id),
    CHECK (occurred_at <= recorded_at)
);
CREATE INDEX git_observations_owner ON git_observations(learner_id,sequence);
CREATE TRIGGER git_observations_no_update BEFORE UPDATE ON git_observations
BEGIN SELECT RAISE(ABORT, 'git evidence is append-only'); END;
CREATE TRIGGER git_observations_no_delete BEFORE DELETE ON git_observations
BEGIN SELECT RAISE(ABORT, 'git evidence is append-only'); END;
CREATE TABLE git_watermarks (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL UNIQUE,
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('git','manual')),
    payload TEXT NOT NULL,
    supersedes TEXT UNIQUE REFERENCES git_watermarks(entry_id),
    canonical TEXT NOT NULL,
    CHECK (supersedes IS NULL OR supersedes != entry_id),
    CHECK (occurred_at <= recorded_at)
);
CREATE INDEX git_watermarks_owner ON git_watermarks(learner_id,sequence);
CREATE TRIGGER git_watermarks_no_update BEFORE UPDATE ON git_watermarks
BEGIN SELECT RAISE(ABORT, 'git watermark log is append-only'); END;
CREATE TRIGGER git_watermarks_no_delete BEFORE DELETE ON git_watermarks
BEGIN SELECT RAISE(ABORT, 'git watermark log is append-only'); END;
INSERT INTO schema_migrations(version,name) VALUES (7,'007_git_evidence');
