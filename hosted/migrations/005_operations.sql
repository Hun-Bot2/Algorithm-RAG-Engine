CREATE TABLE runtime_observations (
    utc_date TEXT NOT NULL,
    route TEXT NOT NULL,
    status INTEGER NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0,
    elapsed_sum_ms REAL NOT NULL DEFAULT 0,
    elapsed_max_ms REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (utc_date,route,status)
);
INSERT INTO schema_migrations(version,name) VALUES (5,'005_operations');
