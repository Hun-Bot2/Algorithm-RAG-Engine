CREATE TRIGGER mission_snapshots_no_update BEFORE UPDATE ON mission_snapshots
BEGIN SELECT RAISE(ABORT, 'mission snapshots are immutable'); END;
CREATE TRIGGER mission_snapshots_no_delete BEFORE DELETE ON mission_snapshots
BEGIN SELECT RAISE(ABORT, 'mission snapshots are immutable'); END;
INSERT INTO schema_migrations(version,name) VALUES (6,'006_immutable_missions');
