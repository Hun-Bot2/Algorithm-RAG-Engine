# Local history usage

This is the M3 offline evidence foundation, not the hosted application. It requires
Python 3.10+ and no additional packages, credentials or model service. Use a private
directory (0700); keep its database, WAL/journals and exports out of static assets.
The CLI creates database/export files with mode 0600. Existing file permissions
are the owner's responsibility. A learner ID is local partitioning, not login.

```bash
mkdir -m 700 private-learning
python3 scripts/learning/history.py --db private-learning/history.sqlite3 --learner me init
```

Create `private-learning/event.json` with explicit evidence (use your actual UTC
timestamps, not the example's times). For imported acceptance, use
`reported_completion`, `hints: null`, source `platform_import`, and provide
`platform` and `external_id` in payload. No duration or mastery is inferred.

```json
{
  "schema_version": 1,
  "event_id": "manual-attempt-001",
  "learner_id": "me",
  "occurred_at": "2026-10-16T01:00:00Z",
  "recorded_at": "2026-10-16T01:02:00Z",
  "kind": "attempt",
  "source": "manual",
  "payload": {
    "problem_key": "leetcode:two-sum",
    "outcome": "assisted",
    "hints": true
  }
}
```

```bash
python3 scripts/learning/history.py --db private-learning/history.sqlite3 --learner me append --file private-learning/event.json
python3 scripts/learning/history.py --db private-learning/history.sqlite3 --learner me history
python3 scripts/learning/history.py --db private-learning/history.sqlite3 --learner me export --file private-learning/backup.json
python3 scripts/learning/history.py --db private-learning/restored.sqlite3 --learner me import --file private-learning/backup.json
```

Export refuses to overwrite an existing backup. Import is one transaction and
idempotent for identical events. Corrections use a fresh event ID and `supersedes`
pointing at the current event, preserving the old record. Export includes both.
To recover, import into a new database and compare exports before switching paths.
Never delete the original database until recovery is independently verified.

Timer payloads and ordering are in `design.md`. A start without stop remains open;
a stop without confirmation remains unconfirmed. Confirm the actual studied
subinterval explicitly, bounded by those actions. Submit timestamps do not close
timers. Confirmed overlapping intervals contribute only their elapsed-time union.
This is recorded evidence, not a verified attention measurement.

Validation: `python3 -m unittest discover -s tests -v`.
