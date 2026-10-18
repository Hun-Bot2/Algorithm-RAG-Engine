"""Trusted local SQLite repository. Hosted authentication is a later boundary.

Frozen by ADR 003 (2026-10-17): recovery and legacy exports only. The canonical
learner store is the JS core (hosted/migrations, hosted/local.mjs).
"""
from contextlib import closing, contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3

from .events import Event, active_events, identifier, timer_state


class EventStore:
    def __init__(self, path):
        self.connection = sqlite3.connect(path, isolation_level=None, timeout=10)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute('PRAGMA foreign_keys = ON')
        try:
            self._migrate()
        except Exception:
            self.close()
            raise

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def _transaction(self):
        self.connection.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.connection.execute('COMMIT')
        except BaseException:
            self.connection.execute('ROLLBACK')
            raise

    def _migrate(self):
        migration = Path(__file__).with_name('migrations') / '001_events.sql'
        raw = migration.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        with self._transaction():
            self.connection.execute('CREATE TABLE IF NOT EXISTS schema_migrations '
                                    '(version INTEGER PRIMARY KEY, sha256 TEXT NOT NULL)')
            versions = self.connection.execute('SELECT version, sha256 FROM schema_migrations').fetchall()
            if versions:
                if [(r['version'], r['sha256']) for r in versions] != [(1, digest)]:
                    raise ValueError('unknown migration version or migration checksum drift')
            else:
                # executescript commits implicitly; execute complete SQL statements instead.
                statement = ''
                for line in raw.decode('utf-8').splitlines(True):
                    statement += line
                    if sqlite3.complete_statement(statement):
                        self.connection.execute(statement)
                        statement = ''
                if statement.strip():
                    raise ValueError('incomplete migration SQL')
                self.connection.execute('INSERT INTO schema_migrations VALUES (1, ?)', (digest,))
            # Check live tables/indexes/triggers too, not only the migration receipt.
            with closing(sqlite3.connect(':memory:')) as expected:
                expected.executescript(raw.decode('utf-8'))
                query = ("SELECT type, name, tbl_name, sql FROM sqlite_master "
                         "WHERE tbl_name IN ('learners', 'events') ORDER BY type, name")
                wanted = expected.execute(query).fetchall()
            actual = [tuple(row) for row in self.connection.execute(query).fetchall()]
            if actual != wanted:
                raise ValueError('live event schema drift')

    def create_learner(self, learner_id):
        identifier(learner_id)
        with self._transaction():
            self.connection.execute('INSERT OR IGNORE INTO learners VALUES (?)', (learner_id,))

    def events(self, learner_id, *, active=False):
        identifier(learner_id)
        rows = self.connection.execute('SELECT * FROM events WHERE learner_id = ? ORDER BY sequence',
                                       (learner_id,)).fetchall()
        result = [self._event(row) for row in rows]
        return active_events(result) if active else result

    @staticmethod
    def _event(row):
        return Event(row['event_id'], row['learner_id'], row['occurred_at'],
                     row['recorded_at'], row['kind'], row['source'],
                     json.loads(row['payload']), row['supersedes'], row['schema_version'])

    def _append(self, event, learner_id):
        e = event.normalized()
        if e.learner_id != learner_id:
            raise ValueError('learner ownership mismatch')
        row = self.connection.execute('SELECT * FROM events WHERE event_id = ?', (e.event_id,)).fetchone()
        if row:
            if self._event(row) != e:
                raise ValueError('conflicting event ID')
            return False
        if e.supersedes:
            row = self.connection.execute('SELECT * FROM events WHERE event_id = ?', (e.supersedes,)).fetchone()
            if row is None:
                raise ValueError('correction target missing')
            old = self._event(row)
            if old.learner_id != learner_id or old.kind != e.kind or old.recorded_at > e.recorded_at:
                raise ValueError('invalid correction ownership/type/time')
            if old.payload.get('interval_id') != e.payload.get('interval_id'):
                raise ValueError('correction changes interval identity')
            if self.connection.execute('SELECT 1 FROM events WHERE supersedes = ?', (e.supersedes,)).fetchone():
                raise ValueError('correction target already superseded')
        self.connection.execute(
            'INSERT INTO events(event_id, learner_id, schema_version, occurred_at, recorded_at, '
            'kind, source, payload, supersedes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (e.event_id, e.learner_id, e.schema_version, e.occurred_at, e.recorded_at,
             e.kind, e.source, json.dumps(e.payload, sort_keys=True), e.supersedes))
        return True

    def append(self, learner_id, event):
        identifier(learner_id)
        with self._transaction():
            inserted = self._append(event, learner_id)
            timer_state(self.events(learner_id))
            return inserted

    def export_document(self, learner_id):
        identifier(learner_id)
        if self.connection.execute('SELECT 1 FROM learners WHERE learner_id = ?', (learner_id,)).fetchone() is None:
            raise ValueError('unknown learner')
        return {'format': 'adaptive-learning-events', 'version': 1, 'learner_id': learner_id,
                'events': [e.document() for e in self.events(learner_id)]}

    def import_document(self, learner_id, document):
        identifier(learner_id)
        if not isinstance(document, dict) or set(document) != {'format', 'version', 'learner_id', 'events'}:
            raise ValueError('invalid export envelope')
        if (document['format'] != 'adaptive-learning-events' or type(document['version']) is not int
                or document['version'] != 1 or document['learner_id'] != learner_id
                or not isinstance(document['events'], list)):
            raise ValueError('unsupported export version or ownership')
        # Validate all records before beginning mutation.
        records = [Event(**item).normalized() for item in document['events']]
        with self._transaction():
            self.connection.execute('INSERT OR IGNORE INTO learners VALUES (?)', (learner_id,))
            added = sum(self._append(e, learner_id) for e in records)
            timer_state(self.events(learner_id))
            return added
