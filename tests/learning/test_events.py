from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from src.learning.events import Event, confirmed_seconds, timer_state
from src.learning.store import EventStore


def time(minute=0):
    return f'2026-10-16T01:{minute:02d}:00Z'


def attempt(key='e1', learner='alice', **changes):
    e = Event(key, learner, time(), time(30), 'attempt', 'manual',
              {'problem_key': 'leetcode:two-sum', 'outcome': 'independent', 'hints': False})
    return replace(e, **changes)


def timer(key, kind, minute, interval='i1', **payload):
    return Event(key, 'alice', time(minute), time(40), kind, 'manual',
                 {'interval_id': interval, **payload})


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'history.sqlite3'
        self.store = EventStore(self.path)
        self.store.create_learner('alice')
        self.store.create_learner('bob')

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_idempotency_and_conflict(self):
        self.assertTrue(self.store.append('alice', attempt()))
        self.assertFalse(self.store.append('alice', attempt()))
        with self.assertRaises(ValueError):
            self.store.append('alice', attempt(occurred_at=time(1)))
        self.assertEqual(len(self.store.events('alice')), 1)

    def test_ownership_in_reads_writes_and_restore(self):
        self.store.append('alice', attempt())
        self.assertEqual(self.store.events('bob'), [])
        with self.assertRaises(ValueError):
            self.store.append('bob', attempt('other'))
        with self.assertRaises(ValueError):
            self.store.import_document('bob', self.store.export_document('alice'))
        with self.assertRaises(ValueError):
            self.store.append('bob', attempt('bad', learner='bob', supersedes='e1'))

    def test_corrections_preserve_original_and_prevent_branches(self):
        self.store.append('alice', attempt())
        corrected = attempt('e2', supersedes='e1', payload={
            'problem_key': 'leetcode:two-sum', 'outcome': 'assisted', 'hints': True})
        self.store.append('alice', corrected)
        self.assertEqual(len(self.store.events('alice')), 2)
        self.assertEqual([e.event_id for e in self.store.events('alice', active=True)], ['e2'])
        with self.assertRaises(ValueError):
            self.store.append('alice', attempt('e3', supersedes='e1'))
        with self.assertRaises(ValueError):
            self.store.append('alice', attempt('e4', supersedes='missing'))

    def test_database_guards(self):
        self.store.append('alice', attempt())
        for sql in ['DELETE FROM events', "UPDATE events SET kind = 'attempt'"]:
            with self.assertRaises(sqlite3.IntegrityError):
                self.store.connection.execute(sql)
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.append('unknown', attempt('x', learner='unknown'))

    def test_invalid_events_leave_no_rows(self):
        invalid = [attempt(schema_version=2), attempt(schema_version=True),
                   attempt(occurred_at='2026-10-16T01:00:00'),
                   attempt(occurred_at=time(31)), attempt(source='llm'),
                   attempt(payload={'problem_key': 'two-sum', 'outcome': 'independent', 'hints': False}),
                   attempt(payload={'problem_key': 'leetcode:two-sum', 'outcome': 'independent', 'hints': None}),
                   attempt(source='platform_import')]
        for e in invalid:
            with self.subTest(e=e), self.assertRaises(ValueError):
                self.store.append('alice', e)
        self.assertEqual(self.store.events('alice'), [])

    def test_optional_activity_context_is_explicit_and_validated(self):
        e = attempt(payload={'problem_key': 'leetcode:two-sum', 'outcome': 'independent',
                             'hints': False, 'activity_type': 'review'})
        self.store.append('alice', e)
        self.assertEqual(self.store.events('alice')[0].payload['activity_type'], 'review')
        for payload in [dict(e.payload, activity_type='transfer'),
                        dict(e.payload, activity_type='transfer', transfer_from='leetcode:two-sum'),
                        dict(e.payload, activity_type=[]), dict(e.payload, transfer_from='atcoder:abc001_a')]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.store.append('alice', attempt('bad', payload=payload))
        transfer = dict(e.payload, activity_type='transfer', transfer_from='atcoder:abc001_a')
        self.store.append('alice', attempt('transfer', payload=transfer))

    def test_import_is_reported_not_mastery(self):
        e = attempt(source='platform_import', payload={
            'problem_key': 'leetcode:two-sum', 'outcome': 'reported_completion',
            'hints': None, 'platform': 'leetcode', 'external_id': 'submission-1'})
        self.store.append('alice', e)
        self.assertEqual(self.store.events('alice')[0].payload['outcome'], 'reported_completion')

    def test_timer_recovery_confirmation_and_overlap(self):
        self.store.append('alice', timer('s1', 'timer_started', 0, problem_key='leetcode:two-sum'))
        self.assertEqual(timer_state(self.store.events('alice'))[0]['status'], 'open')
        self.assertIsNone(timer_state(self.store.events('alice'))[0]['confirmed_seconds'])
        self.assertEqual(confirmed_seconds(self.store.events('alice')), 0)
        self.store.append('alice', timer('t1', 'timer_stopped', 20))
        self.assertEqual(timer_state(self.store.events('alice'))[0]['status'], 'unconfirmed')
        self.store.append('alice', timer('c1', 'timer_confirmed', 25, started_at=time(), ended_at=time(20)))
        self.store.append('alice', timer('s2', 'timer_started', 10, 'i2', problem_key='atcoder:abc001_a'))
        self.store.append('alice', timer('t2', 'timer_stopped', 30, 'i2'))
        self.store.append('alice', timer('c2', 'timer_confirmed', 31, 'i2', started_at=time(10), ended_at=time(30)))
        self.assertEqual(confirmed_seconds(self.store.events('alice')), 1800)
        document = self.store.export_document('alice')
        with EventStore(Path(self.tmp.name) / 'timers-restored.sqlite3') as restored:
            restored.import_document('alice', document)
            self.assertEqual(timer_state(restored.events('alice')), timer_state(self.store.events('alice')))
            self.assertEqual(confirmed_seconds(restored.events('alice')), 1800)

    def test_invalid_timer_order_and_correction_rollback(self):
        with self.assertRaises(ValueError):
            self.store.append('alice', timer('x', 'timer_stopped', 20))
        self.store.append('alice', timer('s', 'timer_started', 0, problem_key='leetcode:two-sum'))
        with self.assertRaises(ValueError):
            self.store.append('alice', timer('c', 'timer_confirmed', 10, started_at=time(), ended_at=time(5)))
        self.store.append('alice', timer('t', 'timer_stopped', 20))
        self.store.append('alice', timer('c', 'timer_confirmed', 21, started_at=time(), ended_at=time(20)))
        with self.assertRaises(ValueError):
            self.store.append('alice', replace(timer('bad', 'timer_stopped', 10), supersedes='t'))
        self.assertEqual(len(self.store.events('alice')), 3)

    def test_export_restore_replay_and_repeat_import(self):
        self.store.append('alice', attempt())
        self.store.append('alice', attempt('e2', supersedes='e1'))
        document = json.loads(json.dumps(self.store.export_document('alice')))
        with EventStore(Path(self.tmp.name) / 'restored.sqlite3') as restored:
            self.assertEqual(restored.import_document('alice', document), 2)
            self.assertEqual(restored.import_document('alice', document), 0)
            self.assertEqual(restored.export_document('alice'), document)
            self.assertEqual(restored.events('alice', active=True), self.store.events('alice', active=True))

    def test_import_conflict_rolls_back_entire_batch_and_learner(self):
        self.store.append('alice', attempt())
        doc = {'format': 'adaptive-learning-events', 'version': 1, 'learner_id': 'bob',
               'events': [attempt('new', learner='bob').document(), attempt(learner='bob').document()]}
        with self.assertRaises(ValueError):
            self.store.import_document('bob', doc)
        self.assertEqual(self.store.events('bob'), [])
        doc['learner_id'] = 'charlie'
        doc['events'] = [attempt('bad', learner='charlie', supersedes='absent').document()]
        with self.assertRaises(ValueError):
            self.store.import_document('charlie', doc)
        with self.assertRaises(ValueError):
            self.store.export_document('charlie')

    def test_migration_reopen_and_drift_rejection(self):
        with EventStore(self.path) as other:
            self.assertEqual(other.connection.execute('SELECT COUNT(*) FROM schema_migrations').fetchone()[0], 1)
        self.store.connection.execute("UPDATE schema_migrations SET sha256='drift'")
        with self.assertRaises(ValueError):
            EventStore(self.path)

    def test_live_schema_drift_rejected(self):
        self.store.connection.execute('DROP TRIGGER events_no_update')
        with self.assertRaises(ValueError):
            EventStore(self.path)

    def test_concurrent_duplicate_writers(self):
        def write(_):
            with EventStore(self.path) as store:
                return store.append('alice', attempt())
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(write, range(8)))
        self.assertEqual(sum(results), 1)
        self.assertEqual(len(self.store.events('alice')), 1)


if __name__ == '__main__':
    unittest.main()
