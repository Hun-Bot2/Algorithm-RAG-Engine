import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]


class HistoryCliTest(unittest.TestCase):
    def test_no_credentials_roundtrip_and_private_permissions(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            db, backup, restored = path / 'history.sqlite3', path / 'backup.json', path / 'restored.sqlite3'
            event = path / 'event.json'
            event.write_text(json.dumps(dict(event_id='cli-1', learner_id='me',
                occurred_at='2026-10-16T01:00:00Z', recorded_at='2026-10-16T01:02:00Z',
                kind='attempt', source='manual', payload=dict(
                    problem_key='leetcode:two-sum', outcome='assisted', hints=True))))
            # No application credentials, provider URLs, venv or model path supplied.
            def run(command, database=db, file=None):
                args = [sys.executable, str(ROOT / 'scripts/learning/history.py'),
                        '--db', str(database), '--learner', 'me', command]
                if file:
                    args += ['--file', str(file)]
                return subprocess.run(args, cwd=ROOT, env={'PATH': os.defpath},
                                      capture_output=True, text=True, check=False)
            self.assertEqual(run('init').returncode, 0)
            self.assertEqual(run('append', file=event).returncode, 0)
            before = run('history')
            self.assertEqual(before.returncode, 0)
            self.assertEqual(json.loads(before.stdout)['confirmed_seconds'], 0)
            self.assertEqual(run('export', file=backup).returncode, 0)
            self.assertNotEqual(run('export', file=backup).returncode, 0)
            self.assertEqual(db.stat().st_mode & 0o777, 0o600)
            self.assertEqual(backup.stat().st_mode & 0o777, 0o600)
            self.assertEqual(run('import', restored, backup).returncode, 0)
            self.assertEqual(run('history', restored).stdout, before.stdout)
