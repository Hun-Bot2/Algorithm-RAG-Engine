import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load_tool(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package_oci = load_tool('package_oci', 'scripts/learning/package_oci.py')
oci_batch = load_tool('oci_batch', 'scripts/learning/oci_batch.py')


class OciTest(unittest.TestCase):
    def test_bundle_is_deterministic_allowlisted_and_hash_verified(self):
        with TemporaryDirectory() as tmp:
            first, second = Path(tmp) / 'one.tar.gz', Path(tmp) / 'two.tar.gz'
            source = ROOT / package_oci.FILES[0]
            original_times = source.stat().st_atime_ns, source.stat().st_mtime_ns
            try:
                package_oci.build(first)
                os.utime(source, (1_700_000_000, 1_700_000_000))
                package_oci.build(second)
            finally:
                os.utime(source, ns=original_times)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with tarfile.open(first, 'r:gz') as archive:
                members = archive.getmembers()
                names = [member.name for member in members]
                self.assertEqual(set(names), set(package_oci.FILES) | {'bundle-manifest.json'})
                self.assertTrue(all(member.mtime == 0 and member.uid == 0 and member.gid == 0 for member in members))
                manifest = json.load(archive.extractfile('bundle-manifest.json'))
                self.assertEqual(set(manifest['files_sha256']), set(package_oci.FILES))
                for name, expected in manifest['files_sha256'].items():
                    self.assertEqual(hashlib.sha256(archive.extractfile(name).read()).hexdigest(), expected)

    def test_extracted_bundle_executes_offline_worker(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bundle = tmp / 'bundle.tar.gz'
            extracted = tmp / 'extracted'
            extracted.mkdir()
            package_oci.build(bundle)
            with tarfile.open(bundle, 'r:gz') as archive:
                archive.extractall(extracted, filter='data')
            result = subprocess.run(
                [sys.executable, str(extracted / 'scripts/learning/oci_batch.py'),
                 '--repo', str(extracted), '--state-dir', str(tmp / 'private-state')],
                cwd=extracted, env={'PATH': os.defpath, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'},
                capture_output=True, text=True, timeout=150, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            response = json.loads(result.stdout)
            self.assertEqual(response['status'], 'completed')
            receipt = json.loads((Path(response['run_directory']) / 'receipt.json').read_text())
            self.assertTrue(receipt['diagnostic_only'])
            self.assertFalse(receipt['model_created'])
            self.assertEqual(receipt['inputs_sha256']['bundle-manifest.json'],
                             hashlib.sha256((extracted / 'bundle-manifest.json').read_bytes()).hexdigest())

    def test_invalid_inputs_fail_before_benchmark_and_bundle_manifest_is_checked(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            extracted = tmp / 'repo'
            extracted.mkdir()
            bundle = tmp / 'bundle.tar.gz'
            package_oci.build(bundle)
            with tarfile.open(bundle, 'r:gz') as archive:
                archive.extractall(extracted, filter='data')
            seed = extracted / 'curriculum/seed.v1.json'
            seed.write_text(seed.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'bundle source hash mismatch'):
                oci_batch.verify_inputs(extracted)
            seed.write_text((ROOT / 'curriculum/seed.v1.json').read_text())
            manifest_path = extracted / 'bundle-manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['files_sha256']['../../outside'] = '0' * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'bundle source hash mismatch'):
                oci_batch.verify_inputs(extracted)

    def test_retention_preserves_unowned_and_symlink_runs(self):
        with TemporaryDirectory() as tmp:
            runs = Path(tmp)
            for index in range(4):
                name = f'run-{index}'
                directory = runs / name
                directory.mkdir()
                (directory / 'receipt.json').write_text(json.dumps({
                    'format': oci_batch.FORMAT, 'schema_version': 1,
                    'run_id': name, 'started_at': f'2026-10-16T00:0{index}:00+00:00'}))
            unknown = runs / 'run-unknown'
            unknown.mkdir()
            (unknown / 'receipt.json').write_text('{"format":"other"}')
            outside = Path(tmp) / 'outside'
            outside.mkdir()
            (outside / 'receipt.json').write_text('{}')
            (runs / 'run-linked').symlink_to(outside, target_is_directory=True)
            oci_batch.prune_owned_runs(runs, 2)
            self.assertFalse((runs / 'run-0').exists())
            self.assertFalse((runs / 'run-1').exists())
            self.assertTrue((runs / 'run-2').exists())
            self.assertTrue((runs / 'run-3').exists())
            self.assertTrue(unknown.exists())
            self.assertTrue((runs / 'run-linked').is_symlink())
            self.assertTrue((outside / 'receipt.json').exists())

    def test_invalid_retention_and_nonblocking_lock(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'max-runs'):
                oci_batch.run(ROOT, tmp / 'state', max_runs=0)
            state = tmp / 'state'
            state.mkdir(mode=0o700)
            descriptor = os.open(state / '.lock', os.O_CREAT | os.O_RDWR, 0o600)
            with os.fdopen(descriptor, 'r+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(oci_batch.run(ROOT, state), {'status': 'already_running', 'exit_code': 75})


if __name__ == '__main__':
    unittest.main()
