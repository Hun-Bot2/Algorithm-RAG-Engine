"""Optional OCI stdlib diagnostic worker. No serving, networking or model fitting."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

FORMAT = 'oci-offline-batch'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.learning.curriculum import validate_curriculum


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(repo):
    registry = json.loads((repo/'assets/curriculum_release.v1.json').read_text())
    expected = {'curriculum/seed.v1.json', 'curriculum/annotations.v1.json'}
    if registry.get('schema_version') != 1 or set(registry.get('files', {})) != expected:
        raise ValueError('invalid curated release registry')
    hashes = {}
    bundle_manifest=repo/'bundle-manifest.json'
    if bundle_manifest.is_file():
        bundle=json.loads(bundle_manifest.read_text())
        if bundle.get('format')!='oci-stdlib-bundle' or bundle.get('schema_version')!=1:
            raise ValueError('invalid bundle manifest')
        for relative,checksum in bundle.get('files_sha256',{}).items():
            target=(repo/relative).resolve()
            if not target.is_relative_to(repo) or digest(target)!=checksum:
                raise ValueError('bundle source hash mismatch')
        hashes['bundle-manifest.json']=digest(bundle_manifest)
    for relative, checksum in registry['files'].items():
        target = (repo/relative).resolve()
        if not target.is_relative_to(repo) or digest(target) != checksum:
            raise ValueError('curated input hash mismatch')
        hashes[relative] = checksum
    seed = json.loads((repo/'curriculum/seed.v1.json').read_text())
    validate_curriculum(seed, require_release=True)
    if len(seed['problems']) > 500:
        raise ValueError('corpus exceeds small-VM profile')
    split = json.loads((repo/'research/splits/evaluation.v1.json').read_text())
    label_path = repo/'research/relationship-judgments.v1.json'
    if split.get('frozen') is not True or split.get('training_problem_keys') != [] or digest(label_path) != split.get('reserved_judgments_sha256'):
        raise ValueError('evaluation snapshot drift or nonempty training split')
    if set(split.get('evaluation_problem_keys', [])) != {p['problem_key'] for p in seed['problems']}:
        raise ValueError('evaluation membership differs from curriculum')
    labels = json.loads(label_path.read_text())
    if labels.get('permitted_uses') != ['evaluation'] or len(labels.get('judgments', [])) > 2000:
        raise ValueError('evaluation-only bounded labels required')
    for relative in ['research/splits/evaluation.v1.json', 'research/relationship-judgments.v1.json', 'research/protocol.v1.json']:
        hashes[relative] = digest(repo/relative)
    return hashes


def cgroup_observation():
    """Read the current process' unified cgroup when available; never infer it."""
    base = Path('/sys/fs/cgroup').resolve()
    try:
        memberships = Path('/proc/self/cgroup').read_text().splitlines()
        relative = next(line[3:] for line in memberships if line.startswith('0::'))
        path = (base/relative.lstrip('/')).resolve()
        if not path.is_relative_to(base):
            return None
        values = {}
        for name in ['memory.max', 'memory.peak', 'memory.current', 'memory.swap.max', 'cpu.max', 'cpu.stat']:
            target = path/name
            if target.is_file():
                values[name] = target.read_text()[:4096].strip()
        return values or None
    except (OSError, StopIteration):
        return None


def write_receipt(path, value):
    with path.open('x', encoding='utf-8') as target:
        json.dump(value, target, indent=2)
        target.write('\n')
    path.chmod(0o600)


def prune_owned_runs(runs, keep):
    owned = []
    for path in runs.iterdir():
        if path.is_symlink() or not path.is_dir() or not path.name.startswith('run-'):
            continue
        receipt = path/'receipt.json'
        if receipt.is_symlink():
            continue
        try:
            value = json.loads(receipt.read_text())
            if value.get('format') == FORMAT and value.get('schema_version') == 1 and value.get('run_id') == path.name:
                owned.append((value['started_at'], path))
        except (OSError, ValueError, KeyError):
            continue
    for _, path in sorted(owned, key=lambda item: (item[0], item[1].name))[:-keep]:
        shutil.rmtree(path)


def run(repo, state_dir, *, max_runs=14):
    repo = Path(repo).resolve()
    state = Path(state_dir).resolve()
    if type(max_runs) is not int or not 1 <= max_runs <= 100:
        raise ValueError('max-runs must be between 1 and 100')
    if state == repo or repo.is_relative_to(state):
        raise ValueError('state directory cannot contain the source repository')
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    state.chmod(0o700)
    descriptor = os.open(state/'.lock', os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(descriptor, 'r+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'already_running', 'exit_code': 75}
        runs = state/'runs'
        if runs.is_symlink():
            raise ValueError('runs directory must not be a symlink')
        runs.mkdir(exist_ok=True, mode=0o700)
        directory = Path(tempfile.mkdtemp(prefix='run-', dir=runs))
        started = time.monotonic()
        receipt = {'format': FORMAT, 'schema_version': 1, 'run_id': directory.name,
                   'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'failed',
                   'profile': 'stdlib-diagnostic-v1', 'network_required': False,
                   'model_created': False, 'production_promoted': False,
                   'configured_limits_boundary': 'Service caps require real systemd/cgroup verification; direct CLI runs do not apply those caps.',
                   'cgroup_before': cgroup_observation()}
        exit_code = 1
        try:
            receipt['inputs_sha256'] = verify_inputs(repo)
            command = [sys.executable, str(repo/'scripts/learning/benchmark.py'), '--curriculum', str(repo/'curriculum/seed.v1.json'),
                       '--annotations', str(repo/'curriculum/annotations.v1.json'), '--judgments', str(repo/'research/relationship-judgments.v1.json'),
                       '--output', str(directory/'benchmark.json')]
            with (directory/'stdout.txt').open('xb') as stdout, (directory/'stderr.txt').open('xb') as stderr:
                result = subprocess.run(command, cwd=repo, stdout=stdout, stderr=stderr, timeout=120,
                    env={'PATH': os.defpath, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'})
            exit_code = result.returncode
            if exit_code != 0:
                raise ValueError('benchmark subprocess failed; inspect private stderr')
            report = json.loads((directory/'benchmark.json').read_text())
            receipt.update(status='completed', benchmark_sha256=digest(directory/'benchmark.json'),
                           diagnostic_only=report.get('diagnostic_only'), label_admission=report.get('label_admission'),
                           ranking_query_groups=report.get('scoring', {}).get('ranking_query_groups'))
        except subprocess.TimeoutExpired:
            receipt['error'] = 'benchmark timeout'; exit_code = 124
        except (OSError, ValueError) as error:
            receipt['error'] = str(error); exit_code = 1
        finally:
            receipt.update(elapsed_wall_seconds=round(time.monotonic()-started, 6), exit_code=exit_code,
                           cgroup_after=cgroup_observation(), child_peak_rss_raw=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                           child_peak_rss_unit='bytes' if sys.platform == 'darwin' else 'kibibytes')
            for path in directory.iterdir():
                if path.is_file():
                    path.chmod(0o600)
            write_receipt(directory/'receipt.json', receipt)
            prune_owned_runs(runs, max_runs)
        return {'status': receipt['status'], 'run_directory': str(directory), 'exit_code': exit_code}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--state-dir', required=True, type=Path)
    parser.add_argument('--max-runs', default=14, type=int)
    args = parser.parse_args()
    try:
        result = run(args.repo, args.state_dir, max_runs=args.max_runs)
        print(json.dumps(result)); sys.exit(result['exit_code'])
    except (OSError, ValueError) as error:
        sys.exit(str(error))
