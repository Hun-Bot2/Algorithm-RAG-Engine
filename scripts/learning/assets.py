"""Verify frozen private assets or build a fail-closed, explicitly approved bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'assets/source_registry.v1.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def asset_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or Path(relative).is_absolute():
        raise ValueError('asset path escapes repository')
    return path


def verify(root=ROOT, registry=None):
    registry = registry if registry is not None else json.loads(REGISTRY.read_text())
    if registry['schema_version'] != 1:
        raise ValueError('unsupported registry')
    for entry in registry['assets']:
        path = asset_path(root, entry['path'])
        if entry['status'] == 'missing':
            if path.exists():
                raise ValueError(f"unreviewed new asset: {entry['path']}")
        elif not path.is_file() or sha256(path) != entry['sha256']:
            raise ValueError(f"missing or changed asset: {entry['path']}")
    return registry


def bundle(destination, root=ROOT, registry=None):
    registry = verify(root, registry)
    approved = [e for e in registry['assets'] if e.get('hosted_release') is True
                and e.get('rights_status') == 'approved' and e['status'] == 'present'
                and 'hosted_bundle' in e.get('permitted_uses', [])]
    destination = Path(destination).resolve()
    # A bundle must be new, outside source trees, and contain only approved paths.
    if destination.exists() or root.resolve().is_relative_to(destination):
        raise ValueError('bundle destination must be new and cannot contain source root')
    for entry in registry['assets']:
        source = asset_path(root, entry['path'])
        if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError('bundle destination overlaps registered source')
    destination.mkdir(parents=True)
    for entry in approved:
        source = asset_path(root, entry['path'])
        target = asset_path(destination, entry['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return len(approved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'bundle'])
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.command == 'verify':
        registry = verify()
        print(f"Verified {sum(e['status'] == 'present' for e in registry['assets'])} assets; "
              f"{sum(e['status'] == 'missing' for e in registry['assets'])} explicitly missing.")
    else:
        if args.destination is None:
            parser.error('bundle requires --destination')
        print(f'Bundled {bundle(args.destination)} approved assets.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))
