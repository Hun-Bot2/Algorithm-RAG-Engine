"""Local private event entry and JSON backup/restore; never performs network calls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.learning.events import Event, confirmed_seconds, timer_state
from src.learning.store import EventStore


def write_private(path, document):
    # Exclusive creation prevents accidental overwrite; mode protects private evidence.
    import os
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as target:
        json.dump(document, target, indent=2)
        target.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--learner', required=True)
    parser.add_argument('command', choices=['init', 'append', 'history', 'export', 'import'])
    parser.add_argument('--file', type=Path)
    args = parser.parse_args()
    if args.command in {'append', 'export', 'import'} and args.file is None:
        parser.error('this command requires --file')
    # Local DB parent must be provisioned privately by its owner.
    if not args.db.exists():
        import os
        descriptor = os.open(args.db, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
    with EventStore(args.db) as store:
        if args.command == 'init':
            store.create_learner(args.learner)
            print('Learner initialized.')
        elif args.command == 'append':
            event = Event(**json.loads(args.file.read_text(encoding='utf-8')))
            print('Inserted.' if store.append(args.learner, event) else 'Already recorded.')
        elif args.command == 'import':
            print(f"Imported {store.import_document(args.learner, json.loads(args.file.read_text(encoding='utf-8')))} events.")
        elif args.command == 'export':
            write_private(args.file, store.export_document(args.learner))
            print('Private export written.')
        else:
            events = store.events(args.learner)
            print(json.dumps({'events': [e.document() for e in events],
                              'timers': timer_state(events),
                              'confirmed_seconds': confirmed_seconds(events)}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, TypeError, OSError) as exc:
        sys.exit(str(exc))
