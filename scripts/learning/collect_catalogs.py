"""Collect private Codeforces and AtCoder metadata; network access is opt-in."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.learning.catalog_sources import collect


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "private-learning" / "catalogs" / "run-001")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fetch", action="store_true", help="fetch both public metadata endpoints")
    mode.add_argument("--replay", type=Path, help="replay raw response bytes from a verified run directory")
    args = parser.parse_args(argv)
    manifest = collect(args.output, fetch=args.fetch, replay=args.replay)
    for source in ("codeforces", "atcoder"):
        receipt = manifest[source]
        suffix = f" error={receipt['error']}" if receipt["error"] else ""
        print(f"{source}: {receipt['status']} bytes={receipt['byte_count']} rows={receipt['normalized_count']} sha256={receipt['sha256']}{suffix}")
    print(f"Private run written to {args.output}")
    if any(manifest[source]["status"] != "ok" for source in manifest if source in ("codeforces", "atcoder")):
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, TypeError) as exc:
        sys.exit(str(exc))
