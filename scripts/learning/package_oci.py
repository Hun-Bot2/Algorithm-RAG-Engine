"""Build a deterministic, explicit-allowlist OCI offline bundle; no secrets or install actions."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
FILES=(
    'src/learning/__init__.py','src/learning/curriculum.py','src/learning/events.py',
    'src/learning/research.py','src/learning/research_contracts.py','src/learning/research_profiles.py',
    'src/learning/research_metrics.py','src/learning/research_training.py',
    'scripts/learning/benchmark.py','scripts/learning/oci_batch.py',
    'assets/curriculum_release.v1.json','curriculum/seed.v1.json','curriculum/annotations.v1.json',
    'research/relationship-judgments.v1.json','research/splits/evaluation.v1.json','research/protocol.v1.json',
    'deploy/oci/algorithm-batch.service','deploy/oci/algorithm-batch.timer','deploy/oci/algorithm-batch.env.example',
    'specs/adaptive-learning/oci.md',
)


def build(output,root=ROOT):
    root=Path(root).resolve()
    payloads={}
    for relative in FILES:
        source=root/relative
        if source.is_symlink() or not source.resolve().is_relative_to(root) or not source.is_file():
            raise ValueError('invalid/missing allowlisted source: '+relative)
        payloads[relative]=source.read_bytes()
    manifest={'format':'oci-stdlib-bundle','schema_version':1,
              'files_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()},
              'boundary':'Offline diagnostic only. No secrets, learner state, model weights, legacy data or deployed worker assets.'}
    payloads['bundle-manifest.json']=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    # No timestamps/owner metadata make identical source snapshots reproducible.
    with Path(output).open('xb') as destination:
        Path(output).chmod(0o600)
        with gzip.GzipFile(filename='',fileobj=destination,mode='wb',mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as archive:
                for name,data in sorted(payloads.items()):
                    info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
                    archive.addfile(info,io.BytesIO(data))
    return {'path':str(output),'files':len(payloads),'size_bytes':Path(output).stat().st_size,
            'sha256':hashlib.sha256(Path(output).read_bytes()).hexdigest()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    try: print(json.dumps(build(args.output)))
    except (ValueError,OSError) as error: sys.exit(str(error))
