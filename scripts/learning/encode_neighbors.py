"""Optional batch indexing from hash-verified local models; no download or training."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.learning.embedding_backend import LocalEmbeddingBackend
from src.learning.research import build_profiles,read_json,provenance
from scripts.learning.benchmark import write_new_json

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-model',required=True,type=Path)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--curriculum',type=Path,default=ROOT/'curriculum/seed.v1.json')
    parser.add_argument('--annotations',type=Path,default=ROOT/'curriculum/annotations.v1.json')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    try:
        if args.output.exists():raise ValueError('output already exists')
        started=time.perf_counter()
        backend=LocalEmbeddingBackend(args.local_model,read_json(args.manifest))
        result=backend.neighbors(build_profiles(read_json(args.curriculum),read_json(args.annotations)))
        result['run_provenance']=provenance(input_paths=[args.manifest,args.curriculum,args.annotations],
            config={'operation':'offline_local_encoding','model_manifest':backend.manifest},elapsed_seconds=time.perf_counter()-started)
        write_new_json(args.output,result)
        print('Wrote optional neighbor artifact; no production promotion performed.')
    except (ValueError,OSError) as error:
        sys.exit(str(error))
