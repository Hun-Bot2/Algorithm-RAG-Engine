"""Read one native problem's separate evidence channels from a frozen map."""
import argparse
import json
from pathlib import Path


def query(root,key,k=10):
    root=Path(root)
    node=next((r for r in map(json.loads,(root/'nodes.jsonl').read_text().splitlines()) if r['problem_key']==key),None)
    if node is None:raise KeyError(key)
    channels={}
    for name in ('semantic','matched_semantic','tag','hybrid'):
        path=root/name/'top50.jsonl'
        if not path.exists():continue
        with path.open() as handle:
            found=next((r for r in map(json.loads,handle) if r['query']==key),None)
        channels[name]=found['results'][:k] if found else None
    return {'node':node,'neighbors':channels,'interpretation':'semantic title similarity and tag proxies are separate; unavailable channels are null'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('key');p.add_argument('--root',type=Path,default=Path('private-learning/unified-map/map-v2'));p.add_argument('--k',type=int,choices=(10,20,50),default=10);a=p.parse_args();print(json.dumps(query(a.root,a.key,a.k),indent=2))
