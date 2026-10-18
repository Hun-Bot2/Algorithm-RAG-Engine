"""Verify final map boundary invariants and immutable earlier research files."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(output):
    output=Path(output)
    receipt=json.loads((output/'manifest.json').read_text())
    for name,expected in receipt['output_sha256'].items():
        assert sha(output/name)==expected, name
    nodes={r['problem_key']:r for r in map(json.loads,(output/'nodes.jsonl').read_text().splitlines())}
    assert len(nodes)==receipt['counts']['catalog_nodes']==24297
    slots={}
    for path in output.glob('*/top50.jsonl'):
        seen=set();count=0
        for r in map(json.loads,path.read_text().splitlines()):
            q=r['query'];assert q not in seen;seen.add(q)
            targets=[t[0] for t in r['results']]
            assert len(targets)==len(set(targets)) and q not in targets
            for t in targets:
                assert nodes[q]['platform']!=nodes[t]['platform']
                assert nodes[q]['representative_key']!=nodes[t]['representative_key']
                group=nodes[q]['duplicate_candidate_group']
                assert group is None or group!=nodes[t]['duplicate_candidate_group']
            count+=len(targets)
        slots[path.parent.name]={'queries':len(seen),'slots':count}
    special=nodes['atcoder:abc084_c']['statement_availability']['quality_flags']
    assert set(special)>={'official_en_ja_constraint_disagreement','sample_exceeds_english_bound'}
    assert not any('p03510' in str(r.get('statement_availability',{})) for r in nodes.values())
    frozen=json.loads((ROOT/'research/unified_map/PRESERVATION.v1.json').read_text())
    for p,v in frozen.items():assert sha(ROOT/p)==v['sha256'],p
    return {'nodes':len(nodes),'methods':slots,'preserved_input_files':len(frozen),'quality_flags':special,'all_checks_passed':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(verify(a.output),indent=2))
