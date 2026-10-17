"""Top10 sparse-label and tag diagnostics for stored reduction neighborhoods."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'research/algorithm_system'))
from research.algorithm_system.evaluate import _bootstrap
from research.unified_map.build import _concepts,sha256
from research.unified_map.reduction import sampled_2d_neighbors
from research.full_catalog.retrieve import Rows


def metrics(keys,neighbors,nodes,labels):
    features={k:_concepts(nodes[k]['normalized_taxonomy']) for k in keys}
    slots=judged=positive=0;tagvals=[];perquery={}
    for i,targets in enumerate(neighbors):
        q=keys[i];grades=[]
        for j in targets:
            if j<0:continue
            t=keys[int(j)];slots+=1;label=labels.get((q,t))
            if label is not None:judged+=1;positive+=label>=2;grades.append(label>=2)
            a,b=features[q],features[t]
            if a and b:tagvals.append(len(a&b)/len(a|b))
        if grades:perquery[q]=sum(grades)/len(grades)
    return {'queries':len(keys),'slots':slots,'judged_slots':judged,'judged_coverage':judged/slots,'observed_precision_judged':positive/judged if judged else None,'query_mean_precision_bootstrap':_bootstrap(perquery,resamples=200),'tag_eligible_slots':len(tagvals),'mean_tag_jaccard':sum(tagvals)/len(tagvals) if tagvals else None}


def run():
    red=ROOT/'private-learning/unified-map/reduction-v2';receipt=json.loads((red/'receipt.json').read_text())
    for name,v in receipt['outputs'].items():
        if sha256(red/name)!=v['sha256']:raise ValueError('reduction_output_hash_mismatch')
    nodes={r['problem_key']:r for r in map(json.loads,(ROOT/'private-learning/unified-map/map-v2/nodes.jsonl').read_text().splitlines())}
    labels={(r['query'],r['target']):r['label'] for r in map(json.loads,(ROOT/'research/full_catalog/labels.v1.jsonl').read_text().splitlines())}
    keys=json.loads((ROOT/'private-learning/full-catalog/embeddings-v1/A1/keys.json').read_text())
    reports={p.name.split('.')[0]:metrics(keys,np.load(p),nodes,labels) for name in receipt['outputs'] if name.endswith('.top10.npy') for p in [red/name]}
    samplekeys=json.loads((red/'sample-keys.json').read_text());rows=Rows.from_catalog(list(nodes.values()),samplekeys)
    for name in receipt['sampled_2d']:
        neighbors=sampled_2d_neighbors(np.load(red/f'sample-{name}.npy'),rows)
        reports['sample_'+name]=metrics(samplekeys,neighbors,nodes,labels)
    out=red/'quality-diagnostics.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps({'reduction_receipt_sha256':sha256(red/'receipt.json'),'label_basis':'historical_AI_semantic_not_algorithm_gold','methods':reports,'comparison_boundary':'allrow linear methods share21297; sampled2D methods share3000; never compare across cohorts'},indent=2)+'\n')

if __name__=='__main__':run()
