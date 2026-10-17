"""Re-evaluate frozen statement ranks with approved identity metadata only."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'research/algorithm_system'))
from research.algorithm_system.evaluate import evaluate
from research.unified_map.build import load_statement_aliases,DEFAULT_APPROVAL,DEFAULT_APPROVAL_RECEIPT,DEFAULT_STATEMENT_CORPUS,sha256


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    aliases,_=load_statement_aliases(DEFAULT_STATEMENT_CORPUS,DEFAULT_APPROVAL,DEFAULT_APPROVAL_RECEIPT)
    approval=json.loads(DEFAULT_APPROVAL.read_text())['mappings']
    source=[json.loads(x) for x in DEFAULT_STATEMENT_CORPUS.read_text().splitlines()]
    metadata=[]
    for r in source:
        key=r.get('dataset_key') or r['problem_key']
        mapping=approval.get(key)
        metadata.append({k:v for k,v in r.items() if k not in {'statement_text','title','description'}} | ({'native_key':mapping['native_key'],'quality_flags':mapping['quality_flags']} if mapping else {}))
    base=DEFAULT_STATEMENT_CORPUS.parent
    manifest=json.loads((base/'manifest.json').read_text())
    methods={}
    for name,info in manifest['methods'].items():
        if 'outputs' not in info:continue
        p=base/name/'top50.jsonl'
        if sha256(p)!=info['outputs']['top50.jsonl']:raise ValueError('statement_ranking_hash_mismatch')
        methods[name]=[json.loads(x) for x in p.read_text().splitlines()]
    labels=ROOT/'research/full_catalog/labels.v1.jsonl'
    report=evaluate(metadata,methods,[json.loads(x) for x in labels.read_text().splitlines()],resamples=200,review_limit=500,review_queries_per_platform=30)
    result=output/'evaluation.json';result.write_text(json.dumps(report,indent=2)+'\n')
    receipt={'source_corpus_sha256':sha256(DEFAULT_STATEMENT_CORPUS),'source_manifest_sha256':sha256(base/'manifest.json'),'approval_sha256':sha256(DEFAULT_APPROVAL),'labels_sha256':sha256(labels),'evaluation_sha256':sha256(result),'rows':len(metadata),'applied_identity_mappings':len(approval),'original_corpus_modified':False,'representation':'statement semantic, not algorithm equivalence'}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':run(ROOT/'private-learning/unified-map/statement-evaluation-v1')
