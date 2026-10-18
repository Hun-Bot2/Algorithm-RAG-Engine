"""Evaluate independent cohorts without converting unknown relevance to negatives."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'research/algorithm_system'))
from research.algorithm_system.evaluate import evaluate as evaluate_rankings


def run(root):
    root=Path(root)
    manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(ROOT/'research/algorithm_system/taxonomy.v1.json')!=manifest['taxonomy_sha256']:raise ValueError('taxonomy_hash_mismatch')
    for name,expected in manifest['output_sha256'].items():
        if digest(root/name)!=expected:raise ValueError('map_output_hash_mismatch:'+name)
    required=['nodes.jsonl']+[name+'/top50.jsonl' for name in ('semantic','matched_semantic','tag','hybrid')]
    if any(name not in manifest['output_sha256'] for name in required):raise ValueError('required_map_outputs_missing')
    corpus=[json.loads(x) for x in (root/'nodes.jsonl').read_text().splitlines()]
    labels=[json.loads(x) for x in (ROOT/'research/full_catalog/labels.v1.jsonl').read_text().splitlines()]
    rankings={p.parent.name:[json.loads(x) for x in p.read_text().splitlines()] for name in ('semantic','matched_semantic','tag','hybrid') for p in [root/name/'top50.jsonl'] if p.exists()}
    groups={'all_semantic':{k:v for k,v in rankings.items() if k=='semantic'},
            'matched_tags':{k:v for k,v in rankings.items() if k!='semantic'}}
    matched=[{r['query'] for r in rankings[name]} for name in ('matched_semantic','tag','hybrid')]
    if any(q!=matched[0] for q in matched[1:]):raise ValueError('matched_query_sets_differ')
    reports={}
    for cohort,methods in groups.items():
        if methods:
            reports[cohort]=evaluate_rankings(corpus,methods,labels,resamples=200,review_limit=500,review_queries_per_platform=30)
    out=root/'evaluation.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(reports,indent=2)+'\n')
    (root/'evaluation-receipt.json').write_text(json.dumps({'map_manifest_sha256':digest(manifest_path),'labels_sha256':digest(ROOT/'research/full_catalog/labels.v1.jsonl'),'evaluation_sha256':digest(out),'query_bootstrap_resamples':200,'label_basis':'historical_ai_semantic_not_algorithm_gold','taxonomy_sha256':manifest['taxonomy_sha256'],'evaluator_sha256':digest(Path(__file__)),'shared_evaluator_sha256':digest(ROOT/'research/algorithm_system/evaluate.py')},indent=2)+'\n')
    return reports

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();run(a.root)
