"""Link native nodes and every admitted statement identity without merging unknowns."""
import json
from pathlib import Path
from research.unified_map.build import ROOT,sha256,DEFAULT_STATEMENT_CORPUS,DEFAULT_APPROVAL,DEFAULT_APPROVAL_RECEIPT,load_statement_aliases


def run():
    load_statement_aliases(DEFAULT_STATEMENT_CORPUS,DEFAULT_APPROVAL,DEFAULT_APPROVAL_RECEIPT)
    target=ROOT/'private-learning/unified-map/package-v1';target.mkdir(parents=True,exist_ok=False)
    native=ROOT/'private-learning/unified-map/map-v2'
    nodes={r['problem_key'] for r in map(json.loads,(native/'nodes.jsonl').read_text().splitlines())}
    mappings=json.loads(DEFAULT_APPROVAL.read_text())['mappings']
    source=[]
    for r in map(json.loads,DEFAULT_STATEMENT_CORPUS.read_text().splitlines()):
        key=r.get('dataset_key') or r['problem_key'];approved=mappings.get(key)
        nk=approved['native_key'] if approved else r.get('native_key')
        source.append({'source_key':key,'retrieval_key':r['problem_key'],'platform':r['platform'],'content_sha256':r['content_sha256'],'native_key':nk,'native_catalog_member':nk in nodes if nk else False,'identity_basis':'user_approved' if approved else 'publisher_native_id' if nk else 'unresolved','quality_flags':approved['quality_flags'] if approved else r.get('quality_flags',[])})
    p=target/'statement-source-index.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in source));p.chmod(0o600)
    refs={}
    for name,path in {'native_map':native/'manifest.json','native_evaluation':native/'evaluation-receipt.json','reduction':ROOT/'private-learning/unified-map/reduction-v2/receipt.json','statement_evaluation':ROOT/'private-learning/unified-map/statement-evaluation-v1/receipt.json'}.items():refs[name]={'path':str(path.relative_to(ROOT)),'sha256':sha256(path)}
    receipt={'schema_version':1,'native_catalog_nodes':len(nodes),'statement_source_nodes':len(source),'statement_rows_linked_to_native_catalog':sum(r['native_catalog_member'] for r in source),'unresolved_statement_native_identities':sum(r['native_key'] is None for r in source),'identified_statements_outside_native_catalog':sum(bool(r['native_key']) and not r['native_catalog_member'] for r in source),'source_corpus_sha256':sha256(DEFAULT_STATEMENT_CORPUS),'approval_sha256':sha256(DEFAULT_APPROVAL),'references':refs,'statement_source_index_sha256':sha256(p),'boundary':'Native and publisher identities remain separate; unlinked rows are not deleted or guessed.'}
    (target/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':run()
