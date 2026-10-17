"""Compile manually inspected evidence; append AI proposals only when requested."""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.identity_resolution.build import validate_evidence,proposal_records,report_markdown
from research.verification_workbench.core import Workbench,canonical,sha

def run(evidence_path,pilot_path,db_path,output,*,append=False):
    evidence_path,pilot_path,output=map(Path,(evidence_path,pilot_path,output))
    if not output.resolve().is_relative_to((ROOT/'private-learning').resolve()):raise ValueError('private output directory required')
    if output.exists():raise FileExistsError('fresh output directory required')
    rows=json.loads(evidence_path.read_text());wb=Workbench(pilot_path,db_path)
    required=sorted({p[s] for p in wb.pilot['items'] for s in ('query','target') if p[s] in wb.corpus and wb.corpus[p[s]]['platform']=='atcoder' and not wb.corpus[p[s]].get('native_key')})
    validation=validate_evidence(rows,wb.corpus,wb.catalog,required)
    proposals=proposal_records(rows,wb)
    if any(r['role']!='ai' or r['independent'] or r['status']!='proposed' for r in proposals):raise ValueError('AI proposed non-independent records only')
    before=wb.records();before_by_id={r['id']:canonical(r) for r in before};before_audit=wb.audit()
    output.mkdir(parents=True)
    def write(name,obj):
        p=output/name;p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');p.chmod(0o600)
    write('evidence.json',rows);write('proposals.json',proposals)
    report=output/'REVIEW-FOR-USER.md';report.write_text(report_markdown(rows));report.chmod(0o600)
    results=[wb.append(r) for r in proposals] if append else []
    after=wb.records();after_by_id={r['id']:canonical(r) for r in after}
    assert all(after_by_id.get(k)==v for k,v in before_by_id.items())
    receipt={'schema_version':1,'evidence_sha256':sha(evidence_path.read_bytes()),'pilot_sha256':sha(pilot_path.read_bytes()),
        'source_corpus_sha256':wb.pilot['sources']['corpus']['sha256'],'source_catalog_sha256':wb.pilot['sources']['catalog']['sha256'],
        'validation':validation,'proposed_unique_identities':sum(r['status']=='ai_proposed' for r in rows),'proposal_records':len(proposals),
        'appended':append,'new_records':sum(not r['duplicate'] for r in results),'duplicate_records':sum(r['duplicate'] for r in results),
        'preserved_existing_records':len(before),'before_audit':before_audit,'after_audit':wb.audit(),
        'human_records_before':sum(r['role']=='human' for r in before),'human_records_after':sum(r['role']=='human' for r in after),
        'accepted_overlay_before':len(wb.accepted_overlay(before)),'accepted_overlay_after':len(wb.accepted_overlay(after)),
        'comparison_statuses':{k:dict(Counter(str(r['comparison'][k]) for r in rows)) for k in ('contest','semantics','constraints','samples')},
        'native_catalog_changed':False,'source_statements_changed':False,'official_full_statements_saved':False,
        'outputs_sha256':{n:sha((output/n).read_bytes()) for n in ('evidence.json','proposals.json','REVIEW-FOR-USER.md')},
        'limits':'Manual AI comparison notes, not authenticated human verification or statement reuse permission. No native IDs promoted.'}
    write('receipt.json',receipt);return receipt

if __name__=='__main__':
    default=ROOT/'private-learning/verification-workbench/pilot-v1'
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--pilot',type=Path,default=default/'pilot.json');p.add_argument('--db',type=Path,default=default/'reviews.sqlite3');p.add_argument('--output',type=Path,required=True);p.add_argument('--append',action='store_true')
    a=p.parse_args();result=run(a.evidence,a.pilot,a.db,a.output,append=a.append)
    print(json.dumps({k:result[k] for k in ('validation','proposal_records','appended','new_records','accepted_overlay_after')},indent=2))
