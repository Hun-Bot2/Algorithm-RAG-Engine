"""Validate and append an explicit AI-only review batch; never accepts human roles."""
import argparse,json
from pathlib import Path
from core import Workbench,canonical,sha

def import_ai(wb,records):
    if len(records)!=len(wb.pairs) or {r.get('pair_id') for r in records}!=set(wb.pairs):
        raise ValueError('exactly one AI relationship draft per pilot pair required')
    for record in records:
        if record.get('role')!='ai' or record.get('kind')!='relationship' or record.get('independent') is not False:
            raise ValueError('AI non-independent relationship drafts only')
        wb.validate(record,wb.records())
    # Idempotent per record; interruption can be safely resumed with same immutable batch.
    results=[wb.append(record) for record in records]
    return {'records':len(records),'new_records':sum(not r['duplicate'] for r in results),
            'duplicate_records':sum(r['duplicate'] for r in results),'audit':wb.audit(),
            'accepted_overlay_count':len(wb.accepted_overlay(wb.records())),'batch_sha256':sha(canonical(records).encode())}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);p.add_argument('--db',type=Path,required=True);p.add_argument('--reviews',type=Path,required=True)
    a=p.parse_args();print(json.dumps(import_ai(Workbench(a.pilot,a.db),json.loads(a.reviews.read_text())),indent=2))
