"""Local evidence workbench. Self-reported roles are not authenticated identities."""
from __future__ import annotations
import hashlib,json,re,sqlite3
from contextlib import contextmanager
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CORPUS=ROOT/'private-learning/algorithm-system/statement-run-v1/corpus.jsonl'
DEFAULT_CATALOG=ROOT/'private-learning/full-catalog/catalog-v1/catalog.v1.jsonl'
DEFAULT_QUEUE=ROOT/'research/algorithm_system/evaluation.v1.json'
CHOICES={'prerequisite':{'query_before_target','target_before_query','neither','unknown'},
         'transfer':{'yes','no','unknown'},'difficulty':{'query_easier','similar','target_easier','unknown'}}

def canonical(obj):return json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def sha(data):return hashlib.sha256(data).hexdigest()
def load_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
def pair_id(q,t):return 'pair-'+sha(canonical([q,t]).encode())[:24]

def prepare(queue_path,corpus_path,catalog_path,output,size=40,seed=17):
    if type(size) is not int or not 1<=size<=1000:raise ValueError('pilot size must be 1..1000')
    paths={'queue':Path(queue_path),'corpus':Path(corpus_path),'catalog':Path(catalog_path)}
    queue=json.loads(paths['queue'].read_text())['review_queue']['items'];corpus={r['problem_key'] for r in load_rows(corpus_path)}
    strata=defaultdict(list);seen=set()
    for row in queue:
        q,t=row['query'],row['target'];pid=pair_id(q,t)
        if q not in corpus or t not in corpus or q==t:raise ValueError('invalid queue endpoint')
        if pid in seen:raise ValueError('duplicate queue pair')
        seen.add(pid);item={k:row[k] for k in ('query','target','methods','stratum')};item['pair_id']=pid
        key=canonical(row['stratum']);strata[key].append(item)
    for group in strata.values():group.sort(key=lambda r:sha(f'{seed}:{r["pair_id"]}'.encode()))
    items=[]
    while len(items)<size and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(items)<size:items.append(strata[key].pop(0))
    pilot={'schema_version':1,'seed':seed,'requested_pairs':size,'actual_pairs':len(items),
           'selection':'seeded hash order within strata; round robin across method/platform/difficulty/topic; no quality claim',
           'sources':{k:{'path':str(p.resolve()),'sha256':sha(p.read_bytes())} for k,p in paths.items()},
           'strata':dict(Counter(canonical(r['stratum']) for r in items)),'items':items}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:f.write(json.dumps(pilot,indent=2)+'\n')
    output.chmod(0o600);return pilot

def title_core(title):
    wrapped=bool(re.match(r'^p\d{5}\s+',title.strip(),flags=re.I))
    title=re.sub(r'^p\d{5}\s+','',title.strip(),flags=re.I)
    if wrapped:
        title=re.split(r'(?:^|\s)-\s+',title,maxsplit=1)[-1]
        title=re.sub(r'^\d+\s+','',title)
    return ' '.join(re.findall(r'\w+',title.casefold()))

class Workbench:
    def __init__(self,pilot_path,db_path):
        self.pilot_path=Path(pilot_path);self.pilot=json.loads(self.pilot_path.read_text());self.pilot_hash=sha(self.pilot_path.read_bytes())
        if self.pilot.get('schema_version')!=1:raise ValueError('unsupported pilot schema')
        for source in self.pilot['sources'].values():
            if sha(Path(source['path']).read_bytes())!=source['sha256']:raise ValueError('source drift')
        self.corpus={r['problem_key']:r for r in load_rows(self.pilot['sources']['corpus']['path'])}
        self.catalog={r['problem_key']:r for r in load_rows(self.pilot['sources']['catalog']['path'])}
        self.pairs={r['pair_id']:r for r in self.pilot['items']}
        if len(self.pairs)!=len(self.pilot['items']):raise ValueError('duplicate pilot pair')
        self.title_index=defaultdict(list)
        for r in self.catalog.values():self.title_index[(r['platform'],title_core(r['title']))].append(r)
        self.db_path=Path(db_path);self.db_path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as c:
            c.executescript('''CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS exposures(pair_id TEXT NOT NULL,reviewer TEXT NOT NULL,created_utc TEXT NOT NULL,PRIMARY KEY(pair_id,reviewer));
            CREATE TRIGGER IF NOT EXISTS no_exposure_update BEFORE UPDATE ON exposures BEGIN SELECT RAISE(ABORT,'append only'); END;
            CREATE TRIGGER IF NOT EXISTS no_exposure_delete BEFORE DELETE ON exposures BEGIN SELECT RAISE(ABORT,'append only'); END;
            CREATE TABLE IF NOT EXISTS records(seq INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,
                payload TEXT NOT NULL,created_utc TEXT NOT NULL,previous_hash TEXT NOT NULL,chain_hash TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON records BEGIN SELECT RAISE(ABORT,'append only'); END;
            CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON records BEGIN SELECT RAISE(ABORT,'append only'); END;''')
            meta=dict(c.execute('SELECT key,value FROM metadata'))
            if meta and meta!={'schema_version':'1','pilot_sha256':self.pilot_hash}:raise ValueError('database belongs to another pilot/schema')
            if not meta:c.executemany('INSERT INTO metadata VALUES(?,?)',[('schema_version','1'),('pilot_sha256',self.pilot_hash)])
        self.db_path.chmod(0o600);self.audit()
    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.db_path,timeout=10);c.execute('PRAGMA foreign_keys=ON')
        try:
            with c:yield c
        finally:c.close()
    def exposures(self):
        with self.connect() as c:return [{'pair_id':p,'reviewer':r,'created_utc':t} for p,r,t in c.execute('SELECT pair_id,reviewer,created_utc FROM exposures ORDER BY pair_id,reviewer')]
    def records(self,c=None):
        if c is None:
            with self.connect() as db:return self.records(db)
        return [dict(json.loads(p),id=i,created_utc=t) for i,p,t in c.execute('SELECT id,payload,created_utc FROM records ORDER BY seq')]
    def audit(self):
        previous='0'*64;count=0
        with self.connect() as c:
            for rid,payload,created,prev,chain in c.execute('SELECT id,payload,created_utc,previous_hash,chain_hash FROM records ORDER BY seq'):
                if prev!=previous or sha(payload.encode())!=rid or sha((prev+created+payload).encode())!=chain:raise ValueError('record chain corrupted')
                previous=chain;count+=1
        return {'records':count,'chain_head':previous,'pilot_sha256':self.pilot_hash,'exposures_sha256':sha(canonical(self.exposures()).encode())}
    def candidates(self,key):
        row=self.corpus[key];result=[]
        for c in self.title_index[(row['platform'],title_core(row['title']))]:
            result.append({'problem_key':c['problem_key'],'title':c['title'],'url':c['url'],'basis':'normalized title equality only; human evidence required','status':'unverified_candidate'})
        return sorted(result,key=lambda c:c['problem_key'])[:10]
    def pair(self,pid,reviewer='',mode='review'):
        p=self.pairs[pid]
        if mode not in {'review','adjudicate'}:raise ValueError('invalid view mode')
        if mode=='adjudicate':
            if not isinstance(reviewer,str) or not reviewer.strip() or len(reviewer)>100 or reviewer!=reviewer.strip():raise ValueError('trimmed reviewer required before unblinding')
            with self.connect() as c:c.execute('INSERT OR IGNORE INTO exposures VALUES(?,?,?)',(pid,reviewer,datetime.now(timezone.utc).isoformat()))
        records=[r for r in self.records() if r['pair_id']==pid]
        independent=any(r['reviewer']==reviewer and r['role']=='human' and r['kind']=='relationship' and r['independent'] for r in records)
        own_identity={r['subject_key'] for r in records if r['kind']=='identity' and r['reviewer']==reviewer and r['role']=='human' and r['independent']}
        visible=[r for r in records if mode=='adjudicate' or r['reviewer']==reviewer or (r['kind']=='relationship' and independent) or (r['kind']=='identity' and r['subject_key'] in own_identity) or (r['kind']=='adjudication' and ((r['task']=='relationship' and independent) or (r['task']=='identity' and r['decision']['subject_key'] in own_identity)))]
        publicpair=dict(p) if independent or mode=='adjudicate' else {k:v for k,v in p.items() if k not in {'methods','stratum'}}
        return {'pair':publicpair,'query':self.corpus[p['query']],'target':self.corpus[p['target']],
                'identity_candidates':{s:self.candidates(p[s]) for s in ('query','target')},'records':visible,'blind':not independent and mode!='adjudicate','unblinded_for_adjudication':mode=='adjudicate'}
    def progress(self):
        all_records=self.records();counts=Counter(r['kind'] for r in all_records)
        accepted=self.accepted_overlay(all_records)
        return {'total_pairs':len(self.pairs),'record_counts':dict(counts),'accepted_adjudications':len(accepted),
                'human_identity_authenticated':False,'pending_human_work':len(self.pairs)-len({r['pair_id'] for r in accepted if r['task']=='relationship'})}
    def _dimensions(self,r):
        sim=r.get('similarity')
        if sim is not None and (type(sim) is not int or sim not in range(4)):raise ValueError('similarity must be 0..3 or null')
        if 'similarity' not in r:raise ValueError('similarity missing')
        for key,values in CHOICES.items():
            if r.get(key) not in values:raise ValueError('invalid '+key)
    def validate(self,r,existing):
        if not isinstance(r,dict):raise ValueError('record object required')
        base={'pair_id','reviewer','role','kind','rationale','evidence'}
        kind=r.get('kind');fields={'relationship':{'similarity',*CHOICES,'independent'},'identity':{'subject_key','native_key','status','independent'},'adjudication':{'task','review_ids','outcome','decision'}}
        if kind not in fields or set(r)!=base|fields[kind]:raise ValueError('unexpected or missing fields')
        if r['pair_id'] not in self.pairs:raise ValueError('pair outside frozen pilot')
        if r['role'] not in {'human','ai'}:raise ValueError('explicit human or ai role required')
        if not isinstance(r['reviewer'],str) or not r['reviewer'].strip() or len(r['reviewer'])>100:raise ValueError('reviewer required')
        if r['reviewer']!=r['reviewer'].strip():raise ValueError('trim reviewer identity')
        if not isinstance(r['rationale'],str) or len(r['rationale'].strip())<10:raise ValueError('substantive rationale required')
        if not isinstance(r['evidence'],list) or not r['evidence'] or any(not isinstance(x,str) or not x.strip() or len(x)>4000 for x in r['evidence']):raise ValueError('nonempty evidence required')
        if kind in {'identity','relationship'} and type(r['independent']) is not bool:raise ValueError('independent boolean required')
        if kind in {'identity','relationship'} and r['independent']:
            duplicate=any(x['id']==sha(canonical(r).encode()) for x in existing)
            if not duplicate and any(x['pair_id']==r['pair_id'] and x['reviewer']==r['reviewer'] for x in self.exposures()):raise ValueError('reviewer has unblinded; independent claim forbidden')
            same=[x for x in existing if x['kind']==kind and x['pair_id']==r['pair_id'] and x['reviewer']==r['reviewer'] and x['independent'] and (kind!='identity' or x['subject_key']==r['subject_key']) and x['id']!=sha(canonical(r).encode())]
            if same:raise ValueError('only first review can claim independence; append correction as non-independent')
        if kind=='relationship':self._dimensions(r)
        if kind=='identity':
            p=self.pairs[r['pair_id']]
            if r['subject_key'] not in {p['query'],p['target']}:raise ValueError('wrong identity subject')
            if r['status'] not in {'proposed','verified','rejected'}:raise ValueError('invalid assertion status')
            c=self.catalog.get(r['native_key'])
            if c is None or c['platform']!=self.corpus[r['subject_key']]['platform']:raise ValueError('native catalog identity required on same platform')
            if c['url'] not in r['evidence']:raise ValueError('canonical native URL evidence required')
        if kind=='adjudication':
            if r['role']!='human' or r['task'] not in {'identity','relationship'} or r['outcome'] not in {'accept','reject'}:raise ValueError('human adjudication task/outcome required')
            ids=r['review_ids']
            if not isinstance(ids,list) or len(ids)!=2 or len(set(ids))!=2:raise ValueError('two distinct reviews required')
            index={x['id']:x for x in existing};reviews=[index.get(i) for i in ids]
            if any(x is None or x['kind']!=r['task'] or x['pair_id']!=r['pair_id'] or x['role']!='human' or not x['independent'] for x in reviews):raise ValueError('two independent human task reviews required')
            reviewers={x['reviewer'] for x in reviews}
            if len(reviewers)!=2 or r['reviewer'] in reviewers:raise ValueError('third distinct human adjudicator required')
            if not isinstance(r['decision'],dict):raise ValueError('decision object required')
            if r['task']=='relationship':
                if set(r['decision'])!={'similarity',*CHOICES}:raise ValueError('complete relationship decision required')
                self._dimensions(r['decision'])
            else:
                if set(r['decision'])!={'subject_key','native_key'}:raise ValueError('identity decision required')
                for k in ('subject_key','native_key'):
                    if any(x[k]!=r['decision'][k] for x in reviews):raise ValueError('identity reviews must agree with decision')
                if r['outcome']=='accept':
                    if any(x['status']=='rejected' for x in reviews):raise ValueError('rejected proposal cannot resolve identity')
                    if self.catalog[r['decision']['native_key']]['url'] not in r['evidence']:raise ValueError('canonical adjudication URL evidence required')
    def append(self,r):
        # Content idempotency excludes server-assigned timestamps.
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE');existing=self.records(c)
            self.validate(r,existing);body=canonical(r);rid=sha(body.encode())
            if any(x['id']==rid for x in existing):return {'id':rid,'duplicate':True}
            previous=c.execute('SELECT chain_hash FROM records ORDER BY seq DESC LIMIT 1').fetchone();prev=previous[0] if previous else '0'*64
            created=datetime.now(timezone.utc).isoformat()
            c.execute('INSERT INTO records(id,payload,created_utc,previous_hash,chain_hash) VALUES(?,?,?,?,?)',(rid,body,created,prev,sha((prev+created+body).encode())))
        return {'id':rid,'duplicate':False}
    @staticmethod
    def accepted_overlay(records):
        latest_reviews={};latest_decisions={}
        for r in records:
            if r['kind'] in {'relationship','identity'}:
                subject=r.get('subject_key',r['pair_id'])
                latest_reviews[(r['kind'],r['pair_id'],subject,r['reviewer'])]=r['id']
            else:
                subject=r['decision'].get('subject_key',r['pair_id'])
                latest_decisions[(r['task'],subject)]=r
        accepted=[];index={r['id']:r for r in records}
        for r in latest_decisions.values():
            if r['outcome']!='accept':continue
            reviews=[index[i] for i in r['review_ids']]
            if all(latest_reviews[(x['kind'],x['pair_id'],x.get('subject_key',x['pair_id']),x['reviewer'])]==x['id'] for x in reviews):accepted.append(r)
        return accepted
    def export(self,path):
        audit=self.audit();records=self.records()
        result={'schema_version':1,'audit':audit,'records':records,'exposures':self.exposures(),'accepted_overlay':self.accepted_overlay(records),
                'limits':'Roles/independence self-attested. Evidence is recorded, not semantically verified by software. No frozen or production assets modified.'}
        with Path(path).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
        Path(path).chmod(0o600);return result


def restore(pilot_path,export_path,new_db):
    """Restore immutable evidence to a fresh store, preserving timestamps and chain."""
    if Path(new_db).exists():raise FileExistsError('restore requires fresh database')
    data=json.loads(Path(export_path).read_text())
    if data.get('schema_version')!=1:raise ValueError('unsupported export schema')
    if data['audit']['pilot_sha256']!=sha(Path(pilot_path).read_bytes()):raise ValueError('export pilot mismatch')
    wb=Workbench(pilot_path,new_db)
    try:
        with wb.connect() as c:
            previous='0'*64
            for row in data['records']:
                payload={k:v for k,v in row.items() if k not in {'id','created_utc'}}
                wb.validate(payload,wb.records(c));body=canonical(payload)
                if sha(body.encode())!=row['id']:raise ValueError('export record hash mismatch')
                created=row['created_utc'];datetime.fromisoformat(created)
                previous_hash=sha((previous+created+body).encode())
                c.execute('INSERT INTO records(id,payload,created_utc,previous_hash,chain_hash) VALUES(?,?,?,?,?)',(row['id'],body,created,previous,previous_hash));previous=previous_hash
            for exposure in data['exposures']:
                if exposure['pair_id'] not in wb.pairs or not exposure['reviewer'].strip() or exposure['reviewer']!=exposure['reviewer'].strip():raise ValueError('invalid export exposure')
                c.execute('INSERT INTO exposures VALUES(?,?,?)',(exposure['pair_id'],exposure['reviewer'],exposure['created_utc']))
        if wb.audit()!=data['audit'] or wb.accepted_overlay(wb.records())!=data['accepted_overlay']:raise ValueError('export audit/overlay mismatch')
    except Exception:
        Path(new_db).unlink(missing_ok=True)
        raise
    return wb
